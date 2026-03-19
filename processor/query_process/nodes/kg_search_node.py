import json
import logging
import re
from typing import Any, Dict, List, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient

from processor.query_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.exceptions import StateFieldError
from processor.query_process.state import QueryGraphState
from prompts.query_prompts import ENTITY_EXTRACT_SYSTEM_PROMPT
from utils.bge_m3_embedding_util import (
    generate_hybrid_embeddings,
    get_bge_m3_embedding_model,
)
from utils.llm_util import get_llm_client
from utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    fetch_chunks_by_chunk_ids,
    get_milvus_client,
)
from utils.neo4j_util import get_neo4j_driver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = get_config()

# 常量
_ALLOWED_ENTITY_LABELS_CN: str = (
    "设备(Device)、部件(Part)、操作(Operation)、步骤(Step)、"
    "警告(Warning)、条件(Condition)、工具(Tool)"
)

_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN_THRESHOLD = 0.5
_SEED_NODE_WEIGHT = 2.0
_NBR_NODE_WEIGHT = 1.0

ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]

# Neo4j的cypher语句
_CYPHER_EXACT_SEEDS = """
    MATCH (n:Entity)
    WHERE n.item_name = $item_name AND n.name = $entity_name
    RETURN n.item_name AS item_name, n.name AS name
"""

_CYPHER_FUZZY_SEEDS = """
    MATCH (n:Entity)
    WHERE toLower(n.name) CONTAINS toLower($entity_name)
        AND n.item_name = $item_name
    RETURN n.item_name AS item_name, n.name AS name
    LIMIT $limit
"""

_CYPHER_ONE_HOP_RELATIONS = """
    MATCH (seed:Entity {name: $name, item_name: $item_name})-[r]-(nbr:Entity)
    WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name = $item_name
    RETURN
        CASE WHEN startNode(r) = seed THEN seed.name ELSE nbr.name END AS head,
        type(r) AS rel,
        CASE WHEN startNode(r) = seed THEN nbr.name ELSE seed.name END AS tail
    LIMIT $limit
"""

_CYPHER_LOOKUP_CHUNK = """
    UNWIND $weighted_nodes AS n
    MATCH (e:Entity {name: n.entity_name, item_name: n.item_name})-[:MENTIONED_IN]->(c:Chunk{item_name: n.item_name})
    WITH c,sum(n.weight) AS score,count(e) AS cnt
    RETURN c.id AS chunk_id,c.item_name AS item_name,score,cnt
    ORDER BY score DESC, cnt DESC, chunk_id ASC
    LIMIT $limit
"""


# 工具函数
def _clean_parse_llm_content(llm_response: str) -> List[str]:
    """
    清洗以及解析LLM的输出
    """
    # 1. 判断LLM的内容是否为空
    if not llm_response:
        return []

    # 2. 清洗JSON代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
    re_sub = re.sub(r"\s*```$", "", text)

    # 3. 反序列解析
    try:
        deserialized_result: Dict[str, Any] = json.loads(re_sub)
    except json.JSONDecodeError:
        logger.error(f"JSON反序列化失败，原因：{re_sub}")
        return []

    # 4. 获取提取的实体名
    entities_name = deserialized_result.get("entities", [])

    if not entities_name:
        return []
    if not isinstance(entities_name, list):
        return []

    seen = set()
    entitise_name_result = []
    for entity_name in entities_name:
        if not entity_name:
            continue
        if not isinstance(entity_name, str):
            continue
        # 判断实体名是否过长
        truncted_entity_name = truncate_entity_name_length(entity_name)
        # 去重保序
        if truncted_entity_name not in seen:
            seen.add(truncted_entity_name)
            entitise_name_result.append(truncted_entity_name)

    return entitise_name_result


def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()

    return (
        name if len(name) < _ENTITY_NAME_MAX_LENGTH else name[:_ENTITY_NAME_MAX_LENGTH]
    )


def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join([f'"{name}"' for name in item_names])
    return f"item_name in [{quoted}]"


def _clean_seed_rows(rows: List[Dict[str, Any]]) -> List[EntitySeedNode]:
    """
    清洗查询种子节点的数据
    Args:
        rows: 查询到的结果
    Returns:
        清洗后的结果
    """
    if not rows:
        return []
    cleaned_seeds = []
    for row in rows:
        item_name = row.get("item_name", "").strip()
        entity_name = row.get("name", "").strip()
        # 为模糊查询做准备
        if not item_name or not entity_name:
            continue

        cleaned_seeds.append({"item_name": item_name, "entity_name": entity_name})
    return cleaned_seeds


def _build_item_entity_pairs(
    aligned_entities_info: List[Dict[str, Any]],
) -> List[ItemEntityPair]:
    """
    从对齐后的实体详情中获取商品名+实体名的pairs并去重
    Args:
        aligned_entities_info (List[Dict[str, Any]]): 对齐后的实体详情列表
    Returns:
        List[ItemEntityPair]: 商品名+实体名的pairs列表
    """
    pairs = []
    seen = set()
    # 1. 判断aligned_entities_info是否为空
    if not aligned_entities_info:
        return pairs

    # 2. 遍历aligned_entities_info
    for entity_info in aligned_entities_info:
        item_name = entity_info.get("item_name", "").strip()
        aligned_entity_name = entity_info.get("aligned", "").strip()
        if not (item_name and aligned_entity_name):
            continue

        key = (item_name, aligned_entity_name)
        if key not in seen:
            seen.add(key)
            pairs.append(
                {
                    "item_name": item_name,
                    "entity_name": aligned_entity_name,
                }
            )

    return pairs


def _one_hop_triples_to_texts(triples: List[OneHopRelation]) -> List[str]:
    if not triples:
        return []
    docs: List[str] = []
    for tr in triples:
        it = (tr.get("item_name") or "").strip()
        h = (tr.get("head") or "").strip()
        r = (tr.get("rel") or "").strip()
        t = (tr.get("tail") or "").strip()
        if not (h and r and t):
            continue
        docs.append(f"[{it}] {h} -({r})-> {t}" if it else f"{h} -({r})-> {t}")
    return docs


class _EntityExtractor:
    """
    实体抽取器
    职责：利用LLM从查询问题中提取实体
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def _extract(self, user_query: str) -> List[str]:
        """
        根据用户问题提取当前问题下的实体名
        """
        # 1. 获取LLM客户端
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            return []

        # 2. 获取prompt
        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            allowed_entity_labels_cn=_ALLOWED_ENTITY_LABELS_CN,
            MAX_ENTITY_NAME_LENGTH=_ENTITY_NAME_MAX_LENGTH,
        )

        # 3. 调用LLM
        try:
            response = llm_client.invoke(
                [
                    SystemMessage(content=entities_name_extract_system_prompt),
                    HumanMessage(content=f"用户的问题是：{user_query}"),
                ]
            )

            # 4. 获取响应结果
            response_content = getattr(response, "content", "").strip()

            # 5. 清洗和解析
            entities_name = _clean_parse_llm_content(response_content)

            return entities_name
        except Exception as e:
            self._logger.error(f"调用LLM失败：{e}")
            return []


class _EntityAligner:
    """
    实体对齐器
    职责：将查询问题中的实体名与知识图谱中的实体名进行对齐，对齐后的实体名能够查询Neo4J
    """

    def __init__(self, collection_name):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name

    def _align(self, entity_names: List[str], item_names: List[str]) -> Dict[str, Any]:
        """
        Args:
            entity_names (List[str]): LLM提取出的查询问题中的实体名
            item_names (List[str]): 数据库中的商品名
        Returns:
            Dict[str, Any]: {
                'entities_aligned_name': [所有对齐后的实体名],
                'entities_aligned_elements': [所有对齐后的实体信息(source_id、distance、origin、aligned、content)]
            }
        """
        fallback_result = {"entities_aligned_name": [], "entities_aligned_elements": []}

        # 1. 判断是否有实体名
        if not entity_names:
            return fallback_result

        # 2. 获取嵌入模型和客户端
        embedding_model = get_bge_m3_embedding_model()
        if not embedding_model:
            self._logger.error("嵌入模型不存在")
            return fallback_result

        milvus_client = get_milvus_client()
        if not milvus_client:
            self._logger.error("Milvus客户端不存在")
            return fallback_result

        # 3. 向量化实体名
        entity_embeddings = generate_hybrid_embeddings(embedding_model, entity_names)
        if entity_embeddings is None:
            self._logger.error("嵌入结果无法获取")
            return fallback_result

        embedding_result_dense = entity_embeddings["dense"]
        embedding_result_sparse = entity_embeddings["sparse"]

        # 4. 搜索
        expr = _item_name_filter_expr(item_names)

        # 5. 遍历所有的实体名
        aligned_entities: List[str] = []
        entity_elements: List[Dict[str, Any]] = []  # 存储所有实体的详细信息
        seen = set()

        for index, entity_name in enumerate(entity_names):
            # 5.1 对齐一个实体的名字
            align_one_result = self._align_one(
                milvus_client,
                entity_name,
                self._collection_name,
                expr,
                embedding_result_dense,
                embedding_result_sparse,
                index,
            )
            # 5.2 将对齐的实体存储到最终结果种
            entity_elements.extend(align_one_result)
            # 5.3 遍历商品下的对齐结果
            for detail in align_one_result:
                aligned_name = detail.get("aligned")
                item_name = detail.get("item_name")
                if aligned_name:
                    # 去重
                    key = (item_name, aligned_name)
                    if key not in seen:
                        seen.add(key)
                        aligned_entities.append(aligned_name)

        self._logger.info(
            f"对齐后的实体数：{len(aligned_entities)}，对齐后的实体名字:{aligned_entities}"
        )
        return {
            "entities_aligned_name": aligned_entities,
            "entities_aligned_elements": entity_elements,
        }

    def _align_one(
        self,
        milvus_client: MilvusClient,
        entity_name: str,
        collection_name: str,
        expr: str,
        embedding_result_dense: List,
        embedding_result_sparse: List,
        index: int,
    ) -> List[Dict[str, Any]]:
        """
        对齐指定实体名字
        """
        # 1. 获取实体的稠密和稀疏向量
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]
        if not dense_vector or not sparse_vector:
            return [
                {
                    "original": entity_name,
                    "aligned": "",
                    "context": "",
                    "reason": "vector不存在",
                }
            ]

        # 2. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector, sparse_vector, expr=expr
        )

        # 3. 执行混合搜索请求
        res = execute_hybrid_search_query(
            milvus_client,
            collection_name,
            hybrid_search_requests,
            ranker_weights=(0.6, 0.4),
            norm_score=True,
            output_fields=["source_chunk_id", "item_name", "context", "entity_name"],
        )

        # 4. 解析结果
        hits = res[0] if res else []
        if not hits:
            return [
                {
                    "original": entity_name,
                    "aligned": "",
                    "score": None,
                    "reason": "搜索结果为空",
                }
            ]

        # 4.1 按item_name分组，每组取最高分
        best_by_item: Dict[str, Dict] = {}
        for hit in hits:
            # a) 获取实体
            entity = hit.get("entity")
            # b) 从实体中获取商品名
            item_name = entity.get("item_name").strip()
            # c) 只保留每个item_name下的最高分
            if item_name not in best_by_item:
                best_by_item[item_name] = hit

        # 4.2 是否有最好的item_name
        if not best_by_item:
            return [
                {
                    "original": entity_name,
                    "aligned": "",
                    "score": None,
                    "reason": "no_valid_item_name",
                }
            ]

        # 4.3 item_name分组输出结果，过滤低于阈值的对象
        results: List[Dict[str, Any]] = []
        for item_name, best in best_by_item.items():
            # a) 获取最好的分数
            score = best.get("distance", 0.0)
            # b) 判断分数值
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN_THRESHOLD):
                continue
            # c) 获取实体信息
            ent = best.get("entity")
            # d) 添加到结果
            results.append(
                {
                    "original": entity_name,
                    "aligned": ent.get("entity_name"),
                    "score": score,
                    "item_name": item_name,
                    "source_chunk_id": ent.get("source_chunk_id"),
                    "reason": "top1_per_item_name",
                }
            )

        # 4.4 全部低于阈值时返回未命中
        if not results:
            return [
                {
                    "original": entity_name,
                    "aligned": "",
                    "score": None,
                    "reason": "所有匹配项得分低于阈值",
                }
            ]

        return results


class _Neo4jGraphReader:
    """
    负责所有对Neo4j的操作
    1. 种子节点查询
        1.1 精确查询
        1.2 模糊查询（兜底）
    2. 查询种子节点一跳关系（双向）
    3. 根据所有的节点（种子节点和邻居节点）反向查询chunk(id, item_name)
    4. 根据所有的chunk_id查询Milvus得到所有的chunk
    """

    def __init__(
        self,
        database,
        kg_max_seed_candidates: int,
        kg_max_total_seeds: int,
        kg_max_triples_per_seed: int,
        kg_max_total_triples: int,
        kg_max_total_chunks: int,
    ) -> None:
        self._database = database
        self._kg_max_seed_candidates = kg_max_seed_candidates
        self._kg_max_total_seeds = kg_max_total_seeds
        self._kg_max_triples_per_seed = kg_max_triples_per_seed
        self._kg_max_total_triples = kg_max_total_triples
        self._kg_max_total_chunks = kg_max_total_chunks
        self._logger = logging.getLogger(__name__)

    def _session(self):
        neo4j_driver = get_neo4j_driver()

        if not neo4j_driver:
            raise RuntimeError("Neo4j驱动获取失败")

        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self, pairs: List[ItemEntityPair]) -> List[EntitySeedNode]:
        """
        根据商品名和实体名查询种子节点
        策略：
            1. 精确查询 -> 一条
            2. 模糊查询（兜底） -> 三条
        Args:
            pairs (List[ItemEntityPair]): _build_item_entity_pairs方法返回的商品名+实体名的pairs列表
        Returns:
            List[EntitySeedNode]: 所有商品名下所有实体名对应的种子节点
        """
        # 1. 判断pairs是否有值
        if not pairs:
            return []

        # 2. 遍历pairs
        final_seeds_result: List[EntitySeedNode] = []
        for pair in pairs:
            # 2.1 获取item_name和entity_name
            item_name = pair.get("item_name", "").strip()
            entity_name = pair.get("entity_name", "").strip()

            # 2.2 过滤掉无效信息
            if not item_name or not entity_name:
                continue

            # 2.3 执行cypher语句（精确查询和模糊查询）
            try:
                with self._session() as session:
                    # 执行种子节点查询
                    candidates_seed_nodes = self._execute_seed_nodes(
                        session, item_name, entity_name, self._kg_max_seed_candidates
                    )

                    final_seeds_result.extend(candidates_seed_nodes)
                    # 截取种子节点个数防止下游查询关系时性能太差
                    if len(final_seeds_result) > self._kg_max_total_seeds:
                        final_seeds_result = final_seeds_result[
                            : self._kg_max_total_seeds
                        ]
                        break

            except Exception as e:
                self._logger.error(f"获取种子节点失败，原因：{str(e)}")
                continue

        self._logger.info(f"共获取到{len(final_seeds_result)}个种子节点")
        return final_seeds_result

    def _execute_seed_nodes(
        self, session, item_name, entity_name, _kg_max_seed_candidates: int
    ) -> List[EntitySeedNode]:
        # 1. 精确查询
        exact_rows = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_EXACT_SEEDS, item_name=item_name, entity_name=entity_name
            ).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)

        # 2. 降级，模糊查询
        fuzzy_rows = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_FUZZY_SEEDS,
                item_name=item_name,
                entity_name=entity_name,
                limit=_kg_max_seed_candidates,
            ).data()
        )
        return _clean_seed_rows(fuzzy_rows)

    def find_one_hop_relations(
        self, seed_nodes: List[EntitySeedNode]
    ) -> List[OneHopRelation]:
        """
        根据种子节点查询所有一跳关系（双向），去重后过滤掉MENTIONED_IN关系的节点(TODO)
        Returns:
            List[OneHopRelation]: [{
                'item_name': 商品名,
                'head': 头
                'rel': 关系
                'tail': 尾
                }]
        """
        # 1. 判断
        if not seed_nodes:
            return []

        # 2. 遍历所有种子节点
        seen = set()
        one_hop_relations_final_result = []
        for seed_node in seed_nodes:
            # 2.1 提取item_name
            item_name = seed_node.get("item_name", "")
            # 2.2 提取对齐后的entity_name
            seed_name = seed_node.get("entity_name", "")
            if not item_name or not seed_name:
                continue
            # 2.3 执行cypher语句
            try:
                with self._session() as session:
                    # 查询种子节点的一跳关系
                    seed_one_hop_relations = self._execute_one_hop_relations(
                        session, item_name, seed_name, self._kg_max_triples_per_seed
                    )
                    if not seed_one_hop_relations:
                        continue
                    # 遍历所有种子节点的所有关系
                    for rel in seed_one_hop_relations:
                        head = rel.get("head")
                        rel_type = rel.get("rel")
                        tail = rel.get("tail")
                        item_name = rel.get("item_name")
                        # 去重，同一个商品下不允许重复
                        key = (item_name, head, rel_type, tail)
                        if key not in seen:
                            seen.add(key)
                            one_hop_relations_final_result.append(rel)
                    # 截取种子节点的关系，防止超过LLM窗口阈值
                    if len(one_hop_relations_final_result) > self._kg_max_total_triples:
                        one_hop_relations_final_result = one_hop_relations_final_result[
                            : self._kg_max_total_triples
                        ]

            except Exception as e:
                self._logger.error(f"查询{seed_name}一跳关系失败，原因{str(e)}")
                continue

        self._logger.info(
            f"查询{len(seed_nodes)}个种子节点成功，共{len(one_hop_relations_final_result)}条"
        )
        return one_hop_relations_final_result

    def _execute_one_hop_relations(
        self, session, item_name: str, seed_name: str, kg_max_triples_per_seed: int
    ) -> List[OneHopRelation]:
        """
        Returns:
            List[OneHopRelation]: 种子节点的关系
        """
        # 1. 根据session执行查询方法
        one_hop_relations = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_ONE_HOP_RELATIONS,
                item_name=item_name,
                name=seed_name,
                limit=kg_max_triples_per_seed,
            ).data()
        )

        # 2. 解析结构
        if not one_hop_relations:
            return []

        # 3. 遍历所有的一跳关系
        one_hop_relations_result = []
        for rel in one_hop_relations:
            head = rel.get("head")
            rel_type = rel.get("rel")
            tail = rel.get("tail")
            # 判断是否存在关系链
            if not (head and rel_type and tail):
                continue

            one_hop_relations_result.append(
                {
                    "head": head,
                    "rel": rel_type,
                    "tail": tail,
                    "item_name": item_name,
                }
            )

        return one_hop_relations_result

    def collect_node_weight(
        self, seed_nodes: List[Dict[str, Any]], one_hop_relations: List[OneHopRelation]
    ) -> List[Dict[str, Any]]:
        """
        为种子节点设置权重（高），为邻居节点设置权重（低）
        Returns:
            List[Dict[str, Any]]: 带权重的节点
        """
        # 1. 判断种子节点是否存在
        if not seed_nodes:
            return []

        # 2. 判断一跳关系是否存在
        if not one_hop_relations:
            return []

        # 3. 遍历种子节点
        seen = set()
        weight_map: Dict[Tuple[str, str], float] = {}  # 存放所有节点的权重
        for seed_node in seed_nodes:
            # 3.1 获取
            item_name = seed_node.get("item_name")
            seed_name = seed_node.get("entity_name")

            key = (item_name, seed_name)
            if key not in seen:
                seen.add(key)
                # 3.2 设置种子节点权重
                weight_map[key] = _SEED_NODE_WEIGHT

        # 4. 遍历一跳四元组
        for rel in one_hop_relations:
            # 4.1 获取
            head = rel.get("head")
            tail = rel.get("tail")
            item_name = rel.get("item_name")
            # 4.2 排除种子节点，设置邻居节点权重
            if head and (item_name, head) not in weight_map:
                weight_map[(item_name, head)] = _NBR_NODE_WEIGHT

            if tail and (item_name, tail) not in weight_map:
                weight_map[(item_name, tail)] = _NBR_NODE_WEIGHT

        return [
            {"item_name": it, "entity_name": en, "weight": w}
            for (it, en), w in weight_map.items()
        ]

    def find_nodes_chunk_id(
        self, weighted_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        根据带权重的节点反查chunk_id，并且基于权重给chunk排序(权重降序->次数降序->chunk_id升序)
        Returns:
            List[Dict[str, Any]]: 带chunk_id的节点
        """
        # 1. 执行cypher
        try:
            with self._session() as session:
                sorted_node_chunk_id = session.execute_read(
                    lambda tx: tx.run(
                        _CYPHER_LOOKUP_CHUNK,
                        weighted_nodes=weighted_nodes,
                        limit=self._kg_max_total_chunks,
                    ).data()
                )

        except Exception as e:
            self._logger.error(f"反查chunk_id失败，原因{str(e)}")
            return []

        # 2. 处理结果
        hits = []
        for chunk_row in sorted_node_chunk_id:
            chunk_id = chunk_row.get("chunk_id", "").strip()
            item_name = chunk_row.get("item_name", "").strip()
            score = chunk_row.get("score")

            if chunk_id and item_name:
                hits.append(
                    {
                        "id": None,
                        "distance": float(score or 0.0),
                        "entity": {
                            "chunk_id": str(chunk_id),
                            "item_name": str(item_name),
                        },
                    }
                )

        return hits


class _ChunkBackFiller:
    """
    1. 根据Neo4j返回的chunk信息获取chunk_ids(entity('chunk_id'))
    2. 根据chunk_ids查询Milvus获取到chunks对象 (批量操作，返回的chunk没有顺序)
    3. 构建映射表将Milvus返回的chunk_id映射到chunk对象
    4. 遍历原有分数降序的chunk_id列表，从映射表中获取对应的chunk
    4. 更新到state
    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def back_fill(
        self, chunk_nodes_sorted: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        1. 获取所有chunk_id
        2. 根据批量的chunk_id查询Milvus
        3. 构建{chunk_id : chunk}对象映射表
        4. 遍历原有的chunk_id列表，从映射表中获取对应的chunk对象

        Args:
            chunk_nodes_sorted: 排好序的chunk节点
        Returns:
            List[Dict[str, Any]]:
        """
        # 1. 判断
        if not chunk_nodes_sorted:
            return []

        # 2. 获取chunk_ids
        chunk_ids: List[Union[str, int]] = self._collect_chunk_ids(chunk_nodes_sorted)

        # 3. 根据chunk_ids查询Milvus
        try:
            chunks: List[Dict[str, Any]] = fetch_chunks_by_chunk_ids(
                collection_name=self._collection_name,
                chunk_ids=chunk_ids,
                output_fields=[
                    "chunk_id",
                    "content",
                    "title",
                    "file_title",
                    "item_name",
                ],
                batch_size=30,
            )
            if not chunks:
                return []
        except Exception as e:
            self.logger.error(f"根据chunk_id批量查询chunk对象失败：{str(e)}")
            return []

        # 4. 构建映射表{chunk_id: chunk}TODO
        chunk_id_map = {
            str(chunk.get("chunk_id")): chunk
            for chunk in chunks
            if chunk.get("chunk_id") is not None
        }

        # 5. 根据真实顺序的chunk_id查询对应的chunk
        return [{'entity':chunk_id_map.get(str(chunk_id))} for chunk_id in chunk_ids]

    def _collect_chunk_ids(
        self, chunk_nodes_sorted: List[Dict[str, Any]]
    ) -> List[Union[str, int]]:
        # 1. 遍历chunk_ids
        chunk_ids = []
        for chunk_node in chunk_nodes_sorted:
            if not chunk_node:
                continue
            # 2. 获取entity
            entity = chunk_node.get("entity", "")
            if not entity:
                continue
            # 3. 获取chunk_id
            chunk_id = entity.get("chunk_id")
            if not chunk_id:
                continue
            # 4. chunk_id转换
            try:
                chunk_ids.append(int(str(chunk_id)))
            except (ValueError, TypeError):
                chunk_ids.append(str(chunk_id))

        return chunk_ids


class KnowledgeGraphSearchNode(BaseNode):
    """
    知识图谱查询主编排器。

    职责：
    - 组装四个服务组件（Extractor / Aligner / GraphReader / Backfiller）
    - 按 pipeline 顺序编排调用

    Pipeline:
    ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
       抽取实体  ──▶   对齐实体   ──▶    Neo4j查询   ──▶ 回填chunk
    └──────────┘   └──────────┘   └────────────┘   └──────────┘
    """

    name = "knowledge_graph_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_input(state)

        # 2. 执行流水线
        kg_result: Dict[str, Any] = self._run_pipeline(
            validated_query, validated_item_names
        )

        # 3. 更新状态
        state["kg_chunks"] = kg_result.get("kg_chunks", [])
        state["kg_triples"] = kg_result.get("kg_triples", [])
        return state

    def _validate_input(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """"""
        # 1. 获取参数
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name, field_name="rewritten_query", expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name, field_name="item_names", expected_type=list
            )

        # 3. 从rewritten_query中剔除商品名TODO
        user_query = rewritten_query
        for name in item_names:
            if not name:
                continue
            pattern = r"\s*".join(re.escape(ch) for ch in name.replace(" ", ""))
            user_query = re.sub(pattern, "", user_query, flags=re.IGNORECASE)

        user_query = " ".join(user_query.split()).strip()

        return user_query, item_names

    def _run_pipeline(
        self, validated_query: str, validated_item_names: List[str]
    ) -> Dict[str, Any]:
        """"""
        # 1. 初始化组件
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(config.entity_name_collection)
        neo4j_graph_reader = _Neo4jGraphReader(
            database=config.neo4j_database,
            kg_max_seed_candidates=config.kg_max_seed_candidates,
            kg_max_total_seeds=config.kg_max_total_seeds,
            kg_max_triples_per_seed=config.kg_max_triples_per_seed,
            kg_max_total_triples=config.kg_max_total_triples,
            kg_max_total_chunks=config.kg_max_total_chunks,
        )
        chunk_back_filler = _ChunkBackFiller(collection_name=config.chunks_collection)

        # 2. 执行实体抽取
        entities_name = entity_extractor._extract(validated_query)
        entities_aligned_name: Dict[str, Any] = entity_aligner._align(
            entities_name, validated_item_names
        )
        # 2.1 获取所有对齐后的实体名
        aligned_entities_name = entities_aligned_name.get("entities_aligned_name")
        # 2.2 获取所有对齐后的实体信息
        entities_aligned_elements = entities_aligned_name.get(
            "entities_aligned_elements"
        )

        # 3. 构建商品名+实体名的pairs
        item_entity_pairs: List[ItemEntityPair] = _build_item_entity_pairs(
            entities_aligned_elements
        )

        # 4. Neo4j操作
        # 4.1 根据pairs查询知识图谱
        seed_nodes: List[EntitySeedNode] = neo4j_graph_reader.find_seed_nodes(
            item_entity_pairs
        )

        # 4.2 根据种子节点查询一跳的关系
        one_hop_relations: List[OneHopRelation] = (
            neo4j_graph_reader.find_one_hop_relations(seed_nodes)
        )

        # 4.3 根据种子节点id一跳关系,分别为其设置权重
        weighted_nodes: List[Dict[str, Any]] = neo4j_graph_reader.collect_node_weight(
            seed_nodes, one_hop_relations
        )

        # 4.4 根据带权重的节点反查chunk，并且基于权重给chunk排序(权重降序->次数降序->chunk_id升序)
        chunk_nodes_sorted: List[Dict[str, Any]] = (
            neo4j_graph_reader.find_nodes_chunk_id(weighted_nodes)
        )

        # 5. Milvus操作：回填chunk_id
        kg_chunks = chunk_back_filler.back_fill(chunk_nodes_sorted)

        # 6. 将一跳关系转换成模型容易理解的真实图谱结构
        triples_docs = _one_hop_triples_to_texts(one_hop_relations)

        # 7. 汇总知识图谱节点的所以信息
        return {
            "kg_chunks": kg_chunks,  # 回填后的切片文本 → 送入 RRF
            "kg_triples": triples_docs,  # 关系文本描述 → 送入答案生成 prompt
            "kg_seed_nodes": seed_nodes,
            "kg_triples_raw": one_hop_relations,
            "kg_entities": entities_name,
            "kg_aligned_entities": aligned_entities_name,
            "kg_alignments": entities_aligned_elements,
        }


if __name__ == "__main__":
    state = QueryGraphState(
        {
            "rewritten_query": "华为擎云B730的操作步骤是什么",
            "item_names": ["华为擎云B730 台式计算机"],
        }
    )
    node = KnowledgeGraphSearchNode()
    result = node.process(state)
    print(result)
