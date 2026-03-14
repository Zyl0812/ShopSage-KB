import json
import logging

import re
from typing import Any, List, Dict, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient

from utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from utils.bge_me_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from utils.llm_util import get_llm_client
from processor.query_process.exceptions import StateFieldError
from processor.query_process.config import get_config
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode
from prompts.query_prompts import ENTITY_EXTRACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = get_config()

# 常量
_ALLOWED_ENTITY_LABELS_CN = "TODO"

_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN_THRESHOLD = 0.5

# 工具函数
def _clean_parse_llm_content(llm_response: str) -> List[str]:
    '''
    清洗以及解析LLM的输出
    '''
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
        logger.error(f'JSON反序列化失败，原因：{re_sub}')
        return []
        
    # 4. 获取提取的实体名
    entities_name = deserialized_result.get('entities', [])
    
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
    
    return name if len(name) < _ENTITY_NAME_MAX_LENGTH else name[:_ENTITY_NAME_MAX_LENGTH]
        


class _EntityExtractor:
    '''
    实体抽取器
    职责：利用LLM从查询问题中提取实体
    '''
    def __init__(self):
        self._logger=logging.getLogger(__name__)
    
    def _extract(self, user_query: str) -> List[str]:
        '''
        根据用户问题提取当前问题下的实体名
        '''
        # 1. 获取LLM客户端
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            return []
        
        # 2. 获取prompt
        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(allowed_entity_labels_cn=_ALLOWED_ENTITY_LABELS_CN,MAX_ENTITY_NAME_LENGTH=_ENTITY_NAME_MAX_LENGTH)
        
        # 3. 调用LLM
        try:
            response = llm_client.invoke([
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f'用户的问题是：{user_query}'),
            ])
            
            # 4. 获取响应结果
            response_content = getattr(response, 'content', '').strip()
            
            # 5. 清洗和解析
            entities_name = _clean_parse_llm_content(response_content)
            
            return entities_name
        except Exception as e:
            self._logger.error(f'调用LLM失败：{e}')
            return []

def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ', '.join([f'"{name}"' for name in item_names])
    return f'item_name in [{quoted}]'


class _EntityAligner:
    '''
    实体对齐器
    职责：将查询问题中的实体名与知识图谱中的实体名进行对齐，对齐后的实体名能够查询Neo4J
    '''
    def __init__(self, collection_name):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name
    
    
    def _align(self, entity_names: List[str], item_names: List[str]) -> Dict[str, Any]:
        '''
        Args:
            entity_names (List[str]): LLM提取出的查询问题中的实体名
            item_names (List[str]): 数据库中的商品名
        Returns:
            Dict[str, Any]: {
                'entities_aligned': [所有对齐后的实体名],
                'entity_elements': [所有对齐后的实体信息(source_id、distance、origin、aligned、content)]
            }
        '''
        fallback_result = {'entities_aligned': [], 'entity_info': []}
        
        # 1. 判断是否有实体名
        if not entity_names:
            return fallback_result
        
        # 2. 获取嵌入模型和客户端
        embedding_model = get_bge_m3_embedding_model()
        if not embedding_model:
            self._logger.error('嵌入模型不存在')
            return fallback_result
            
        milvus_client = get_milvus_client()
        if not milvus_client:
            self._logger.error('Milvus客户端不存在')
            return fallback_result
        
        # 3. 向量化实体名
        entity_embeddings = generate_hybrid_embeddings(embedding_model, entity_names)
        if entity_embeddings is None:
            self._logger.error('嵌入结果无法获取')
            return fallback_result
        
        embedding_result_dense = entity_embeddings['dense']
        embedding_result_sparse = entity_embeddings['sparse']
        
        # 4. 搜索
        expr = _item_name_filter_expr(item_names)
        
        # 5. 遍历所有的实体名
        aligned_entities: List[str] = []
        entity_elements: List[Dict[str, Any]] = []  # 存储所有实体的详细信息
        for index, entity_name in enumerate(entity_names):
            # 5.1 对齐一个实体的名字
            align_one_result = self._align_one(milvus_client, entity_name, self._collection_name, expr, embedding_result_dense, embedding_result_sparse, index)
            # 5.2 构建所有实体的名字
            aligned_entity_name = align_one_result.get('aligned', '')
            aligned_entities.append(aligned_entity_name)
            # 5.3 构建对齐后的实体详细信息
            entity_elements.append(align_one_result)
        
        self._logger.info(f'对齐后的实体数：{len(aligned_entities)}，对齐后的实体名字:{aligned_entities}')
        return {
            'entities_aligned': aligned_entities,
            'entity_elements': entity_elements,
        }
    
    def _align_one(self, milvus_client: MilvusClient, entity_name: str, collection_name: str, expr: str, embedding_result_dense: List, embedding_result_sparse: List, index: int) -> Dict[str, Any]:
        '''
        对齐指定实体名字
        '''
        # 1. 获取实体的稠密和稀疏向量
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]
        if not dense_vector or not sparse_vector:
            return {'original': entity_name, 'aligned': '', 'context': '', 'reason': 'vector不存在'}
        
        # 2. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(dense_vector, sparse_vector, expr=expr)
        
        # 3. 执行混合搜索请求
        res = execute_hybrid_search_query(
            milvus_client, 
            collection_name,
            hybrid_search_requests,
            ranker_weights=(0.6, 0.4),
            norm_score=True,
            output_fields=['source_chunk_id', 'item_name', 'context', 'entity_name'],
        )
        
        # 4. 判断结果是否存在
        if not res or not res[0]:
            return {'original': entity_name, 'aligned': '', 'context': '', 'reason': '搜索结果为空'}
        
        # 5. 解析最佳结果
        best_entity = self._pick_best_entity_name(res[0])
        if best_entity is None:
            return {'original': entity_name, 'aligned': '', 'context': '', 'reason': '没有最佳结果'}
        
        
        # 6. 返回结果
        return {
            'original': entity_name, 
            'aligned': best_entity['entity_name'],
            'source_chunk_id': best_entity['source_chunk_id'],
            'item_name': best_entity['item_name'],
            'context': best_entity['context'],
            'reason': 'top1'
        }
    
    def _pick_best_entity_name(self, results: List[Dict[str, Any]]):
        '''
        从五个搜索结果中选择最佳的实体名
        '''
        if not results:
            return None
        
        # 获取第一个
        first_entity = results[0]
        if not first_entity:
            return None
        
        distance = float(first_entity.get('distance', 0.0))
        
        # 判断得分是否超过阈值
        return first_entity if distance > _DEFAULT_ENTITY_NAME_ALIGN_THRESHOLD else None



class KnowledgeGraphSearchNode(BaseNode):
    name = 'knowledge_graph_search_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_input(state)
        
        # 2. 执行流水线
        result = self._run_pipeline(validated_query, validated_item_names)
        
        return result
   
    
    def _validate_input(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        ''''''
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')
        
        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
        
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)
        
        # 3. 从rewritten_query中剔除商品名
        for item_name in item_names:
            rewritten_query = rewritten_query.replace(item_name, '')
        
        return rewritten_query, item_names
        
        
    def _run_pipeline(self, validated_query: str, validated_item_names: List[str]):
        ''''''
        # 1. 初始化组件
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(config.entity_name_collection)
        
        # 2. 执行实体抽取
        entities_name = entity_extractor._extract(validated_query)
        entities_aligned_name: Dict[str, Any] = entity_aligner._align(entities_name, validated_item_names)
        
        return entities_aligned_name
    
if __name__ == '__main__':
    state = QueryGraphState({'rewritten_query': '华为擎云B730的操作步骤是什么', 'item_names': ['华为擎云B730 台式计算机']})
    node = KnowledgeGraphSearchNode()
    result = node.process(state)
    print(result)