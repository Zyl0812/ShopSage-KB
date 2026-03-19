from typing import Any, DefaultDict, Dict, List, Tuple

from processor.query_process.base import BaseNode
from processor.query_process.state import QueryGraphState


class RRFNode(BaseNode):
    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrk_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1. 拿到各路搜索的结果（排除网络搜索）---> 网络搜索的结果没有chunk_id
        vector_search_chunks = state.get("embedding_chunks") or []
        hyde_search_chunks = state.get("hyde_embedding_chunks") or []
        kg_search_chunks = state.get("kg_chunks") or []

        # 2. 为不同路的搜索结果设置权重
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
            "kg_search_result": (self._normalize_input(kg_search_chunks), 0.7),
        }
        # 3. 构建列表
        rrf_inputs = list(search_source.values())

        # 4. 利用RRF计算公式获取所有路查询到的chunk对应得分
        merge_result: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(
            rrf_inputs, self._rrk_k, self._top_k
        )

        # 5. 获取rrf_chunks 只取文档，不要分数
        rrf_chunks = [doc for doc, _ in merge_result]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if merge_result:
            scores = [s for _, s in merge_result]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新state
        state["rrf_chunks"] = rrf_chunks

        return state

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一处理各路检索到的结果
        Args:
            rrf_input: 各路搜索结果的列表，每个元素是一个字典，包含chunk_id和其他信息
        Returns:
            处理后的标准结果列表，每个元素是一个字典，包含chunk_id和其他信息
        """
        diff_path_result = []
        for doc in rrf_input:
            # 1. 判断结构是否有效
            if not isinstance(doc, dict):
                continue
            # 2. 获取entity
            entity = doc.get("entity")
            if not entity:
                continue

            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(
        self,
        rrf_inputs: List[Tuple[List[Dict[str, Any]], float]],
        rrf_k: int,
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用RRF计算公式合并各路检索结果，对得分排序
        Args:
            rrf_inputs: 各路搜索结果的列表，每个元素是一个元组，包含chunk列表和权重
            rrf_k: RRF公式中的k值
            top_k: 返回的结果数量
        Returns:
            合并以及排序后的结果列表，每个元素是一个元组
        """
        chunk_score = DefaultDict(float)  # 存放chunk对应的得分
        chunk_map = {}  # 存放所有chunk对象

        for rrf_input, weight in rrf_inputs:
            # 遍历三条输入来源
            for idx, doc in enumerate(rrf_input, 1):
                # 遍历某一路中的所有chunk
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue

                chunk_score[chunk_id] += weight / (idx + rrf_k)
                chunk_map.setdefault(chunk_id, doc)

        # 按得分降序排序并截取前top_k个值
        sorted_results = sorted(
            [(chunk_map[chunk_id], score) for chunk_id, score in chunk_score.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:top_k] if top_k else sorted_results


if __name__ == "__main__":
    

    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)
    
    # 模拟三路检索结果
    # chunk_1 命中 3 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4, chunk_5 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
        "kg_chunks": [
            {"id": None, "distance": 2.0, "entity": {"chunk_id": "chunk_5", "content": "知识图谱结果#1"}},
            {"id": None, "distance": 1.0, "entity": {"chunk_id": "chunk_1", "content": "知识图谱结果#2"}},
        ],
    }
    
    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)
    
    rrf_node = RRFNode()
    result = rrf_node.process(mock_state)
    
    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")
        
    print("-" * 60)
    print("测试完成")
    