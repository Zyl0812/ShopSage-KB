import logging

from typing import List, Tuple


from processor.query_process.exceptions import StateFieldError
from utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from utils.bge_me_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorSearchNode(BaseNode):
    name = 'vector_search_node'
    
    def process(self, state: QueryGraphState) -> QueryGraphState:
        
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_query_inputs(state)
        
        # 2. 获取嵌入模型以及Milvus客户端
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()
        
        if embedding_model is None or milvus_client is None:
            return state
        
        # 3. 对问题进行向量化
        embedding_result = generate_hybrid_embeddings(embedding_model, [validated_query])
        if not embedding_result:
            return state
        
        # 4. 构建过滤表达式
        item_name_filter_expr = f'item_name in {validated_item_names}'
        
        # 4. 创建混合搜索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filter_expr
        )
        
        # 5. 执行混合搜索
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            norm_score=True,
            output_fields=['chunk_id', 'content', 'item_name'],
        )
        
        if not res or not res[0]:
            return state
        
        # 6. 更新state
        state['embedding_chunks'] = res[0]
    
        return state
    
    
    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        
        rewrritten_query = state.get('rewritten_query', '')
        item_names = state.get('item_names', [])
        
        if not rewrritten_query or not isinstance(rewrritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
            
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)
        
        return rewrritten_query, item_names
        

# if __name__ == '__main__':
#     import json
    
#     state = QueryGraphState({
#         'rewritten_query': '华为擎云B730如何使用',
#         'item_names': ['华为擎云B730 台式计算机'],
#     })
    
#     node = VectorSearchNode()
#     result = node.process(state)
    
#     for r in result.get('embedding_chunks', []):
#         print(json.dumps(r, ensure_ascii=False, indent=2))