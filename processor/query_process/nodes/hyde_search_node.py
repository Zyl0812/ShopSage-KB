import logging

from typing import List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_process.exceptions import StateFieldError
from utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from utils.bge_me_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from utils.llm_util import get_llm_client
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode
from processor.query_process.prompts.kg_query_prompt import USER_HYDE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HyDeSearchNode(BaseNode):
    name = 'hyde_search_node'
    
    def process(self, state: QueryGraphState) -> QueryGraphState:
        
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_query_inputs(state)
        
        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validated_item_names)
    
        # 3. 对假设性文档嵌入
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()
        
        if not embedding_model or not milvus_client:
            return state
        
        # 4. 假设性文档嵌入（注入问题+假设性文档）
        embedding_document = f'{validated_query}\n{hy_document}'
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_document)
        
        if not embedding_result:
            return state
        
        # 5. 创建混合搜索请求
        item_name_filter_expr = f'item_name in {validated_item_names}'
        hybird_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filter_expr
        )
        
        # 6. 执行混合搜索
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybird_search_requests,
            norm_score=True,
            output_fields=['chunk_id', 'content', 'item_name']
        )
        if not res or not res[0]:
            return state
        
        state['hyde_embedding_chunks'] = res[0]
        return state
    
    
    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        
        rewrritten_query = state.get('rewritten_query', '')
        item_names = state.get('item_names', [])
        
        if not rewrritten_query or not isinstance(rewrritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
            
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)
        
        return rewrritten_query, item_names
    
    
    def _generate_hy_document(self, query: str, item_names: List[str]) -> str:
        # 1. 获取LLM客户端
        llm_client = get_llm_client()
        if llm_client is None:
            return ''
            
        # 2. 获取系统提示词以及用户提示词
        user_prompt = USER_HYDE_PROMPT_TEMPLATE
        system_prompt = f'您是一位{item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明'
        
        # 3. 获取AImessage
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        # 4. 获取内容
        llm_response_content = getattr(llm_response, 'content', '').strip()
        if not llm_response_content:
            return ''
        
        return llm_response_content
