import json
import re
import logging

from typing import Any, Dict, List, Tuple
from json.decoder import JSONDecodeError
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_util import get_llm_client
from utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from utils.bge_me_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.prompts.item_name_extract_prompt import ITEM_NAME_EXTRACT_TEMPLATE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ItemNameAligner():
    '''
    主要职责：
    1. 查询向量数据库
    2. 评分对齐
    3. 分数差异过滤
    '''
    
    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 1. 查询向量数据库
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)
        
        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_result)
        
        # 3. 分数差异过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filte(confirmed, search_result)
        
        return confirmed, options
        
    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        '''
        根据LLM提取的商品名，查询向量数据库
        Args:
            item_names (List[str]): LLM提取的商品名列表
        Returns:
            List[Dict[str, Any]]: 向量数据库查询结果
                Dict[str, Any]: {
                    'extracted_name' : 'LLM提取出来的商品名', 
                    'matches' : [{'item_name': '向量数据库中的商品名', 'score': '对应的分数值'}]
                    }
        '''
        search_result = []
        # 1. 获取Milvus客户端和嵌入模型
        milvus_client = get_milvus_client()
        if not milvus_client:
            return search_result
        
        embedding_model = get_bge_m3_embedding_model()
        if embedding_model is None:
            logger.error("获取嵌入模型失败")
            return search_result
        
        # 2. 对item_name进行嵌入，获取稠密稀疏向量
        hybrid_embeddings = generate_hybrid_embeddings(embedding_model, item_names)
        if hybrid_embeddings is None:
            logger.error("生成嵌入向量失败")
            return search_result

        # 3. 遍历LLM提取的所有商品名
        for idx, item_name in enumerate(item_names):
            # 混合向量检索            
            # 3.1 创建混合检索的请求
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embeddings['dense'][idx],
                sparse_vector=hybrid_embeddings['sparse'][idx],
            )
            # 3.2 执行请求
            hybrid_search_result = execute_hybrid_search_query(milvus_client, collection_name = 'kb_item_names_v2', search_requests=hybrid_search_requests, ranker_weights=(0.5, 0.5), norm_score=True, output_fields=['item_name'])
                
            # 3.3 对结果进行解析
            hybrid_search_requests = {
                'extracted_name': item_name,
                'matches': [
                    {
                        'item_name': h['entity']['item_name'],
                        'score': h['distance']
                    }
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            
            # 3.4 将构建好的查询结果放入到最终搜索结果中
            search_result.append(hybrid_search_requests)
        return search_result
    
    
    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        '''
        根据向量数据库检索到的商品名，将商品名分为“确认”和“候选”两级
        分数阈值作为放到confirmed或者options的条件，默认(0.7, 0.6)
        核心原则：confirmed优先级高于options，即只有当confirmed为空时才考虑options
        Returns:
            Tuple[List[str], List[str]]: (confirmed, options)
                confirmed有：将confirmed中的商品名，传给下游四路检索
                options有：确认在咨询哪一款商品
                二者都有：只考虑confirmed中的商品名
                若二者都没有：直接返回没有找到具体的商品名
        '''
        # 1. 定义容器
        confirmed = []
        options = [] # 最多只留三个
        config = get_config()
        for item in search_results:
            # 2.1 获取LLM提取的商品名
            extracted_name = item['extracted_name']
            
            # 2.2 对某一个商品名下的相似的item_name的分数值进行降序
            matches = sorted(item['matches'], key=lambda x: x['score'], reverse=True)
            
            # 2.3 获取matches中分数值比能进入到confirmed容器阈值大的对象
            high = [match for match in matches if match['score'] >= config.item_name_high_confidence]
            
            # 3. 询问是否能进入到confirmed
            if high:
                # 3.1 如果有和数据库中名字完全相同的
                extract = next((h for h in high if h['item_name'] == extracted_name), None)
                if extract:
                    picked = extract['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    # 没有找到精确的并且high中有多个对象
                    for h in high[:3]:
                        if h['item_name'] not in options and h['item_name'] not in confirmed:
                            options.append(h['item_name'])
            
            # 4. 询问是否能进入到options中  
            else:
                mid = [m for m in matches if m['score'] >= config.item_name_mid_confidence and m['item_name'] not in confirmed and m['item_name'] not in options]
                if mid:
                    for m in mid[:3]:
                        picked = m['item_name']
                        options.append(picked)
                
        return confirmed, options[:3]
        
    def _item_name_score_filte(self, confirmed: List[str], search_results: List[Dict[str, Any]]) -> List[str]:
        '''
        将误判的item_name从confirmed中剔除，留下真实的item_name
        策略：分数与最高分差值超过阈值则视为误判的item_name
        '''
        # 1. 定义字典容器（存储confirmed中item_name在向量数据库中的分数值）
        item_name_score = {}
        
        for search_result in search_results:
            # 1. 获取matches
            matches = search_result.get('matches', [])
            for m in matches:
                item_name = m['item_name']
                score = m['score']
                if item_name in confirmed:
                    item_name_score[item_name] = max(score, item_name_score.get(item_name, 0))
        
        # 2. 对item_name_score进行排序
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)

        if not sorted_item_name_score:
            return confirmed

        # 3. 取出分数值最大的（问题询问的比较明确）
        max_score = sorted_item_name_score[0][1]
        result = [name for name, score in sorted_item_name_score if max_score - score <= 0.15]

        return result
    
class ItemNameExtractor:
    '''
    基于用户的原始问题+[历史对话]，提取商品名
    询问场景：
        1. 单级询问（一个商品） -> [item_name, fake_name...]
        2. 多级询问 -> [item_name_1, item_name_2, fake_name...]
    '''
    def extract_item_name(self, original_query: str) -> Dict[str, List[Any]|Any]:
        
        
        result: Dict[str, Any] = {'item_names': [], 'rewritten_query': ''}
        history = ''
        
        # 1. 获取LLM客户端（返回JSON格式）
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            return result
        
        # 2. 定义提示词
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history if history else '暂无上下文', query=original_query)
        system_prompt = '你是一个专业的客服助手，擅长理解用户意图和提取关键信息。'
        
        # 3. 调用LLM
        ai_message = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        # 安全提取 content：如果有 content 属性就取出来，否则就直接用本身
        raw_content = ai_message.content if hasattr(ai_message, 'content') else ai_message
        
        # 强转为字符串，彻底消除 IDE 对“可能是 list”的类型警告
        response = str(raw_content)
        
        if not response.strip():
            return result
        
        # 4. 清洗和解析LLM响应
        try:
            parsed_result = self._clean_parse(response)
            result['rewritten_query'] = parsed_result.get('rewritten_query', original_query)
            result['item_names'] = parsed_result.get('item_names')
            
        except json.JSONDecodeError as e:
            logger.error(f'清洗以及解析失败: {str(e)}')
        
        return result

    
    def _clean_parse(self, response: str) -> Dict[str, Any]:
        # 1. 清洗JSON围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        
        # 2. 反序列化
        try:
            parsed_llm_result = json.loads(content)
            # 2.1 清洗item_name和rewritten_query
            raw_item_names = parsed_llm_result.get('item_names', [])
            if not isinstance(raw_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [row_item.strip() for row_item in raw_item_names if row_item.strip()]
            
            # 2.2 清洗rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query', '')
            if not isinstance(raw_rewritten_query, str):
                clean_rewritten_query = ''
            else:
                clean_rewritten_query = raw_rewritten_query.strip()
            
            return {'item_names': clean_item_names, 'rewritten_query': clean_rewritten_query}
            
        except JSONDecodeError as e:
            raise ValueError(f'JSON解析失败: {str(e)}')
        

class ItemNameConfirmNode(BaseNode):
    name = 'item_name_confirm_node'
    
    def __init__(self):
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()
    
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state.get('original_query')
        
        # 2. 调用LLM提取商品名（基于原始问题提取item_name）
        llm_result = self._item_name_extractor.extract_item_name(original_query)
        item_names = llm_result.get('item_names')
        rewritten_query = llm_result.get('rewritten_query')
        
        if item_names:
            # 3. 查询向量数据库以及过滤（评分对齐，分数差异过滤）
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []
        
        # 4. 决定state的key（继续or结束）
        self._decide(state, item_names, confirmed, options, rewritten_query)
        
        return state
    
    
    def _decide(self, state: QueryGraphState, item_names, confirmed: List[str],
                options: List[str], rewritten_query):

        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed

        elif options:
            state['answer'] = (f"我不确定您指的是哪款产品。"
                               f"您是在询问以下产品吗：{'、'.join(options)}？")
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            
            
if __name__ == "__main__":

    test_state =  QueryGraphState({"original_query": "你们店里那款RS-12 数字万用表怎么测试电阻？"})  # pyright: ignore[reportArgumentType]

    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)
    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")
