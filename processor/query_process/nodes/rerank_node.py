import logging

from typing import List, Dict, Any

from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode
from utils.bge_reranker_util import get_reranker_model



class RerankNode(BaseNode):
    name = 'rerank_node'
    
    def process(self, state: QueryGraphState) -> QueryGraphState:
        
        # 1. 获取query
        user_query = state.get('rewritten_query') or state.get('original_query')
        
        # 2. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_sources(state)
        
        # 3. Rerank精排
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)
        
        # 3. 动态top_k截取（断崖检测）
        topk_docs = self._cliff_cut_off(reranked_docs)
        
        state['reranked_docs'] = topk_docs
        
        return state
        
    
    def _merge_multi_sources(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        final_docs = []
        # 1. 获取RRF本地文档
        for rrf_doc in (state.get('rrf_chunks') or []):
            if not isinstance(rrf_doc, dict):
                continue
                
            content = rrf_doc.get('content', '').strip()
            if not content:
                continue
            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id', '').strip()
            
            # 格式化本地RRF的chunk结构
            format_doc = self._format_docs(content=content, title=title, chunk_id=chunk_id, source='local')
            final_docs.append(format_doc)
        
        # 2. 获取MCP远程文档
        for web_doc in (state.get('web_search_docs') or []):
            if not isinstance(web_doc, dict):
                continue
                
            content = web_doc.get('content', '').strip()
            if not content:
                continue
            
            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()
            
            format_doc = self._format_docs(content=content, title=title, url=url, source='web')
            final_docs.append(format_doc)
            
        self.logger.info(f'收集到{len(final_docs)}条文档进行rerank精排')
        return final_docs
            
            
    def _format_docs(self, content: str, title: str='', chunk_id=None, url:str='', source:str='') -> Dict[str, Any]:
        
        return {
            'content': content,
            'title': title,
            'chunk_id': chunk_id,
            'url': url,
            'source': source
        }
    
    
    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''
        使用Rerank模型对合并后的文档进行精排。
        
        Args:
            user_query (str): 用户查询
            merged_multi_docs (List[Dict[str, Any]]): 合并后的文档列表
        
        Returns:
            List[Dict[str, Any]]: 精排后的文档列表
        '''
        
        # 1. 判断合并后的多源文档是否存在
        if not merged_multi_docs:
            return []
        
        # 2. 获取reranker模型
        reranker_model = get_reranker_model()
        if reranker_model is None:
            self.logger.error('重排序模型获取失败')
            return []
            
        # 3. 构建 Q-D Pairs
        pairs = [(user_query, doc['content']) for doc in merged_multi_docs]
        
        # 4. 计算
        try:
            rerank_score = reranker_model.compute_score(pairs)
        except Exception as e:
            self.logger.error(f'重排序计算失败: {e}')
            return []
        
        # 5. 排序
        sorted_docs = [{**doc, 'score': score} for score, doc in sorted(zip(rerank_score, merged_multi_docs), key=lambda x: x[0], reverse=True)]
        
        return sorted_docs
    
    
    def _cliff_cut_off(self, reranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''
        动态top_k截取（断崖检测）。
        
        Args:
            reranked_docs (List[Dict[str, Any]]): 精排后的文档列表
        
        Returns:
            List[Dict[str, Any]]: 截取后的文档列表
        '''
        if not reranked_docs:
            return []
    
        # 确定截断范围
        upper_bound = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)
    
        cutoff_pos = upper_bound  # 默认取最大值
    
        # 从 min_topk 位置开始检测
        for i in range(lower_bound - 1, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")
    
            # 跳过无得分的文档
            if current_score is None or next_score is None:
                continue
    
            # 计算差值
            abs_gap = current_score - next_score                        # 绝对差值：主要满足高分文档
            rel_gap = abs_gap / (abs(current_score) + 1e-6)         # 相对比例：主要满足低分文档
    
            # 发现断崖，立即截断
            if abs_gap >= self.config.rerank_gap_abs or rel_gap >= self.config.rerank_gap_ratio:
                cutoff_pos = i + 1
                self.logger.debug(
                    f"断崖检测: 位置 {i+1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
                )
                break
    
        return reranked_docs[:cutoff_pos]
        