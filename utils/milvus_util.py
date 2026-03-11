import os
import logging

from typing import Optional
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

client : Optional[MilvusClient] = None

def get_milvus_client() -> Optional[MilvusClient]:
    global client
    
    if client is not None:
        return client
    
    try:
        client = MilvusClient(
            uri=os.getenv("MILVUS_URI", "http://192.168.10.130:19530")
        )
        return client

    except Exception as e:
        logger.error(f"Milvus客户端创建失败: {e}")
        client = None
        return None

# ------------------------------------------------------------------
# 混合检索
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 创建混合检索请求
# ------------------------------------------------------------------
def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None, limit=5):
    """
    创建混合搜索请求
    Args:
        dense_vector: 稠密向量
        sparse_vector: 稀疏向量
        dense_params: 稠密向量搜索参数，默认为None
        sparse_params: 稀疏向量搜索参数，默认为None
        expr: 查询表达式，默认为None
        limit: 返回结果数量限制，默认为5
    Returns: 
        包含稠密和稀疏搜索请求的列表
    """
    # 默认参数
    if dense_params is None:
        dense_params = {'metric_type': 'COSINE'}
    if sparse_params is None:
        sparse_params = {'metric_type': 'IP'}
    
    # 创建稠密向量搜索请求
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        limit=limit,
        expr=expr,
    )    
    # 创建稀疏向量搜索请求
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        limit=limit,
        expr=expr,
    )
    
    return [dense_req, sparse_req]

# ------------------------------------------------------------------
# 执行混合检索请求
# ------------------------------------------------------------------
def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=False,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """
    执行混合搜索
    :param collection_name: 集合名称
    :param search_requests: 搜索请求列表，通常是[dense_req, sparse_req]
    :param ranker_weights: 权重排名器的权重，默认为(0.5, 0.5)
    :param norm_score: 是否对分数进行归一化，默认为True
    :param limit: 返回结果数量限制，默认为5
    :param output_fields: 要返回的字段列表，默认为None
    :param search_params: 搜索参数，默认为None
    :return: 搜索结果
    """
    try:
        # 创建权重融合排序器
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        
        # 默认输出字段
        if output_fields is None:
            output_fields = ['item_name']
        
        # 执行混合搜索
        results = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )
        
        # 动态计算所有查询返回的结果总数
        total_hits = sum(len(hits) for hits in results) if results else 0
        logger.info(f'Milvus 混合搜索完成，共处理 {len(results) if results else 0} 条结果，总命中数 {total_hits}')
        return results
    except Exception as e:
        logger.error(f"执行Milvus混合搜索时发生错误: {e}")
        return []