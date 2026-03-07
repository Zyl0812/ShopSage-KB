import os
import logging

from typing import Optional
from dotenv import load_dotenv

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None


def get_beg_m3_embedding_model():
    global bge_m3_ef

    # 1.判断
    if bge_m3_ef is not None:
        return bge_m3_ef

    # 2. 获取参数
    model_name = os.getenv('BGE_M3_PATH', 'BAAI/bge-m3')
    device = os.getenv('BGE_DEVICE', 'cpu')
    use_fp16_str = os.getenv('BGE_FP16', 'False')
    use_fp16 = use_fp16_str.lower() in ('true', '1', 'yes')

    # 3. 定义嵌入模型对象 # 默认维度1024
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )
    # 4. 返回
    return bge_m3_ef

def print_sparse_matrix(sparse_vector, top_k=20):
    """
    打印稀疏矩阵信息

    Args:
        sparse_vector: CSR格式的稀疏向量
        top_k: 显示权重最高的前k个token
    """
    if hasattr(sparse_vector, 'indptr') and hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'data'):
        # CSR格式
        print("=== 稀疏矩阵信息 ===")
        print("格式: CSR")
        print(f"维度: {sparse_vector.shape}")
        print(f"非零元素数量: {sparse_vector.nnz}")
        print("\n--- CSR结构 ---")
        print(f"indptr (行指针): {sparse_vector.indptr}")
        print(f"indices (token索引): {sparse_vector.indices}")
        print(f"data (权重值): {sparse_vector.data}")

        # 转换为字典格式
        indices = sparse_vector.indices
        data = sparse_vector.data
        token_weight_dict = {int(idx): float(weight) for idx, weight in zip(indices, data)}

        # 按权重排序
        sorted_items = sorted(token_weight_dict.items(), key=lambda x: x[1], reverse=True)

        print(f"\n--- Token权重映射 (前{min(top_k, len(sorted_items))}个) ---")
        for token_id, weight in sorted_items[:top_k]:
            print(f"  token_id={token_id}: weight={weight:.6f}")
    else:
        print("未知的稀疏向量格式")

if __name__ == '__main__':
    embedding_model = get_beg_m3_embedding_model()

    query = "我喜欢Python语言"

    # print(embedding_model.encode_queries([query]))
    result = embedding_model.encode_documents([query])
    print(result)

    # 稠密向量：
    # dense = result['dense'][0].tolist()

    # 稀疏向量（CSR:核心目标：将整个空间那些非0的元素存储起来:行指针[indptr](0 6) 权重列表[data](0.01,0.21,0.13,0.04,0.5,0.6) tokenId 列表[indices](1000,900,10,1,2,9999)）
    # Mivlus:sparse:{"tokenId":'weight',....}
    # 打印稀疏矩阵
    print_sparse_matrix(result['sparse'])
