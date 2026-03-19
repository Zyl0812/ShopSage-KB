"""
测试 reranker 模型在 pairs 数量为 1 时返回的是 float 还是 List[float]
"""
from dotenv import load_dotenv
load_dotenv()

from utils.bge_reranker_util import get_reranker_model

reranker = get_reranker_model()

# 单个 pair
single_pair = [("万用表怎么测电阻", "使用电阻档位，将表笔连接被测电阻两端")]
single_result = reranker.compute_score(single_pair)
print(f"单个 pair 返回类型: {type(single_result)}, 值: {single_result}")

# 多个 pairs 作为对比
multi_pairs = [
    ("万用表怎么测电阻", "使用电阻档位，将表笔连接被测电阻两端"),
    ("万用表怎么测电阻", "打开电源开关，按下启动按钮"),
]
multi_result = reranker.compute_score(multi_pairs)
print(f"多个 pairs 返回类型: {type(multi_result)}, 值: {multi_result}")
