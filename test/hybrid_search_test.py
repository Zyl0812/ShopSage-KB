"""
混合向量检索测试 —— 动态字段 + item_name 过滤

步骤：
1. 创建测试集合（含动态字段支持）
2. 用 BGE-M3 生成混合向量并插入测试数据
3. 执行带 item_name 过滤的混合检索
4. 清理测试集合
"""

import logging
from pymilvus import DataType
from utils.milvus_util import (
    get_milvus_client,
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)
from utils.bge_m3_embedding_util import (
    get_bge_m3_embedding_model,
    generate_hybrid_embeddings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────
TEST_COLLECTION = "test_hybrid_search"
EMBEDDING_DIM = 1024

# 测试数据：content + 动态字段 item_name
TEST_DOCS = [
    {"content": "万用表RS-12可以测量交流电压和直流电压", "item_name": "万用表RS-12"},
    {"content": "万用表RS-12的电阻测量范围为0到200兆欧", "item_name": "万用表RS-12"},
    {"content": "万用表RS-12可以测量交流电压和直流电压", "item_name": "万用表 RS-12"},
    {"content": "万用表RS-12的电阻测量范围为0到200兆欧", "item_name": "万用表 RS-12"},
    {"content": "示波器DS-100支持双通道同时采集", "item_name": "示波器DS-100"},
    {"content": "示波器DS-100的带宽为100MHz", "item_name": "示波器DS-100"},
    {"content": "电烙铁T-60适用于精密焊接作业", "item_name": "电烙铁T-60"},
]


def create_test_collection(client):
    """创建测试集合，启用动态字段"""
    # 如果已存在则先删除
    if client.has_collection(TEST_COLLECTION):
        client.drop_collection(TEST_COLLECTION)
        logger.info(f"已删除旧集合: {TEST_COLLECTION}")

    # 1. 构建 schema，启用动态字段
    schema = client.create_schema(enable_dynamic_field=True)

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM,
    )
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # 2. 构建索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_idx",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_idx",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    # 3. 创建集合
    client.create_collection(
        TEST_COLLECTION, schema=schema, index_params=index_params
    )
    logger.info(f"测试集合 {TEST_COLLECTION} 创建成功（动态字段已启用）")


def insert_test_data(client, embedding_model):
    """生成向量并插入测试数据，item_name 作为动态字段写入"""
    contents = [doc["content"] for doc in TEST_DOCS]

    # 生成混合向量
    embeddings = generate_hybrid_embeddings(embedding_model, contents)
    if embeddings is None:
        raise RuntimeError("向量生成失败")

    # 组装插入数据
    rows = []
    for i, doc in enumerate(TEST_DOCS):
        rows.append({
            "dense_vector": embeddings["dense"][i],
            "sparse_vector": embeddings["sparse"][i],
            "item_name": doc["item_name"],  # 动态字段
        })

    result = client.insert(collection_name=TEST_COLLECTION, data=rows)
    logger.info(f"插入 {result['insert_count']} 条测试数据")


def search_with_filter(client, embedding_model, query: str, filter_expr: str = None):
    """执行混合检索，可选 item_name 过滤"""
    logger.info(f"查询: '{query}' | 过滤: {filter_expr or '无'}")

    # 生成查询向量
    embeddings = generate_hybrid_embeddings(embedding_model, [query])
    if embeddings is None:
        raise RuntimeError("查询向量生成失败")

    dense_vec = embeddings["dense"][0]
    sparse_vec = embeddings["sparse"][0]

    # 创建检索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=filter_expr,
        limit=5,
    )

    # 执行混合检索
    results = execute_hybrid_search_query(
        milvus_client=client,
        collection_name=TEST_COLLECTION,
        search_requests=reqs,
        ranker_weights=(0.5, 0.5),
        limit=5,
        output_fields=["item_name"],
    )

    # 打印结果
    if results:
        for hits in results:
            for hit in hits:
                print(f"  id={hit['id']}, distance={hit['distance']:.4f}, item_name={hit['entity'].get('item_name')}")
    else:
        print("  无结果")
    print()


if __name__ == "__main__":
    client = get_milvus_client()
    embedding_model = get_bge_m3_embedding_model()

    # # 1. 创建测试集合
    # print("=" * 50)
    # print("步骤1: 创建测试集合")
    # print("=" * 50)
    # create_test_collection(client)

    # # 2. 插入测试数据（item_name 为动态字段）
    # print("=" * 50)
    # print("步骤2: 插入测试数据")
    # print("=" * 50)
    # insert_test_data(client, embedding_model)


    print("=" * 50)
    print("步骤3: 不带空格（过滤 item_name in ['万用表RS-12']）")
    print("=" * 50)
    search_with_filter(
        client, embedding_model,
        "电压测量",
        filter_expr='item_name in ["万用表RS-12"]',
    )

    print("=" * 50)
    print("步骤4: 带空格：使用item_name in ['万用表 RS-12']过滤）")
    print("=" * 50)
    search_with_filter(
        client, embedding_model,
        "电压测量",
        filter_expr='item_name in ["万用表 RS-12"]',
    )
    

    print("=" * 50)
    print("步骤5: 不带空格：使用item_name == '万用表RS-12'过滤）")
    print("=" * 50)
    search_with_filter(
        client, embedding_model,
        "电压测量",
        filter_expr='item_name == "万用表RS-12"',
    )

    print("=" * 50)
    print("步骤6: 带空格（过滤 item_name='万用表 RS-12'）")
    print("=" * 50)
    search_with_filter(
        client, embedding_model,
        "电压测量",
        filter_expr='item_name == "万用表 RS-12"',
    )

    # 5. 混合检索 —— 按 item_name like 模糊过滤
    print("=" * 50)
    print("步骤5: 混合检索（模糊过滤 item_name like '万用表%'）")
    print("=" * 50)
    search_with_filter(
        client, embedding_model,
        "电压测量",
        filter_expr='item_name like "万用表%"',
    )