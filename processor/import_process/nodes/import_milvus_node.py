import logging
import json

from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional
from dataclasses import dataclass
from pymilvus import DataType
from pymilvus.milvus_client import MilvusClient

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.state import ImportGraphState
from processor.import_process.exceptions import ValidationError
from processor.import_process.config import get_config
from utils.milvus_util import get_milvus_client

'''
门面+建造者设计模式
门面角色：ImportMilvusNode.process()
    1. 数据校验
    2. insert
    3. 更新state
建造者（和Milvus操作相关的类）
    类1. MilvusSchemaBuilder:专门负责对Milvus的约束进行操作
    类2. MilvusIndexBuilder:专门负责对Milvus的索引进行操作
    类3. MilvusInsertBuilder:专门负责对Milvus做插入操作
    
    类4. ScalarFieldSpec:专门负责管理Milvus标量字段（对大多数标量的共性字段做提取复用）
'''

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int]=None
    
# Sequence:有序可读序列
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec('content', DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec('title', DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec('parent_title', DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec('file_title', DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec('item_name', DataType.VARCHAR, max_length=65535),
)

class _MilvusSchemaBuilder:
    '''
    专门负责构建约束
    '''
    
    @staticmethod
    def build(client: MilvusClient, dim: int):
        logger.info('开始构建schema')
        # 1. 创建约束对象
        schema = client.create_schema(enable_dynamic_field=True)
        
        # 2. 构建主键字段约束
        schema.add_field(
            field_name='chunk_id',
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        
        # 3. 构建向量字段约束
        schema.add_field(
            field_name='dense_vector',
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        schema.add_field(
            field_name='sparse_vector',
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        
        # 4. 构建标量字段约束
        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {'field_name': scalar_field.field_name, 'datatype': scalar_field.datatype, 'nullable': True}

            if scalar_field.max_length is not None:
                kwargs['max_length'] = scalar_field.max_length

            schema.add_field(**kwargs)
        
        logger.info('构建schema完成')
        return schema

class _MilvusIndexBuilder:
    '''
    负责处理Milvus的索引
    '''
    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f'开始构建集合{collection_name}索引')
        # 1. 创建索引对象
        index = client.prepare_index_params(collection_name=collection_name)
        
        # 2. 给向量字段添加索引
        # 2.1 稠密向量字段添加索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        # 2.2 稀疏向量字段添加索引
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        logger.info(f'构建集合{collection_name}索引完成')
        return index

class _MilvusInsertBuilder:
    '''
    负责将数据插入到Milvus中，回填chunk_id
    '''
    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name
    
    def insert(self, chunks:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f'开始插入{len(chunks)}条数据到Milvus')
        # 1. 插入数据到Milvus中
        inserted_result = self._client.insert(collection_name=self._collection_name, data=chunks)
        
        inserted_count = inserted_result.get('insert_count')
        inserted_ids = inserted_result.get('ids')
        
        # 2. 回填chunk_id
        self._fill_chunk_ids(chunks, inserted_ids)  # pyright: ignore[reportArgumentType]
        
        logger.info(f'完成插入{inserted_count}条数据')
        return chunks
    
    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], inserted_ids: List[Any]):
        for chunk, chunk_id in zip(chunks, inserted_ids):
            chunk['chunk_id'] = chunk_id


class ImportMilvusNode(BaseNode):
    name = 'import_milvus_node'
    
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        validated_chunks, dim, config = self._validate_get_inputs(state)
        
        # 2. 获取Milvus客户端
        milvus_client = get_milvus_client()
        
        # 3. 判断客户端
        if milvus_client is None:
            return state
        
        # 4. 获取集合名字
        collection_name = getattr(config, 'chunks_collection', 'test_chunks_collection')
        
        # 5. 确保集合存在
        self._ensure_has_collection(milvus_client, collection_name, dim)
        
        # 6. 插入数据到Milvus中
        inserter = _MilvusInsertBuilder(milvus_client, collection_name)
        final_chunks = inserter.insert(validated_chunks)
        
        # 7. 回填chunk_id到state
        state['chunks'] = final_chunks
        return state
    
    
    def _validate_get_inputs(self, state: ImportGraphState):
        self.log_step('step1', '参数校验')
        config = get_config()
        # 1. 获取chunks并校验
        chunks = state.get('chunks')
        if not chunks:
            raise ValidationError('待入库的chunk不存在', self.name)
        
        # 2. 遍历chunks，校验是否有混合向量
        validated_chunks = []
        for chunk in chunks:
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error('待入库chunk的混合向量不存在')

        # 3. 判断有效集合
        if not validated_chunks:
            # validated_chunks为空，说明所有chunk都缺少混合向量
            raise ValidationError('入库的chunk都无效', self.name)

        # 4. 获取向量维度
        dim = len(validated_chunks[0].get('dense_vector'))
        self.logger.info(f'导入Milvus向量数据库的有效chunk数量：{len(validated_chunks)}，且chunk的向量维度为{dim}')
        
        return validated_chunks, dim, config
        
    
    def _ensure_has_collection(self, milvus_client: MilvusClient, collection_name: str, dim: int, delete_flag: bool = False):
        self.log_step('step2', f'准备集合{collection_name}创建')
        # 1. 是否要删除集合
        if delete_flag and milvus_client.has_collection(collection_name):
            self.logger.info(f'集合{collection_name}已删除')
            milvus_client.drop_collection(collection_name)
        
        # 2. 判断集合是否存在
        if milvus_client.has_collection(collection_name):
            self.logger.info(f'集合{collection_name}已存在')
            return 
        
        # 3. 创建约束
        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        
        # 4. 创建索引
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)
        
        # 5. 创建集合
        milvus_client.create_collection(collection_name, schema=schema, index_params=index)        




def _cli_main() -> None:
    setup_logging()

    temp_dir = Path(
        r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\output_temp_dir\万用表RS-12的使用\hybrid_auto")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

    state: ImportGraphState = {
        "chunks": content.get("chunks", [])
    }

    import_milvus = ImportMilvusNode()
    result_state = import_milvus.process(state)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_state, fh, ensure_ascii=False, indent=4)

    logger.info(f"备份临时文件{output_path}成功")


if __name__ == "__main__":
    _cli_main()