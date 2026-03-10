import json
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import DataType
from langchain_core.messages import SystemMessage, HumanMessage

from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState
from processor.import_process.exceptions import ValidationError, EmbeddingError
from processor.import_process.config import get_config
from processor.import_process.prompts.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
    
from utils.llm_util import get_llm_client
from utils.milvus_util import get_milvus_client
from utils.bge_me_embedding_util import get_bge_m3_embedding_model




class ItemNameRecognitionNode(BaseNode):
    name = 'item_name_recognition'
    
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        chunks, file_title, config = self._validate_inputs(state)
        
        # 2. 构建LLM上下文（让LLM提取商品名）
        item_name_context = self._prepare_item_name_context(file_title, chunks, config)
        
        # 3. 调用LLM
        item_name = self._recognize_item_name_by_llm(file_title, item_name_context, config)
        
        # 4. 嵌入商品名
        dense, sparse = self._embedding_item_name(item_name)
        
        # 5. 存储到Milvus数据库
        self._save_to_milvus(state, file_title, item_name, dense, sparse, config)
        
        # 6. 回填item_name到state和chunks
        self._fill_item_name(item_name, state, chunks)
        
        return state
    
    def _validate_inputs(self, state: ImportGraphState):
        self.log_step('step1', '校验输入参数')
        
        config = get_config()
        
        # 1. 获取信息
        file_title = state.get('file_title')
        chunks = state.get('chunks')
        item_name_chunk_k = config.item_name_chunk_k
        
        # 2. 判断参数
        if not file_title:
            raise ValidationError('文件标题为空', self.name)
            
        if not chunks or not isinstance(chunks, list):
            raise ValidationError('chunk无效', self.name)
        
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError('item_name_chunk_k无效', self.name)
        
        self.logger.info(f'检测到文件{file_title}，对应的切片长度{len(chunks)}')
        # 3. 返回
        return chunks, file_title, config
        
        
    def _prepare_item_name_context(self, file_title: str, chunks: List[Dict[str, Any]], config) -> str:
        self.log_step('step2', '构建商品名提取的上下文')
        
        result = [file_title]
        total_length = 0
        for index, chunk in enumerate(chunks[:config.item_name_chunk_k]):
            # 1. 判断chunk结构类型
            if not isinstance(chunk, Dict):
                continue
            
            # 2. 提取
            content = chunk.get('content')
            
            spices = f'[切片] - {index+1} - {content}'
            result.append(spices)
            
            # 3. 计算总长度，并判断是否超过最大阈值
            total_length += len(spices)
            if total_length > config.item_name_chunk_size:
                break
            
        return '\n\n'.join(result)[:config.item_name_chunk_size]
            
            
    def _recognize_item_name_by_llm(self, file_title: str, context: str, config) -> str:
        self.log_step('step3', '通过LLM识别商品名')
        
        # 1. 获取LLM客户端
        llm_client = get_llm_client('qwen-flash')
        if llm_client is None:
            self.logger.warning(f'LLM初始化失败，安全回退到标题名{file_title}')
            return file_title
        
        # 2. 构建LLM提示词（格式化用户提示词模板）
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=context)
        
        # 3. 调用模型(content不能放带变量的字符串)
        try:
            response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            # 4. 获取LLM响应内容
            response_content = getattr(response, 'content', '').strip()
            
            # 5. 判断
            if not response_content or response_content.upper() == 'UNKNOWN':
                self.logger.warning(f'LLM无法提取有效的商品名,安全回退到标题名{file_title}')
                return file_title
            else:
                self.logger.info(f'LLM提取到的商品名: {response_content}')
                return response_content
        except Exception as e:
            self.logger.error(f'LLM调用失败，安全回退到标题名: {file_title}，原因: {str(e)}')
            return file_title
        
          
    def _embedding_item_name(self, item_name: str) -> Tuple[List, Dict]:
        self.log_step('step4', '开始进行向量嵌入')
        try:
            # 1. 获取嵌入模型
            embedding_model = get_bge_m3_embedding_model()

            # 2. 嵌入item_name
            embedding_result = embedding_model.encode_documents([item_name])

            # 3. 获取稠密和稀疏向量
            dense = embedding_result['dense'][0].tolist()
            start_index = embedding_result['sparse'].indptr[0]
            end_index = embedding_result['sparse'].indptr[1]
            weights = embedding_result['sparse'].data[start_index:end_index].tolist()
            token_ids = embedding_result['sparse'].indices[start_index:end_index].tolist()
            sparse: Dict[int, float] = {
                int(token_id): float(weight)
                for token_id, weight in zip(token_ids, weights)
            }
            return dense, sparse
        except Exception as e:
            self.logger.error(f"嵌入商品名:{item_name}失败,原因是：{str(e)}")
            raise EmbeddingError(f"嵌入商品名:{item_name}失败,原因是：{str(e)}", self.name)
        
    
    def _save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str, dense_vector: Optional[List[float]], sparse_vector: Optional[dict], config):
        """保存到 Milvus"""
        self.log_step("step_5", "保存到 Milvus")
        
        # 1. 参数校验
        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整，跳过保存")
            return
        
        # 2. 操作Milvus
        try:
            # 2.1 获取 Milvus 客户端
            client = get_milvus_client()
            
            if client is None:
                return

            # 2.2 获取集合名字
            collection_name = config.item_name_collection

            # 2.3 对集合名字做幂等性校验（判断集合是否存在，不存在就创建新的）
            if not client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(client, collection_name)

            # 2.4 构建字典结构数据 
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }

            # 2.5 插入数据
            result = client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已保存到 Milvus，ID: {result['ids'][0]}")

        except Exception as e:
            self.logger.warning(f"Milvus 保存失败: {e}")
    
    
    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合"""
        self.logger.info(f"创建集合: {collection_name}")

        # 1. 定义约束
        schema = client.create_schema(enable_dynamic_fields=True)
        # 1.1 主键约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,is_primary=True, auto_id=True, max_length=100)
        # 1.2 标量字段的约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # 1.3 向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        # 2.1 创建稠密向量字段索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        # 2.2 创建稀疏向量字段索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功")
    
    
    def _fill_item_name(self, item_name: str, state: ImportGraphState, chunks: List[Dict[str, Any]]):
        self.log_step('step6', '回填item_name到state和chunks')
        
        for chunk in chunks:
            chunk['item_name'] = item_name # 方便下游模型参考
            
        state['item_name'] = item_name # 方便程序员使用
        
        

if __name__ == '__main__':
    
    # 1. 读取chunks.json
    chunks_json_path = r'D:\atguigu\shopkeer_brain\knowledge\processor\import_process\output_temp_dir\万用表RS-12的使用\hybrid_auto\chunks.json'
    
    with open(chunks_json_path, 'r', encoding='utf-8') as f:
        chunks_content = json.load(f)
    
    state = ImportGraphState({
        'file_title': '万用表RS-12的使用',
        'chunks': chunks_content
    })
    
    item_name_recognizer = ItemNameRecognitionNode()
    
    result = item_name_recognizer.process(state)
    
    for dict in result:
        print(dict)
        print('-'*50)