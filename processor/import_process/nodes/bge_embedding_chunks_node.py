from typing import List, Dict, Any
from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState
from processor.import_process.exceptions import ValidationError
from processor.import_process.config import get_config
from utils.bge_me_embedding_util import get_beg_m3_embedding_model


class BGEEmbeddingChunksNode(BaseNode):
    '''
    1. 获取所有的chunks要拼接向量的内容
    2. 批量嵌入chunk的(embedding_content: item_name + chunk.get('content'))
    3. 将所有chunk的embedding结果存储到列表中，返回给下一个节点
    '''
    
    name = 'BGE_Embedding_Chunks_Node'
    
    def process(self, state: ImportGraphState) -> ImportGraphState:
        
        # 1. 参数校验
        validated_chunks, config = self._validate_state(state)
        
        # 2. 获取批量嵌入的阈值
        embedding_batch_size = getattr(config, 'embedding_batch_size', 16)
        
        # 3. 准备分批嵌入
        final_chunks = []
        total_length = len(validated_chunks)
        for i in range(0, total_length, embedding_batch_size):
            batch_chunks = validated_chunks[i:i+embedding_batch_size]
            final_chunk = self._process_batch_chunks(batch_chunks, i, total_length)
            final_chunks.extend(final_chunk)
        
        # 4. 返回最终的chunks
        state['chunks'] = final_chunks
        return state
    
    def _validate_state(self, state: ImportGraphState):
        self.log_step('step1', '参数校验')
        
        config = get_config()
        
        # 1. 获取chunks
        chunks = state.get('chunks', [])
        
        # 2. 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise ValidationError('chunks无效', self.name)
        
        # 3. 返回chunks
        self.logger.info(f'嵌入的块数:{len(chunks)}')
        return chunks, config
    
    
    def _process_batch_chunks(self, batch_chunks: List[Dict[str, Any]], i: int, total_length: int):
        '''
        拼接要嵌入的内容，把嵌入的向量注入到chunk中，返回最终的chunk
        '''
        self.log_step('step2', f'批量处理chunk嵌入：批次{i+1} / {i+len(batch_chunks)}')
        
        # 1. 循环处理所有chunk要嵌入的内容
        embedding_contents = []
        for _, chunk in enumerate(batch_chunks):
            # 1.1 提取content
            content = chunk.get('content')
            # 1.2 提取item_name
            item_name = chunk.get('item_name')
            # 1.3 拼接要嵌入的内容
            embedding_content = f'{item_name}\n{content}'
            embedding_contents.append(embedding_content)
        
        # 2. 批量嵌入
        model = get_beg_m3_embedding_model()
        embedding_result = model.encode_documents(embedding_contents)
        
        
        # 3. 循环处理所有chunk的向量注入到每一个chunk中
        for i, chunk in enumerate(batch_chunks):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][i].tolist()
            # 3.2 结构csr矩阵并获取稀疏向量
            csr_array = embedding_result['sparse']
            indptr = csr_array.indptr
            
            start_indptr = indptr[i]
            end_indptr = indptr[i + 1]
            
            token_id = csr_array.indices[start_indptr:end_indptr].tolist()
            weight = csr_array.data[start_indptr:end_indptr].tolist()
            
            sparse_vector = dict(zip(token_id, weight))
            
            # 3.3 注入
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector
        
        self.logger.info('开始处理')
        return batch_chunks