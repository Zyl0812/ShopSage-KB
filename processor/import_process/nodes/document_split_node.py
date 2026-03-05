import json
import os
import re
import time

from typing import Dict, Tuple, List, Any

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.state import ImportGraphState
from processor.import_process.config import get_config

from utils.markdown_util import MarkdownTableLinearizer

from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitNode(BaseNode):
    name = 'document_split_node'
    
    def process(self, state: ImportGraphState):
        # 1. 获取参数
        md_content, file_title, max_content_length, min_content_length = self._get_input(state)
        
        # 2. 根据标题切割
        sections = self._split_by_headings(md_content, file_title)
        
        # 3. 切割后的处理(切分or合并)
        final_chunks = self._split_and_merge(sections, max_content_length, min_content_length)
        
        # 4. 组装
        chunks = self._assemble_chunk(final_chunks)
        
        # 5. 更新state:chunks
        state['chunks'] = chunks

        # 6. 日志统计
        self._log_summary(md_content, chunks, max_content_length)
        
        # 7. 备份
        self._backup_chunks(state, chunks)
        
        return state
        
        # ------------------------------------------------------------------ #
        #                       日志 & 备份                                    #
        # ------------------------------------------------------------------ #

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")

        except Exception as e:
            self.logger.warning(f"备份失败: {e}")
    
    def _get_input(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        
        self.log_step('step1', '切分文档的参数校验以及获取')
        
        config = get_config()
        # 1. 获取md_content
        md_content = state.get("md_content", '')
        
        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 3. 获取文件标题
        file_title = state.get('file_title', '')
        
        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <=0 or config.max_content_length < config.min_content_length:
            raise ValueError('切片长度参数校验失败')
            
        return md_content, file_title, config.max_content_length, config.min_content_length
    
    
    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
        '''
        根据MD的标题进行切分
        Args:
            md_content (str): markdown内容
            file_title (str): 文件标题
        Returns:
            List[dict]: 切分后的sections
                sections:{
                    'title': str,
                    'body': str,
                    'file_title': str,
                    'parent_title': str,（父标题）
                }
        '''
        self.log_step('step2', '根据标题进行切割')
        
        # 1. 定义变量
        in_fence = False
        body_lines = []
        sections = []
        current_level = 0
        current_title = ''
        hierarchy = [''] * 7    # 7个长度但是0号索引不用
        
        # 2. 定义正则表达式
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        
        # 3. 切分
        content_lines = md_content.split('\n')
        
        def _flush():
            '''
            封装section对象 
            '''
            body = '\n'.join(body_lines)
            if body or current_title:
                parent_title = ''
                for i in range(current_level-1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
            
                return sections.append({
                    'title': current_title if current_title else file_title,
                    'body': body,
                    'file_title': file_title,
                    'parent_title': parent_title if parent_title else file_title
                })
                
        for line in content_lines:
            # 3.1 判断是否存在代码块围栏
            if line.strip().startswith('```') or line.strip().startswith('~~~'):
                in_fence = not in_fence
                
            match = heading_re.match(line) if not in_fence else None
            
            if match:
                # 标题行
                _flush()
                current_level = len(match.group(1))
                current_title = line
                hierarchy[current_level] = current_title
                
                body_lines = []
                for i in range(current_level+1, 7):
                    hierarchy[i] = ''

            else:
                # 除了标题的其他行
                body_lines.append(line)
        _flush()
        
        return sections
        
        
    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int):
        '''
        Args:
            sections (List[Dict]): 根据一级标题切分后的所有section块
            max_content_length (int): 每一个section的content内容（title+body）最大内容长度（标题注入）
            min_content_length (int): 每一个section的content内容 长度如果小于该值，尝试进行合并
        
        Returns:
            List[section]
        '''
        self.log_step('step3', '切分长内容以及合并短内容')
        
        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length))
            
        # 2. 合并
        final_sections = self._merge_short_sections(current_sections, min_content_length)
        
        # 3. 返回
        return final_sections
        
    def _split_long_section(self, section: Dict[str, Any], max_content_length: int):
        '''
        只有满足条件的才会切（当前section的内容是否达到了最大值）
        Args:
            sections (Dict): 根据一级标题切分后的所有section块
            max_content_length (int): 每一个section的content内容（title+body）最大内容长度（标题注入）
        Returns:
            List[section]
        '''
        self.log_step('step4', '进行长内容切分')
        
        # 1. 获取section对象的属性
        title = section.get('title', '')
        body = section.get('body', '') # 可能为空
        file_title = section.get('file_title', '')
        parent_title = section.get('parent_title', '')
        
        # 2. 判断表格
        if '<table>' in body:
            # TODO
            self.logger.info('检测到表格')
            body = MarkdownTableLinearizer.process(body)
        
        # 3. 对标题做校验
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f'检测文件{file_title}对应的{title}长度过长')
            title = title[:50]
        
        # 4. 拼接title的前缀
        title_prefix = f'{title}\n\n'
        
        # 5. 计算总长度
        total_length = len(title_prefix) + len(body or '')
        
        # 6. 小于或刚好满足阈值（直接返回）
        if total_length <= max_content_length:
            return [section]
        
        # 7. 计算body可用的长度
        body_length = max_content_length - len(title_prefix)
        
        if body_length <= 0:
            return [section]
            
        # 8.切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )
        texts = text_splitter.split_text(body)
        
        if len(texts) <= 1:
            return [section]
            
        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                'title': title + '-' + f'{index+1}',
                'body': text,
                'file_title': file_title,
                'parent_title': parent_title,
                'part': f'{index+1}'
            })
        
        return sub_section
    
    
    
    def _merge_short_sections(self, sections: List[Dict[str, Any]], min_content_length: int):
        '''
        贪心累加算法
        两个局限性：1. 可能撑爆最大阈值；2. 孤儿小块
        Args:
            sections (List[Dict]): 根据一级标题切分后的所有section块
            min_content_length (int): 每一个section的content内容 长度如果小于该值，尝试进行合并
        Returns:
            List[section]
        '''
        current_section = sections[0]
        final_sections = []
        
        # 1. 遍历合并
        for section in sections[1:]:
            # 同源判断
            same_parent = (current_section['parent_title'] == section['parent_title'])
            if same_parent and len(current_section['body']) < min_content_length:
                # 同源并且长度不足够
                # body的合并
                current_section['body'] = (
                    current_section.get('body', '').rstrip() + section.get('body', '').lstrip()
                )
                # 更新current_title，简单的能涵盖住合并进来的内容
                current_section['title'] = current_section['parent_title']
                current_section['part'] = '0'
            else:
                # 不同源或长度足够
                # 将原来的section进行封箱
                final_sections.append(current_section)
                # 更新section
                current_section = section
        # 对剩下的section做处理
        final_sections.append(current_section)
        
        # 3. 对所有section的part做处理（为每一个父标题设置对应的part计数器）
        part_counter = {}
        result = []
        for final_section in final_sections:
            if 'part' in final_section:
                # 获取section的父标题
                parent_title = final_section['parent_title']
                # 给计数器赋值
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                # 更新part
                new_part = part_counter[parent_title]
                final_section['part'] = str(new_part)
                # 更新title
                final_section['title'] = final_section['parent_title'] + '-' + str(new_part)
            result.append(final_section)
        
        return result

    
    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''
        最终组合chunk
        Args:
            final_chunks:
                
        Return:
            
        '''
        self.log_step('step4', '组装最终的切片信息')
        chunks = []
        for chunk in final_chunks:
            # 1. 获取chunk的信息
            title = chunk.get('title', '')
            file_title = chunk.get('file_title', '')
            parent_title = chunk.get('parent_title', '')
            body = chunk.get('body', '')
            content = f'{title}\n\n{body}'
            
            # 2. 构建最终chunk对象
            assemble_chunk = {
                'title': title,
                'file_title': file_title,
                'parent_title': parent_title,
                'content': content,
            }
            
            # 3. 判断part是否存在
            if 'part' in chunk:
                assemble_chunk['part'] = chunk['part']
                
            chunks.append(assemble_chunk)
        
        return chunks
        
        
if __name__ == '__main__':
    start_time = time.time()
    setup_logging()

    document_node = DocumentSplitNode()
    file_path = r"D:\atguigu\shopkeer_brain\knowledge\test\test_hierarchy.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = ImportGraphState({
        "file_title": "万用表RS-12的使用",
        "md_content": content,
        "file_dir": r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\output_temp_dir\万用表RS-12的使用\hybrid_auto"
    })
    
    result = document_node.process(state).get('chunks')    