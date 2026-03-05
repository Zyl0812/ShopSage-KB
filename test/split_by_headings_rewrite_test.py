"""
用 MarkdownHeaderTextSplitter 重写 _split_by_headings 的 demo

原方法手动逐行解析 Markdown，维护 hierarchy 数组来追踪父标题层级。
MarkdownHeaderTextSplitter 会在每个 Document 的 metadata 中直接记录所有祖先标题，因此可以很方便地从 metadata 中提取 title 和 parent_title。
MarkdownHeaderTextSplitter输出结构：
    [Document(metadata={'H1': '1. 核心业务系统介绍', 'H2': '1.1 订单模块'}, page_content='# 1. 核心业务系统介绍  \n## 1.1 订单模块\n订单模块负责处理所有的交易。'),...]
"""
import re

from pathlib import Path
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

# H1~H6 对应关系（顺序即层级深度）
HEADERS_TO_SPLIT = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
    ("#####", "H5"),
    ("######", "H6"),
]


def split_by_headings(md_content: str, file_title: str) -> List[dict]:
    """
    用 MarkdownHeaderTextSplitter 重写 _split_by_headings。
    Args:
        md_content:  Markdown 文本
        file_title:  文件标题（无标题 section 的兜底值）

    Returns:
        List[dict]:
            {
                'title':       当前标题（含 # 前缀），无标题时为 file_title
                'body':        标题之后的正文
                'file_title':  文件标题
                'parent_title': 父标题（含 # 前缀），找不到时为 title 本身
            }
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
        return_each_line=False,
    )
    docs = splitter.split_text(md_content)

    sections = []
    for doc in docs:
        meta = doc.metadata   # e.g. {'H1': '核心业务系统介绍', 'H2': '订单模块'}
        body = doc.page_content
        # 去掉 body 开头的标题行（如 "## 1. 水果区 (Level 2)"）
        body = re.sub(r'^#{1,6} .*\n?', '', body, count=1)

        # 1. 找当前标题和父标题
        title = file_title
        if meta:
            keys = list(meta)
            # metadata中最后一个标题为当前标题
            last_key = keys[-1]
            # metadata中倒数第二个标题为父标题，若无倒数第二个标题则父标题为当前标题
            parent_key = keys[-2] if len(keys) >= 2 else keys[-1]
        
            title = meta[last_key]
            parent_title = meta[parent_key]

        # 2. 无meta数据时说明当前内容没有实际标题，标题和父标题均为file_title
        else:
            title = parent_title = file_title

        sections.append({
            'title': title,
            'body': body,
            'file_title': file_title,
            'parent_title': parent_title
        })

    return sections

        
if __name__ == '__main__':
    md_path = Path(r"D:\atguigu\shopkeer_brain\knowledge\test\test_hierarchy.md")
    md_content = md_path.read_text(encoding="utf-8")
    file_title = "test_hierarchy"
    
    sections = split_by_headings(md_content, file_title)
    # print(sections)
    for i, sec in enumerate(sections):
        print(f"{'='*60}")
        print(f"Section {i + 1}")
        print(f"  title       : {sec['title']}")
        print(f"  parent_title: {sec['parent_title']}")
        print(f"  file_title  : {sec['file_title']}")
        body_preview = sec['body'].replace('\n', '\\n')[:100]
        print(f"  body        : {body_preview}")
    print('='*60)
    print(f"共 {len(sections)} 个 section")
    