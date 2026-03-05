from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

md_path = Path(r"D:\atguigu\shopkeer_brain\knowledge\test\test_hierarchy.md")
md_content = md_path.read_text(encoding="utf-8")

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ],
    strip_headers=False,
    return_each_line=False,
)

docs = splitter.split_text(md_content)
print(docs)
for doc in docs:
    print(doc)