import json

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from processor.import_process.base import setup_logging
from processor.import_process.nodes.entry_node import EntryNode
from processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from processor.import_process.nodes.md_img_node import MdImgNode
from processor.import_process.nodes.document_split_node import DocumentSplitNode
from processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from processor.import_process.nodes.bge_embedding_chunks_node import BGEEmbeddingChunksNode
from processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from processor.import_process.nodes.kg_graph_node import KnowledgeGraphNode
from processor.import_process.state import ImportGraphState, create_default_state


# 路由函数
def import_router(state: ImportGraphState):
    if state.get('is_md_read_enabled'):
        return 'md'
    if state.get('is_pdf_read_enabled'):
        return 'pdf'
    return END

def create_import_graph() -> CompiledStateGraph:
    """
    定义导入业务的graph状态拓扑图（langgraph构建流水线）
    """

    # 1. 定义状态图
    graph_pineline = StateGraph(ImportGraphState)

    # 2. 定义节点
    # 2.1 定义入口节点
    graph_pineline.set_entry_point("entry_node")

    # 2.2 添加剩下的节点
    nodes = {
        "entry_node": EntryNode(), 
        "pdf_to_md_node": PdfToMdNode(), 
        "md_img_node": MdImgNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "bge_embedding_chunks_node": BGEEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
        "kg_graph_node": KnowledgeGraphNode(),
    }

    # 2.3 添加节点
    for key, value in nodes.items():
        graph_pineline.add_node(key, value)

    # 3. 定义边
    # 条件边
    graph_pineline.add_conditional_edges('entry_node', import_router, {'md': 'md_img_node', 'pdf': 'pdf_to_md_node', END: END})
    # 顺序边
    graph_pineline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pineline.add_edge("md_img_node", "document_split_node")
    graph_pineline.add_edge("document_split_node", "item_name_recognition_node")
    graph_pineline.add_edge("item_name_recognition_node", "bge_embedding_chunks_node")
    graph_pineline.add_edge("bge_embedding_chunks_node", "import_milvus_node")
    graph_pineline.add_edge("import_milvus_node", "kg_graph_node")
    graph_pineline.add_edge("kg_graph_node", END)

    # 4. 编译
    return graph_pineline.compile()

kb_import__graph_app = create_import_graph()

if __name__ == "__main__":
    setup_logging()
    # 1. 直接得到可执行图
    compiled_graph = create_import_graph()
    
    # 2. 构建state
    state = {
        "import_file_path": r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf",
        "file_dir": r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\output_temp_dir",
    }
    init_state = create_default_state(**state)
    
    # 3. 调用stream（用流式获取每一个节点的处理情况：event事件[节点名字 节点处理后的状态]）
    final_state = None
    for event in kb_import__graph_app.stream(init_state):
        for node_name, state in event.items():
            print(f"运行节点: {node_name}, State: {state}")
            final_state = state
    
    print(json.dumps(final_state, indent=2, ensure_ascii=False))
    
    # 4. 打印图结构（ASCII可视化）
    print('-' * 50)
    print("图结构：")
    compiled_graph.get_graph().print_ascii()