import logging

from typing import List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_process.exceptions import StateFieldError
from utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from utils.bge_me_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from utils.llm_util import get_llm_client
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class KnowledgeGraphSearchNode(BaseNode):
    name = 'knowledge_graph_search_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:
        