import os
import logging

from typing import Optional
from pymilvus import MilvusClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

client : Optional[MilvusClient] = None

def get_milvus_client() -> Optional[MilvusClient]:
    global client
    
    if client is not None:
        return client
    
    try:
        client = MilvusClient(
            uri=os.getenv("MILVUS_URI", "http://192.168.10.130:19530")
        )
        return client

    except Exception as e:
        logger.error(f"Milvus客户端创建失败: {e}")
        client = None
        return None