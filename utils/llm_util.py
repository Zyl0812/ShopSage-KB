import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

cache_llm_client = {}

def get_llm_client(model_name :str = None, temperature: float = 0.0, response_format: bool = False):  # pyright: ignore[reportArgumentType]
    if response_format:
        model_kwargs = {'response_format': {'type': 'json_object'}}
    else:
        model_kwargs = {}
    
    model_name = model_name or os.getenv("ITEM_MODEL", 'qwen-flash')
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    # 将 api_key 转换为 SecretStr
    secret_api_key = SecretStr(api_key) if api_key else None
    
    # 复合缓存key
    cache_key = (model_name, response_format)
    if cache_key in cache_llm_client:
        # 命中缓存，直接返回缓存中的客户端
        return cache_llm_client[cache_key]
    
    client = ChatOpenAI(
        model=model_name,
        api_key=secret_api_key,
        base_url=api_base,
        temperature=temperature,
        extra_body={'enable_thinking': False},
        model_kwargs=model_kwargs
    )
    
    # 缓冲同步
    cache_llm_client[cache_key] = client
    
    return client
    

if __name__ == "__main__":
    llm_client = get_llm_client()
    
    ai_message = llm_client.invoke('你是谁，结果返回json格式：{"model": ""}')
    
    print(ai_message.content)