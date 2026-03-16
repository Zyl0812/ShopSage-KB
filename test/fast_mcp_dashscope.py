import asyncio
import json

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

from knowledge.processor.config import QueryConfig, get_query_config


async def web_mcp_search(config: QueryConfig, params):
    server_url = config.mcp_dashscope_base_url

    auth = BearerAuth(token=config.openai_api_key)

    async with Client(server_url, auth=auth) as client:
        tools = await client.list_tools()
        print("Tools:", tools)

        # 如果成功，再试调用
        exec_result = await client.call_tool("bailian_web_search", params)
        if not exec_result:
            return None
        text_content: str = exec_result.content[0].text
        return text_content


def load_data(text_content: str):
    try:
        json_result = json.loads(text_content)
        if not json_result:
            return []
        pages = json_result.get("pages", [])
        search_result = []
        for page in pages:
            snippet = page.get("snippet", "")
            title = page.get("title", "")
            url = page.get("url", "")
            search_result.append({"snippet": snippet, "title": title, "url": url})
        return search_result
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    steup = get_query_config()
    par = {"query": "今天2026年3月16日的小米汽车的股价是多少", "count": 2}
    par = {"query": "今天东京天气怎么样", "count": 2}
    result = asyncio.run(web_mcp_search(steup, par))
    result = load_data(result)
    for r in result:
        print(json.dumps(r, ensure_ascii=False, indent=2))
