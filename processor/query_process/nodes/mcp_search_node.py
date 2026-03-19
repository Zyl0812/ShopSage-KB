import asyncio
import json
import logging

from typing import List, Tuple

from fastmcp import Client

from processor.query_process.exceptions import StateFieldError
from processor.query_process.state import QueryGraphState
from processor.query_process.base import BaseNode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MCPSearchNode(BaseNode):
    '''
    负责从网络上查询当前的问题[整个知识库没有找到该问题，用网络搜索的结果兜底]
    通过 Tavily MCP (Streamable HTTP) 进行网络搜索
    '''
    name = 'MCP_search_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1. 参数校验
        validated_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 通过 Tavily MCP 搜索
        search_query = f'{" ".join(validated_item_names)} {validated_query}'
        search_results = self._tavily_mcp_search(search_query)

        # 3. 将搜索结果写入 state
        if search_results:
            state['web_search_docs'] = search_results

        return state


    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:

        rewrritten_query = state.get('rewritten_query', '')
        item_names = state.get('item_names', [])

        if not rewrritten_query or not isinstance(rewrritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewrritten_query, item_names


    def _tavily_mcp_search(self, query: str) -> list:
        """通过 Tavily MCP 进行网络搜索"""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()

            return asyncio.run(self._async_tavily_search(query))
        except Exception as e:
            logger.error(f'Tavily MCP 搜索失败: {e}')
            return []


    async def _async_tavily_search(self, query: str) -> list:
        """异步调用 Tavily MCP Streamable HTTP 服务"""
        mcp_url = f'https://mcp.tavily.com/mcp/?tavilyApiKey={self.config.tavily_api_key}'

        async with Client(mcp_url) as client:
            result = await client.call_tool(
                'tavily_search',
                arguments={
                    'query': query,
                    'search_depth': 'advanced',
                    'max_results': 2
                }
            )

            return self._parse_mcp_result(result)


    def _parse_mcp_result(self, result) -> list:
        """解析 MCP 工具调用返回的结果"""
        docs = []
        if not result or not result.content:
            return docs

        for content_block in result.content:
            if not hasattr(content_block, 'text'):
                continue

            try:
                data = json.loads(content_block.text)
                results = []

                if isinstance(data, dict) and 'results' in data:
                    results = data['results']
                elif isinstance(data, list):
                    results = data

                for item in results:
                    docs.append({
                        'content': item.get('content', ''),
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                    })
            except json.JSONDecodeError:
                docs.append({
                    'content': content_block.text,
                    'url': '',
                    'title': '',
                })

        return docs


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    state = QueryGraphState(
        {
            'rewritten_query': '华为擎云B730如何使用',
            'item_names': ['华为擎云B730 台式计算机']
        }
    )

    node = MCPSearchNode()
    result = node.process(state).get('web_search_docs', [])
    print(result)