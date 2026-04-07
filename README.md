# 掌柜智库 - 知识库问答系统

基于 RAG + 知识图谱的商品知识库智能问答系统，支持 PDF 文档导入、四路混合检索、重排序及流式答案生成。

## 系统架构

```
文档导入流程                        查询流程
──────────────                    ──────────────────────────────────
PDF 上传                           用户提问
  ↓                                  ↓
PDF → Markdown (MinerU)          商品名确认 (LLM + Milvus 向量匹配)
  ↓                                  ↓
图片处理 & 切分                  ┌──────────────────────────┐
  ↓                              │     四路并行检索           │
向量生成 (BGE-M3)                │  向量检索 / HyDE 检索     │
  ↓                              │  知识图谱检索 / 网络搜索   │
导入 Milvus + Neo4j              └──────────────────────────┘
                                          ↓
                                     RRF 融合排序
                                          ↓
                                     BGE 重排序
                                          ↓
                                    LLM 生成答案 (流式/非流式)
```

## 技术栈

| 模块 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 流程编排 | LangGraph |
| LLM | 阿里云 DashScope (Qwen 系列) |
| 向量模型 | BGE-M3 (稠密 + 稀疏双路) |
| 重排序模型 | BGE-Reranker-Large |
| 向量数据库 | Milvus |
| 图数据库 | Neo4j |
| 文档数据库 | MongoDB |
| 对象存储 | MinIO |
| PDF 解析 | MinerU |
| 网络搜索 | Tavily MCP |

## 目录结构

```
knowledge/
├── api/                    # FastAPI 路由
│   ├── import_router.py    # 文档导入接口 (port 8000)
│   └── query_router.py     # 问答查询接口 (port 8001)
├── processor/
│   ├── import_process/     # 文档导入 LangGraph 流程
│   │   └── nodes/          # 各处理节点
│   └── query_process/      # 查询 LangGraph 流程
│       └── nodes/          # 各检索 & 生成节点
├── service/                # 业务服务层
├── schema/                 # Pydantic 请求/响应模型
├── prompts/                # LLM 提示词模板
├── utils/                  # 工具类 (Milvus / Neo4j / MongoDB / SSE 等)
├── core/                   # 依赖注入 & 路径配置
└── front/                  # 前端页面
    ├── import.html         # 文档导入页
    └── chat.html           # 问答页
```

## 快速开始

### 1. 环境准备

需要本地或远程部署以下服务：
- Milvus
- Neo4j
- MongoDB
- MinIO

### 2. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或 pip
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入各服务地址和密钥
```

### 4. 启动服务

```bash
# 文档导入服务
uv run python -m api.import_router

# 问答查询服务
uv run python -m api.query_router
```

### 5. 访问页面

| 页面 | 地址 |
|------|------|
| 文档导入 | http://localhost:8000/import |
| 智能问答 | http://localhost:8001/chat |
| 导入 API 文档 | http://localhost:8000/docs |
| 查询 API 文档 | http://localhost:8001/docs |

## 查询流程说明

1. **商品名确认**：LLM 结合历史对话从用户问题中提取商品名，通过 Milvus 向量匹配对齐并过滤，不确定时反问用户
2. **四路并行检索**：向量检索、HyDE 假设性文档检索、知识图谱检索、Tavily 网络搜索同步执行
3. **RRF 融合**：倒数排名融合合并多路结果
4. **重排序**：BGE-Reranker 精排，分差截断过滤低相关文档
5. **答案生成**：基于参考内容 + 历史对话 + 图谱关系调用 LLM，支持 SSE 流式输出

## SSE 流式协议

流式查询通过 SSE 推送以下事件：

| 事件 | 含义 |
|------|------|
| `ready` | 连接建立 |
| `progress` | 节点执行进度 |
| `delta` | LLM 输出 token |
| `final` | 完整答案（流结束） |
