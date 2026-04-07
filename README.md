# ShopSage-KB - 知识库问答系统

基于 **RAG + 知识图谱**的商品文档知识库智能问答系统，专为家电/仪器仪表等垂直行业的商品知识检索场景设计。

系统分为**文档导入**和**智能问答**两条主链路：

- **导入链路**：上传 PDF/MD 格式的商品手册，经 MinerU 解析、文档切分、BGE-M3 双路向量化后，分别写入 Milvus（向量检索）和 Neo4j（知识图谱），实现结构化与非结构化知识的统一入库。
- **问答链路**：用户提问后，系统先通过 LLM 结合多轮历史对话提取并确认商品名（防止指代歧义），再并行发起四路检索——**稠密向量检索、HyDE 假设性文档检索、知识图谱检索、Tavily 网络实时搜索**，经 RRF 融合 + BGE-Reranker 重排序后，将最相关的上下文连同图谱关系和对话历史一起送入 LLM 生成答案，全程支持 **SSE 流式推送**，前端可实时感知节点执行进度和 token 输出。

## 技术亮点

- **多轮对话感知的商品名确认**：LLM 提取商品名时融入历史对话，解决"它怎么用？"等指代消解问题；通过 BGE-M3 向量匹配 + 双阈值评分对齐 + 分差过滤，区分精确命中、候选模糊和无法识别三种情况，分别导向继续检索、反问用户或提示重新输入，避免错误答案扩散。

- **四路异构检索并行**：稠密向量检索（语义相似）、HyDE 假设性文档检索（扩增召回）、知识图谱检索（结构化实体关系）、Tavily 网络实时搜索（补充时效信息）四路同步执行，覆盖不同知识形态，经 RRF 倒数排名融合消除跨来源分数量纲差异。

- **BGE-Reranker 精排 + 分差截断**：重排序后按相邻文档分差动态截断，自动过滤低相关长尾文档，而非依赖固定 Top-K，保证送入 LLM 的上下文质量。

- **char_budget 上下文预算机制**：生成提示词时对重排序文档、历史对话、图谱关系三类来源分级分配字符预算，优先保留高质量检索结果，防止超出 LLM 上下文窗口。

- **SSE 流式全链路推送**：节点执行进度（progress）、LLM token 增量（delta）、最终答案（final）均通过 SSE 独立事件推送，前端可实时渲染进度条和流式文字，后端使用线程队列桥接同步 LangGraph 与异步 FastAPI。

- **LangGraph 并行编排 + Annotated State 并发安全**：查询流程基于 LangGraph 构建，四路检索节点真正并行执行；State 字段统一使用 `Annotated[T, reducer]` 声明合并策略，避免并行分支汇合时的状态冲突。

## 系统架构

![架构图](./docs/architecture.drawio.svg)

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
