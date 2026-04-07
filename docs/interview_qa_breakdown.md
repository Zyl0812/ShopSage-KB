# 知识库项目面试问答版拆解

这份文档不是 README，也不是 PPT，而是一份面向技术面试的答题稿。内容按 6 个固定问题组织，所有判断都对应当前仓库里的实际实现。

## 1. 项目是做什么的

面试官如果让我先用 1 分钟介绍这个项目，我会这样说：

这是一个面向企业知识库场景的导入加检索问答系统，核心目标是把 PDF 或 Markdown 文档加工成可检索的知识资产，再通过多路检索和大模型生成答案。

它不是单纯的聊天机器人，而是拆成了两条主流程：

1. 导入流程：把文件上传后转换、切片、识别商品或产品名称、写入向量库，同时抽取知识图谱关系。
2. 查询流程：先识别用户问题里的商品名称，再并行做向量检索、HyDE 检索、知识图谱检索和网络检索，最后融合排序并生成答案。

如果用一句话概括，这个项目本质上是在做“企业文档知识化”和“面向具体产品问题的检索增强问答”。

对应事实：

- 导入接口在 `api/import_router.py`
- 查询接口在 `api/query_router.py`
- 导入工作流在 `processor/import_process/main_graph.py`
- 查询工作流在 `processor/query_process/main_graph.py`

## 2. 系统架构怎么分层

如果面试官问系统怎么分层，我会按 5 层来讲。

第一层是接口层。

- 导入服务暴露了 `/import`、`/upload`、`/status/{task_id}`，在 `api/import_router.py`
- 查询服务暴露了 `/chat.html`、`/query`、`/stream/{task_id}`、`/history/{session_id}`，在 `api/query_router.py`

第二层是应用服务层。

- `service/import_file_service.py` 负责上传文件、保存本地和 MinIO、触发导入图
- `service/query_service.py` 负责生成会话和任务、提交查询、执行查询图、读取答案和历史记录
- `service/task_service.py` 负责查询和更新任务状态

第三层是流程编排层。

- 导入流程用一套 LangGraph，在 `processor/import_process/main_graph.py`
- 查询流程用另一套 LangGraph，在 `processor/query_process/main_graph.py`
- 两套流程都把节点抽象成 `BaseNode`，分别定义在 `processor/import_process/base.py` 和 `processor/query_process/base.py`

第四层是能力和存储层。

- MinIO 存原始上传文件和图片资源，代码在 `utils/minio_util.py`
- Milvus 存文档切片向量和实体或商品名向量，代码在 `utils/milvus_util.py`
- Neo4j 存知识图谱关系，代码在 `utils/neo4j_util.py`
- MongoDB 存会话历史，代码在 `utils/mongo_history_util.py`
- LLM 调用封装在 `utils/llm_util.py`

第五层是前端演示层。

- 文件导入页面在 `front/import.html`
- 问答页面在 `front/chat.html`

我会补一句，这个项目现在不是一个统一单体服务，而是两个独立 FastAPI 服务加两套图编排，业务链路是清楚的，但工程边界还比较分散。

## 3. 导入链路怎么跑

如果面试官追问数据是怎么导进去的，我会按下面这条链讲。

1. 用户通过 `/upload` 上传文件，`service/import_file_service.py` 先把文件保存到本地临时目录，再同步到 MinIO。
2. 上传完成后，后台任务启动导入图，入口在 `processor/import_process/main_graph.py`。
3. `entry_node` 先判断是 Markdown 还是 PDF，决定后续分支，节点在 `processor/import_process/nodes/entry_node.py`。
4. 如果是 PDF，就先经过 `pdf_to_md_node` 做 PDF 转 Markdown，节点在 `processor/import_process/nodes/pdf_to_md_node.py`。
5. 然后统一经过 `md_img_node`，扫描 Markdown 里的图片，调用视觉模型做图片摘要，再把摘要回写到 Markdown，节点在 `processor/import_process/nodes/md_img_node.py`。
6. 接着 `document_split_node` 会按照标题和长度规则切片，生成后续检索要用的 chunks，节点在 `processor/import_process/nodes/document_split_node.py`。
7. `item_name_recognition_node` 会结合文件标题和部分切片内容识别商品或产品名称，并把商品名向量写到 Milvus，节点在 `processor/import_process/nodes/item_name_recognition_node.py`。
8. `bge_embedding_chunks_node` 对切片批量做 dense 加 sparse 向量化，节点在 `processor/import_process/nodes/bge_embedding_chunks_node.py`。
9. `import_milvus_node` 把切片写入 Milvus 的知识库集合，节点在 `processor/import_process/nodes/import_milvus_node.py`。
10. 最后 `kg_graph_node` 会从切片中抽取实体和关系，同时写入 Neo4j 和实体向量集合，节点在 `processor/import_process/nodes/kg_graph_node.py`。

我在面试里会强调两点：

- 这条链不是普通 ETL，而是“文档加工 + 语义索引 + 图谱构建”的复合导入流程。
- 商品名识别是个关键设计，因为它直接影响后面查询时的召回范围和对齐效果。

## 4. 查询链路怎么跑

如果面试官问查询侧是怎么回答问题的，我会按检索增强的流程讲。

1. 前端调用 `/query`，请求定义在 `schema/query_schema.py`，接口在 `api/query_router.py`。
2. `service/query_service.py` 会生成 `session_id` 和 `task_id`，流式模式下还会创建 SSE 队列。
3. 查询图入口是 `processor/query_process/main_graph.py`，第一个节点是 `item_name_confirm`，实现放在 `processor/query_process/nodes/item_name_confirm_node.py`。
4. 这个节点先抽取问题里的商品名，再用 Milvus 的商品名集合做相似匹配；如果商品名不明确，可以直接返回候选或提前形成答案。
5. 如果还需要继续检索，就进入并行搜索阶段：
6. `vector_search_node` 做标准向量混合检索，节点在 `processor/query_process/nodes/vector_search_node.py`。
7. `hyde_search_node` 先构造假设性文档，再去做 HyDE 检索，节点在 `processor/query_process/nodes/hyde_search_node.py`。
8. `kg_search_node` 从知识图谱里找实体关系，再回填相关 chunk，节点在 `processor/query_process/nodes/kg_search_node.py`。
9. `mcp_search_node` 通过 Tavily MCP 做网络搜索兜底，节点在 `processor/query_process/nodes/mcp_search_node.py`。
10. 并行结果汇合后，`rrf_node` 先做 RRF 融合，节点在 `processor/query_process/nodes/rrf_node.py`。
11. `rerank_node` 再做重排序，把真正相关的文档提到前面，节点在 `processor/query_process/nodes/rerank_node.py`。
12. `answer_output_node` 负责组装 prompt、调用 LLM 生成答案、流式推送结果，并把用户问题和回答写入 Mongo 历史记录，节点在 `processor/query_process/nodes/answer_output_node.py`。

我会把这个流程总结成一句话：

它不是只靠单一向量检索，而是把商品名确认、混合检索、图谱检索、网络兜底、融合排序和答案生成串成了一条完整问答链。

## 5. 现阶段最大问题是什么

如果面试官问“你怎么看这个项目的现状”，我会先肯定业务思路，再明确讲工程问题。这里我会先讲最关键的 4 个，再补充另外 4 个。

第一类，入口和部署边界分散。

- `main.py` 只是一个输出 `Hello from knowledge!` 的占位文件，不是真实入口。
- 真正可启动的服务分散在 `api/import_router.py` 和 `api/query_router.py`，而且端口还是 8000 和 8001 两套。
- 这会直接提高接手成本、部署复杂度和前后端联调复杂度。

第二类，任务状态设计只适合单进程演示。

- `utils/task_util.py` 里的 `_tasks_running_list`、`_tasks_done_list`、`_tasks_result`、`_tasks_status` 都是进程内存变量。
- 这意味着一旦服务重启，任务状态就丢失；如果开多进程或多实例，请求也拿不到统一状态。
- 所以它适合本地 demo，不适合真正生产。

第三类，仓库安全和产物治理有风险。

- 当前 `.env` 已被 git 跟踪。
- `processor/import_process/import_temp_dir`
- `processor/import_process/output_temp_dir`
- `test/`

这些目录和文件里已经有样例输入、输出和测试脚本被跟踪进仓库。说明当前仓库同时混入了配置、样例产物和实验文件，容易带来安全、仓库膨胀和环境污染问题。

第四类，依赖声明和实际代码不完全一致。

- `pyproject.toml` 声明了 LangGraph、Milvus、Neo4j、Mongo 等依赖。
- 但接口层代码实际直接使用了 `fastapi`、`uvicorn`、`python-dotenv`，这些不在 `pyproject.toml` 的 dependencies 里。
- `requirements.txt` 也只列了部分依赖，和 `pyproject.toml` 不是统一来源。

这会导致环境可复现性差，新同学拉代码后不一定能一次装齐。

除此之外，我还会补充 4 个次级但很真实的问题。

- `test/` 目录下大多是连接性脚本和实验代码，比如 `test/docker_test.py`、`test/pymilvus_test.py`、`test/neo4j_test.py`，它们更像手工验证，不是稳定回归测试体系。
- 多个节点职责偏重，尤其 `processor/import_process/nodes/kg_graph_node.py` 和 `processor/query_process/nodes/answer_output_node.py` 同时承担了较多业务编排、格式处理和外部调用逻辑，可维护性一般。
- 项目对外部依赖非常强，包括 Milvus、Neo4j、Mongo、MinIO、Tavily MCP 和兼容 OpenAI 的模型服务，本地开发和 CI 的可复现性比较弱。
- 当前前端只是演示页，接口、状态、存储和页面耦合在一起，离标准化产品化还有距离。

如果面试官追问“那测试到底算不算完善”，我会直接回答：

不算完善。现在更多是连通性验证和实验脚本，能证明作者在打通链路，但不能证明系统具备稳定回归能力。

## 6. 如果继续做会怎么改

如果面试官问“你接手后会怎么改”，我会按优先级给 4 条路线，而不是泛泛地说重构。

第一优先级，合并入口与配置管理，统一启动方式。

- 目标是让项目有一个明确启动入口，而不是靠两个 router 文件各自 `uvicorn.run(...)`。
- 同时把配置读取统一起来，避免 `pyproject.toml`、`requirements.txt`、`.env` 和代码里的配置项各自漂移。

第二优先级，把任务状态迁到 Redis 或数据库。

- 先把 `utils/task_util.py` 的内存态任务状态替换成 Redis 或持久化存储。
- 这样才能支持服务重启恢复、跨实例查询状态和更稳定的流式任务追踪。

第三优先级，补齐依赖清单、环境模板和最小可运行文档。

- 把依赖收敛到一个统一来源。
- 提供 `.env.example` 而不是把真实 `.env` 混在仓库里。
- 给出最小启动说明，明确哪些外部服务是必需的，哪些可以 mock。

第四优先级，建立分层测试体系。

- 纯函数或工具层做单元测试。
- 图节点做集成测试，重点覆盖输入输出状态变化。
- Milvus、Neo4j、Mongo、MinIO 的连接验证单独保留为 smoke test，不和业务回归测试混在一起。

如果面试官让我补一句“为什么这样排优先级”，我会回答：

因为第一阶段先解决可启动、可部署、可观测的问题；第二阶段解决状态可靠性；第三阶段解决环境复现；第四阶段再把研发效率和回归质量拉起来。

## 面试时可以这样收尾

这个项目的亮点不在于前端，而在于它把知识导入、向量检索、知识图谱和大模型问答串成了一个完整闭环。它的短板也很明显，主要集中在工程化、可运维性和测试体系上。

所以如果面试官问我对这个项目的评价，我会说：

业务方向是对的，技术路线也是成立的，已经做出了一个能跑通的知识库问答原型；但从原型走向可交付系统，还需要补齐入口统一、状态持久化、依赖治理和测试体系这几块工程基础。
