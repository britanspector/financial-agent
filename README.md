# Financial Agent

用于深度学习和求职展示的本地多工具 Agent 工程探索。全部用户数据为 synthetic，不连接真实公司内部系统。

**当前阶段：Phase 2 LangGraph Agent Skeleton 实现完成。** 现有 4 个用户数据 Tool、2 个行情 Tool 和 3 个 RAG Tool 已通过组合 Registry 接入依赖感知的并行执行图；Task 目前由调用方人工构造，不包含 LLM Planner。

## 项目结构与依赖

```text
financial-agent/
├── .env.example / .gitignore / pyproject.toml / README.md
├── src/financial_agent/
│   ├── __init__.py / __main__.py / config.py
│   ├── schemas.py / logging_config.py
│   ├── agent/                  # AgentState、Task、LangGraph execution graph
│   ├── demo_faults.py           # 仅测试/demo 使用的故障序列
│   ├── tools/                  # ToolResult、ToolRegistry、CompositeToolRegistry
│   ├── user_data/              # models、fixtures、repository、auth、audit、service、runtime
│   ├── market_data/            # Market models、Normalizer、TushareProvider、Service
│   └── knowledge/              # manifest schema、Markdown ingestion、source-aware chunking
├── tests/
│   ├── conftest.py / test_config.py / test_schemas.py
│   ├── test_logging.py / test_bootstrap.py
│   ├── user_data/              # 数据、权限、Tool、故障、审计、CLI
│   └── market_data/            # REST Provider、Tool 与 live test
└── data/
    └── knowledge/              # 72 份 synthetic Markdown 和 manifest.json
```

Python >=3.11；本机验证使用 Python 3.13.5。依赖范围定义在 pyproject.toml。

| 依赖 | 用途 |
| --- | --- |
| jieba >=0.42,<1 | 中文 BM25 分词 |
| langgraph >=1.0,<2 | Task 依赖调度、并行 fan-out 和结果归并 |
| numpy >=2,<3 | 本地 dense cosine similarity 与 embedding index |
| pydantic >=2.10,<3 | schema、输入与结果校验 |
| pydantic-settings >=2.7,<3 | 环境变量和可选 .env 加载 |
| pytest >=8,<10（dev） | 自动化测试 |
| setuptools >=77,<83（build） | 包构建及 editable 安装 |

Market Data 复用 `httpx` 直连 Tushare REST，不安装 Tushare/AKShare SDK、ORM 或 RAG 依赖。

## 安装与验证

在仓库根目录运行（Windows PowerShell，无需激活虚拟环境）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# 可选，仅首次创建
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m financial_agent
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

macOS / Linux 使用 .venv/bin/python 替换解释器路径。也可通过安装后的 financial-agent 命令启动。无参数自检保留 Phase 0 行为：stdout 为 `{"phase": 0, "status": "ready", "data_mode": "synthetic"}`，日志写入 stderr；无需 API key，不访问外部服务。首次安装依赖需要联网。

配置优先级：显式 Settings 构造参数 > 进程环境变量 > 当前工作目录的 .env > 默认值。只支持 synthetic 模式；API key 使用 SecretStr 且从配置序列化中排除。.env 已忽略，不提交真实凭证。

## 生成数据与独立调用 Tool

```powershell
.\.venv\Scripts\python.exe -m financial_agent generate-synthetic-data --overwrite
.\.venv\Scripts\python.exe -m financial_agent list-tools
```

正式 synthetic 数据库默认是 `data/synthetic-2000.db`，固定 seed 为 `20260910`；生成器会在同目录构建临时数据库后原子发布，显式 `--overwrite` 才会覆盖旧库。旧的 `seed-user-data` 仅保留为早期 fixture 命令，四个业务 Tool 使用 `generate-synthetic-data` 生成的数据库。

在当前 PowerShell 进程环境中生成测试 key 和服务端权限映射：

```powershell
$env:FINANCIAL_AGENT_CALLER_API_KEY = 'synthetic-' + [guid]::NewGuid().ToString('N')
$env:FINANCIAL_AGENT_USER_API_KEYS = ConvertTo-Json -Compress -Depth 4 -InputObject @(
    @{
        api_key = $env:FINANCIAL_AGENT_CALLER_API_KEY
        principal_id = 'synthetic-demo'
        user_ids = @('syn-user-0001', 'syn-user-0002', 'syn-user-0003')
        scopes = @('read:customer_context', 'read:margin_account', 'read:portfolio_positions', 'read:portfolio_analytics')
    }
)

.\.venv\Scripts\python.exe -m financial_agent call-tool get_customer_context --user-id syn-user-0001
.\.venv\Scripts\python.exe -m financial_agent call-tool get_margin_account --user-id syn-user-0001 --start-date 2026-05-01 --end-date 2026-06-30 --limit 60 --offset 0
.\.venv\Scripts\python.exe -m financial_agent call-tool get_portfolio_positions --user-id syn-user-0001
.\.venv\Scripts\python.exe -m financial_agent call-tool get_portfolio_analytics --user-id syn-user-0001
```

FINANCIAL_AGENT_USER_API_KEYS 是宿主维护的 JSON 数组，条目字段为 api_key、principal_id、user_ids、scopes。默认 [] 拒绝全部访问；重复 key、非法配置和未知 scope 被拒绝。HTTP Client 从 FINANCIAL_AGENT_CALLER_API_KEY 读取调用 API Key，并通过 X-API-Key Header 发送；scopes 和白名单来自 Server 宿主配置，不能由 Agent 参数自报。

路径变量 FINANCIAL_AGENT_USER_DB_PATH、FINANCIAL_AGENT_AUDIT_PATH 默认分别为 data/synthetic-2000.db、data/audit.jsonl。HTTP Client 配置为 FINANCIAL_AGENT_USER_DATA_BASE_URL（默认 http://127.0.0.1:8000）和 FINANCIAL_AGENT_USER_DATA_TIMEOUT_SECONDS（默认 5 秒）。这些路径相对当前工作目录解析。

启动本地 HTTP Server：

```powershell
.\.venv\Scripts\python.exe -m financial_agent serve-user-data
```

成功的 HTTP 响应直接返回业务领域模型；错误响应使用 transport 层 `ApiError`。Agent Tool 调用仍由 HTTP Client 转换为统一 `ToolResult`。Client 使用 `httpx.AsyncClient`，当前同步 ToolRegistry 通过明确的同步边界调用它，不执行自动重试。

Python 直接调用（凭证仍来自环境配置）：

```python
from financial_agent.config import Settings
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.runtime import build_user_tools

settings = Settings()
registry = build_user_tools(settings)  # 通过本地 FastAPI 服务调用
result = registry.invoke(
    "get_portfolio_positions",
    {"user_id": "syn-user-0001"},
    context=CallContext(api_key=settings.caller_api_key),
)
print(result.model_dump_json())
```

## Synthetic 业务语义

数据集包含 2,000 个用户、60 个交易日和八类业务数据。archetype 只存在于生成器内部，最终数据库不保存 archetype、persona 或 user_type 标签。客户权限字段 `max_allowed_product_risk_level` 表示客户允许购买的最高产品风险等级，不是 API Key scope，也不是当前持仓风险。

观察期假设没有外部入金/出金，因此 `portfolio_return = period_net_pnl / beginning_equity`。账本满足 `period_net_pnl = realized_pnl + ending_unrealized_pnl - beginning_unrealized_pnl - financing_interest - securities_lending_fee - transaction_fees - other_fees`。`beginning_unrealized_pnl` 是 synthetic accounting simplification 的汇总指标，不代表完整期初持仓快照。

排名按 `region + asset_bucket` 组成 peer group；`return_rank`、`drawdown_rank`、`risk_adjusted_rank` 均为组内 competition ranking，rank 1 为最好，`peer_group_size` 为实际组大小。收益、回撤和风险调整收益均基于数据库实际保存值计算。

## 数据样本与访问边界

用户 ID 为 `syn-user-0001` 到 `syn-user-2000`。每个用户有客户上下文、两融账户快照、60 日 margin history、行业/个股持仓、收益风险指标、因子暴露和 peer-group 排名；不同用户的资产规模、杠杆、集中度、换手、持有期和风险表现由内部生成规则组合产生。

身份、区域和证券标识都是 synthetic。users / accounts / holdings / transactions 四张表设置主键、外键和查询索引。金额、价格、数量存为十进制文本，读取为 Decimal，JSON 输出字符串；组合与交易声明 currency: CNY。这是固定快照，不是完整会计账本。

调用路径为 ToolRegistry → HttpUserDataClient → FastAPI → UserDataService → UserDataRepository → SQLiteUserDataRepository。Server 继续使用只读 repository 和参数绑定；repository 是受信任的应用内部接口，不能绕过 service 暴露为 Agent Tool。SQLite 文件不承担操作系统级访问隔离。

## Tool 契约与错误分类

| Tool | 输入 | scope |
| --- | --- | --- |
| get_customer_context | user_id | read:customer_context |
| get_margin_account | user_id、可选 start_date / end_date / limit / offset | read:margin_account |
| get_portfolio_positions | user_id | read:portfolio_positions |
| get_portfolio_analytics | user_id | read:portfolio_analytics |

拒绝未知输入字段与空 user_id。`get_margin_account` 的日期范围为 `[start_date, end_date)`，limit 默认 60、范围 1–366，offset 默认 0；页数据与 total 分开返回，翻过最后一页仍保留 total。

ToolResult[T] 固定包含 status / data / source / latency / error / request_id：

- status 为 success / empty / error，source 为 synthetic_user_db。
- latency 是单调时钟测得的执行毫秒数，不含最终审计写入；request_id 为每次调用新建的 UUID。
- error 为 null 或包含 code / message / http_status / retryable 的对象；错误时 data 为 null。
- 空结果保留合法数据结构。组合无账户或全部账户无持仓时为 empty，账户及现金余额仍返回；交易页无记录为 empty；档案可空字段为 null 时仍为 success。
- CLI 对 success/empty 返回退出码 0，工具错误返回 1。

| 场景 | code | HTTP 语义码 | retryable |
| --- | --- | --- | --- |
| 缺失/错误 key | UNAUTHORIZED | 401 | false |
| scope 或用户白名单不足 | FORBIDDEN | 403 | false |
| 白名单内用户不存在 | NOT_FOUND | 404 | false |
| 工具名不存在 | UNKNOWN_TOOL | 404 | false |
| 参数不合法 | INVALID_ARGUMENT | 422 | false |
| 模拟超时 | TIMEOUT | 504 | true |
| 模拟限流 | RATE_LIMITED | 429 | true |
| 模拟临时故障 | TEMPORARY_FAILURE | 503 | true |
| 数据库不可读/缺失 | DATA_UNAVAILABLE | 503 | true |
| 审计不可写 | AUDIT_UNAVAILABLE | 500 | false |
| 未预期的内部错误 | INTERNAL_ERROR | 500 | false |

执行顺序为鉴权 → scope → 参数校验 → 用户白名单 → 故障 hook → repository。未授权不读库、不消耗故障；未获授权的不存在用户也返回 403。DATA_UNAVAILABLE 的 retryable 只是错误分类，本步不会自动重试或修复配置。

每次调用尝试追加一条 JSONL 审计，记录 UTC 时间、请求 ID、工具、已验证身份、校验后的目标 ID、结果码和耗时。认证失败不记录身份，未知工具统一记录 unknown。不记录 key、完整参数或结果正文，错误消息不回显 SQL、路径和异常原文。审计写入失败时丢弃数据并返回 AUDIT_UNAVAILABLE；这次审计无法保证落盘。

## LangGraph Agent execution skeleton

Phase 2 使用调用方人工构造的 `Task(task_id, tool_name, arguments, dependencies)`，不调用 Planner。执行状态 `financial_agent.agent.AgentState` 包含 query、history、tasks、tool_results、errors、final_output 和 iteration_count；调用凭证通过 Graph 构建参数注入，不进入 state 或结果。

Graph 拓扑为 `START → dispatcher → execute_tool → collect_results → dispatcher`，完成后进入 `finalize → END`。Dispatcher 每轮通过 LangGraph `Send` 将所有无未完成依赖的 Task 并行 fan-out；`tool_results` 和 `errors` reducer 只接收节点产生的增量，`collect_results` 不重写已有 Result Pool。依赖失败会生成 `DEPENDENCY_FAILED` 并阻断下游；无法解析的依赖通过错误 reason 区分 `missing_dependency` 和 `cycle_or_deadlock`。

`CompositeToolRegistry` / `merge_registries()` 仅按工具名路由到原有 User、Market、RAG Registry，不修改 Phase 1 `ToolRegistry` 的注册、校验、鉴权、审计或错误行为。完整运行时可通过 `build_agent_tools(settings)` 组合全部 9 个 Tool；对应的行情和 RAG Provider 仍要求环境变量凭证及已构建的 embedding index。

```python
from financial_agent.agent import Task, run_execution_graph
from financial_agent.schemas import UserQuery

state = run_execution_graph(
    UserQuery(query="查询客户持仓"),
    [Task(
        task_id="positions",
        tool_name="get_portfolio_positions",
        arguments={"user_id": "syn-user-0001"},
        dependencies=[],
    )],
    registry,
    context=call_context,
)
print(state.final_output.model_dump_json())
```

## HTTP 错误与审计

HTTP API 不暴露 Agent `ToolResult`：成功响应是领域模型，失败响应是 `ApiError`。Client 将 401、403、404、422、5xx 以及 timeout/连接失败恢复为 Agent 层 `ToolResult`。每次 Client 调用生成 `X-Request-ID`，Server 将其用于业务 ToolResult 和 JSONL 审计；transport 错误仅通过日志记录，不伪造业务 AuditSink 事件。

## Market Data v0.1

Market Tool 使用 `MarketDataService → TushareProvider → Tushare REST API`，不经过 User Data HTTP Service，也不做用户数据 API Key、scope 或用户白名单鉴权。公开注册函数为 `build_market_tools(settings)`；Tushare token 仅从 `FINANCIAL_AGENT_TUSHARE_TOKEN` 注入。

两个 Tool 为 `get_market_snapshot` 和 `get_market_history`。当前 120 积分版本只支持沪深 A 股：symbol 在内部统一为 `600519.SH`、`000001.SZ`；不支持指数、实时行情或 qfq/hfq。history 输入只包含 `symbol/start_date/end_date`，日期区间为 `[start_date, end_date)`，结果中的 `source` 固定为 `tushare`、`asset_type` 固定为 `stock`、`adjustment` 固定为 `none`。

`get_market_snapshot` 是日终快照：Provider 向前查询 90 个自然日并选择最新 daily 记录，`snapshot_kind=daily_close`。`as_of` 表示对应交易日 15:00 Asia/Shanghai，不表示 Tushare 实际入库时间。Tushare 的 `vol` 单位为手、`amount` 单位为千元；领域模型分别转换为股和元。

本地配置示例（不要提交真实 token）：

```powershell
$env:FINANCIAL_AGENT_TUSHARE_TOKEN = "<rotated-token>"
```

TushareProvider 使用同步 `httpx.Client` 直接 POST `https://api.tushare.pro`，解析 `code/msg/data.fields/data.items`，不依赖 Tushare SDK、pandas 或 DataFrame，也不自动重试。进入后续 LangGraph 并行 Tool 阶段后，再评估线程池隔离。

Market Tool 的 `EMPTY_RESULT` 目前没有 HTTP transport；如果 `ToolError.http_status=404`，它只是兼容现有 Agent 错误模型的元数据，不代表本阶段提供 HTTP endpoint。

默认测试不会访问真实行情。真实 Tushare smoke test 单独运行；未配置 token 时自动跳过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m live
```

## Multi-source RAG v0.1 数据与 ingestion

`data/knowledge/manifest.json` 管理 24 篇研究报告、24 个 FAQ 和 24 份公告/政策。公司、券商、机构及正文均为 synthetic；metadata 不包含检索难度、期望答案或其他测试标签。研报覆盖 8 家公司和跨时间、跨券商观点，其中 8 篇包含 Markdown 表格；FAQ 保留相近问题与生效版本；政策保留发布日期、生效日期和当前状态。

`financial_agent.knowledge.KnowledgeIngestor` 严格读取和校验 manifest，拒绝重复 ID、重复路径、越界路径、缺失文件和不匹配 metadata。第一版 chunking 保持 FAQ 整篇，研究报告按 Markdown heading 切分且长段按句切分，公告/政策额外识别条款边界；Markdown 表格保持为整体。默认 corpus 产生 96 个稳定 Chunk：研究报告 48、FAQ 24、公告/政策 24。

本地检索链为中文 Jieba BM25 + NumPy cosine dense retrieval → RRF → 独立 reranker provider，输出统一 `Evidence[]`，不生成最终回答。研究、公告政策和业务 FAQ 分别通过三个独立 Tool contract 暴露，metadata/time filter 在召回前执行。Research 使用独立 `ResearchRetrievalPolicy`，按 query 命中的公司实体数选择 12/5、20/8 或 30/10 的候选/最终 Top-K。

Embedding index 使用 corpus、Chunk 内容、模型名和维度生成指纹，并将 float32 NumPy 数据与 JSON metadata 持久化到 Git 忽略的本地目录。缺失、模型变化、维度变化、Chunk 顺序或内容变化都会判定为 missing/stale，要求重新构建。

Qwen HTTP adapter 默认使用 `qwen3.7-text-embedding` 1024 维和 `qwen3.7-text-rerank`；将 embedding model 改成 `qwen3.7-text-embedding-flash` 即可切换低成本版本，模型变化会使旧索引自动 stale。默认 base URL 使用北京 DashScope 兼容入口；生产环境应通过 Settings 改为所属地域的业务空间专属 `/api/v1` 地址。

```powershell
$env:FINANCIAL_AGENT_QWEN_API_KEY = "<your-key>"
# 可选：$env:FINANCIAL_AGENT_QWEN_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
.\.venv\Scripts\python.exe -m financial_agent build-rag-index
.\.venv\Scripts\python.exe -m pytest -q -m live
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```

三个 Tool 均只返回 `Evidence[]`：

| Tool | filters |
| --- | --- |
| `search_research_reports` | `companies`、`brokers`、`as_of`（publish_date 上界） |
| `search_regulatory_knowledge` | `issuer`、`as_of`（publish/effective date 上界） |
| `search_business_knowledge` | `category`、`as_of`（effective_date 上界） |

```powershell
# 重新生成固定 corpus
.\.venv\Scripts\python.exe scripts\generate_knowledge_corpus.py
```

```python
from pathlib import Path

from financial_agent.knowledge import KnowledgeIngestor

chunks = KnowledgeIngestor().ingest(Path("data/knowledge/manifest.json"))
```

## 测试/demo 故障序列

正常 build_user_tools 不装配故障 hook。测试/demo 可显式构造服务（沿用上文 settings）：

```python
from financial_agent.demo_faults import FaultSequence
from financial_agent.user_data.audit import JsonlAuditSink
from financial_agent.user_data.auth import CredentialStore
from financial_agent.user_data.repository import SQLiteUserDataRepository
from financial_agent.user_data.runtime import register_user_tools
from financial_agent.user_data.service import UserDataService

registry = register_user_tools(UserDataService(
    SQLiteUserDataRepository(settings.user_db_path),
    CredentialStore.from_json(settings.user_api_keys),
    JsonlAuditSink(settings.audit_path),
    before_read=FaultSequence(["timeout", "429", "503", "success"]),
))
```

使用上文 registry.invoke 连续调用，依次得到 504、429、503、真实查询结果；耗尽后持续正常查询。序列由该 service 的授权且参数合法的调用消费，跨工具共享。没有 sleep、随机失败、真实 deadline 或自动重试；故障不是 Tool 参数。

默认 pytest 使用小规模 fixture；2,000 用户完整生成、可复现性和全库一致性属于 integration test：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m integration
```

## 本机验证

2026-09-09，Windows / Python 3.13.5：

- Phase 0：15 个测试通过，模块与 CLI 自检成功。
- Phase 1.1：106 个测试通过，包含全部 Phase 0 测试；pip check 无依赖冲突。
- Phase 1.2：116 个默认测试通过；2 个 Tushare live tests 通过；pip check 无依赖冲突。
- Phase 1.4（2026-09-11）：151 个默认测试、2 个 integration tests、5 个 live tests 全部通过；真实 Qwen 3.7 retrieval eval 为 Hit@1 0.9583、Hit@5 1.0000、MRR 0.9792、Recall@5 1.0000。
- Phase 2（2026-09-11）：162 个默认测试和 2 个 integration tests 通过；LangGraph 单任务、并行、依赖、失败阻断、未知 Tool、非法参数、增量 Result Pool 和不可解析依赖均为离线测试。
- 覆盖四个业务 Tool、FastAPI endpoint、Async HTTP Client、空数据/缺失值、401、403、404、422、超时、429、503，以及 Decimal、分页、时间边界、只读/外键/SQL 注入、故障顺序、审计与 CLI。
- Market Data 测试使用 `httpx.MockTransport`，不访问 live provider；live smoke test 使用 `pytest -m live` 单独运行。

实际直接依赖版本：LangGraph 1.2.11、Pydantic 2.13.5、pydantic-settings 2.15.0、pytest 9.1.1。记录是本机验证结果，不是锁文件或用户阶段验收。

## 当前能力

- 可安装的 src-layout 包、配置校验、日志、CLI 和 pytest。
- 2,000 用户 synthetic SQLite、只读 repository adapter、服务层鉴权、四个业务 Tool，以及本地 FastAPI/Async HTTP Client 链路。
- 用户白名单与 scopes、统一类型化结果、错误分类、JSONL 审计、可注入故障序列。
- 72 份多源知识文档、严格 manifest 校验和统一的 source-aware Chunk 输出。
- 中文 BM25、NumPy dense cosine、RRF、provider rerank、metadata/time filter、Adaptive Top-K 和独立 retrieval eval。
- LangGraph execution/control plane、人工 Task DAG、并行 ready-task fan-out、依赖失败阻断和结构化 FinalResult。
- 不侵入现有 Registry 的 9 Tool 组合路由，完整保留 ToolResult 与 retryable 信息。

## 已知限制

- 尚无 Planner LLM、Plan Validator、Retry、Verifier、Rewrite/Replan、Context Manager 或 Agentic RL。
- dependencies 本轮只控制执行顺序，尚未定义将上游 ToolResult 绑定到下游 arguments 的表达式与解析规则。
- Graph 尚未提供持久化 checkpoint、跨进程恢复、任务取消或生产级并发配额。
- 真实 Qwen index 和 live smoke test 依赖宿主通过环境变量提供 API Key；默认测试不会访问外网。
- Knowledge retrieval 尚未实现复杂表格解析、精细版本推理、query rewrite、decomposition、HyDE、GraphRAG 或向量数据库。
- Market Data v0.1 已提供同步 Tushare REST Provider；120 积分下尚无指数、复权、实时行情、分钟线、Level-2、新闻、资金流或指标库。
- 当前为同步本地访问；审计没有多进程并发保证、轮转或防篡改能力。日志不是通用敏感数据脱敏器。
- 未定义计划结构或 token budget，模型 key 仅预留；不支持多币种、会计对账和数据库迁移。
- 依赖只有兼容范围，未锁定全部传递依赖；只在当前 Windows 环境验证。

## 下一步

下一阶段接入 Structured Planner：定义 Planner 的结构化 Task DAG 输出、上游结果到下游参数的绑定规则，并在执行前加入工具名、参数 schema、依赖环和任务规模校验。之后再独立设计 retry、Verifier、replan 和 context 管理，不在当前 execution skeleton 中提前实现。

后续工程约束：外部模型和数据源必须经 adapter/registry，LangGraph node 不直接依赖 provider SDK；API key 仅由环境配置注入；所有用户数据为 synthetic；每个功能补测试，并同步更新 README 的“当前能力 / 已知限制 / 下一步”。
