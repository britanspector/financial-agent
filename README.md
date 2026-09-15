# Financial Agent

用于深度学习和求职展示的本地多工具 Agent 工程探索。全部用户数据为 synthetic，不连接真实公司内部系统。

**当前阶段：Phase 5.1 Context Budget + Selection 实现完成。** 现有 4 个用户数据 Tool、2 个行情 Tool 和 3 个 RAG Tool 的公开契约可供 Planner 选用；Qwen Planner、Answer Writer 和 Verifier 统一通过 Context Manager 获取受策略和预算控制的历史上下文，Phase 4.3 的有界 Rewrite/Replan 闭环保持不变。

## 项目结构与依赖

```text
financial-agent/
├── .env.example / .gitignore / pyproject.toml / README.md
├── src/financial_agent/
│   ├── __init__.py / __main__.py / config.py
│   ├── schemas.py / logging_config.py
│   ├── agent/                  # AgentState、Task、LangGraph execution graph
│   ├── answering/              # 证据约束的 Answer Writer 与 Qwen adapter
│   ├── context/                # Context policy、确定性 selection、token estimation
│   ├── verifier/               # Structured Verifier、Qwen adapter 与 Eval metrics
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
├── eval/planner/                # 与 prompt/source 解耦的固定 Planner Eval JSONL
├── eval/loop/                   # 15 条离线闭环黑盒验收场景
├── eval/context/                # 12 条长历史 Context Selection baseline
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

Phase 4.1 的 Retry 完全位于单次 `execute_tool` 节点内部，不会重新调度 LangGraph 节点。默认每个 Task 在首次 attempt 失败后最多重试 2 次，指数退避为 0.5 秒、1.0 秒；只有 `ToolResult.status=error` 且 `error.retryable=true` 才会重试。Binding 在 Task 进入执行节点前只解析和校验一次，所有 attempt 复用同一份参数；重试不会重新执行上游 Task。

全图默认最多开始 36 次真实 Tool attempt。120 秒 deadline 的准确语义是：deadline 到达后禁止开始新的 Tool attempt，它不是强制终止时间，也不会杀掉已经开始的同步 Tool。因此整个 graph 的实际完成时间可能略超过 120 秒。若 Task 已有真实失败结果但无法开始下一次 retry，最终保留最后一次真实 Tool 错误；只有首次 attempt 尚未开始就被预算阻止时才返回 `EXECUTION_BUDGET_EXCEEDED`。退避等待计入 deadline。

`TaskExecutionResult.retry_count` 表示首次 attempt 之后实际发生的重试次数，`max_retry` 表示本次执行策略上限。`FinalResult` 额外返回 `attempt_count`、`execution_duration_ms`、`attempt_budget_exhausted` 和 `deadline_exceeded`。策略可由 `FINANCIAL_AGENT_EXECUTION_MAX_RETRY`、`FINANCIAL_AGENT_EXECUTION_INITIAL_BACKOFF_SECONDS`、`FINANCIAL_AGENT_EXECUTION_BACKOFF_MULTIPLIER`、`FINANCIAL_AGENT_EXECUTION_MAX_ATTEMPTS`、`FINANCIAL_AGENT_EXECUTION_DEADLINE_SECONDS` 配置，并显式传入执行入口：

```python
from financial_agent.agent import RetryPolicy

retry_policy = RetryPolicy.from_settings(settings)
state = run_execution_graph(request, tasks, registry, retry_policy=retry_policy)
```

## Structured Verifier

Phase 4.2 新增 `financial_agent.verifier.StructuredVerifier`，输入原始 `UserQuery`、已验证并执行的非空 Task Plan、完整 `TaskExecutionResult` 列表，以及调用方提供的 `DraftAnswer`。本阶段不负责生成草稿，也不自动执行 Rewrite/Replan；`clarify` 和 `no_tool` 继续使用 Planner 的既有短路路径。

Verifier 输出固定为 `VerificationResult(decision, reason, missing_evidence, failed_task_ids)`。`decision` 只能是 `PASS`、`REWRITE` 或 `REPLAN`：PASS 表示草稿和证据足以回答问题；REWRITE 表示现有 Tool Results 已有全部必要证据，只需修正遗漏、矛盾、引用或表达；REPLAN 仅表示现有 Tool Results 本身缺少必要证据，必须新增、更换或重新执行 Tool。文案质量差本身不能触发 REPLAN。PASS/REWRITE 的 `missing_evidence` 必须为空，REPLAN 必须明确缺失证据。

`failed_task_ids` 不由模型生成，而是代码根据 error Tool Results 按 Plan 顺序确定性写入。草稿证据使用 `EvidenceReference(task_id, source_path)`，与 Result Binding 共用静态公开字段检查和运行期路径解析；路径固定从 `ToolResult.data` 开始，只允许字段名和非负 list index。Plan/Result ID 不完整、重复、Tool 名不一致，或证据引用失败/私有/不存在的结果时，会在调用模型前抛出 `InvalidVerificationInputError`。

```python
from financial_agent.verifier import DraftAnswer, EvidenceReference, build_verifier

verifier = build_verifier(settings, registry)
verification = verifier.verify(
    request,
    tasks,
    final_result.task_results,
    DraftAnswer(
        answer="当前草稿答案",
        evidence=[EvidenceReference(task_id="t1", source_path=["stocks", 0, "stock_code"])],
    ),
)
```

Verifier 默认复用 `FINANCIAL_AGENT_QWEN_API_KEY`，model、base URL、timeout 和 temperature 分别由 `FINANCIAL_AGENT_VERIFIER_MODEL`、`FINANCIAL_AGENT_VERIFIER_BASE_URL`、`FINANCIAL_AGENT_VERIFIER_TIMEOUT_SECONDS`、`FINANCIAL_AGENT_VERIFIER_TEMPERATURE` 配置。固定 9 条 Eval 位于 `eval/verifier/verifier_cases.jsonl`，运行 `python scripts/evaluate_verifier.py` 可观察 PASS precision、REWRITE/REPLAN 混淆率、不必要 REPLAN 率和漏判 REPLAN 率；该命令访问真实模型，不属于默认 pytest。

2026-09-13 使用真实 `qwen3.7-flash` 运行全部 9 条固定 Verifier Eval：9/9 判定正确，PASS precision 1.0000，REWRITE/REPLAN 混淆率 0.0000，不必要 REPLAN 率 0.0000，漏判 REPLAN 率 0.0000。首轮发现模型曾输出语义为“需要新 Tool”的 reason 却选择 REWRITE；将 response schema 改为按 decision 分支的 strict `oneOf`，并加入逐项核对用户所需证据的一致性检查后复测通过。

## Bounded Replan Loop

Phase 4.3 新增 `financial_agent.loop.run_agent_loop`，将 Planner → Execute → Answer Writer → Verifier 连成有界闭环。PASS 返回最终答案；REWRITE 只把当前 Plan、Tool Results、草稿和 Verifier feedback 交给 Answer Writer，不调 Tool；REPLAN 把已有结果和 feedback 交给 Planner，要求返回完整替换计划，再执行并重新生成答案。Phase 4.3 仍不把循环实现成 LangGraph 节点重调度；单 Task Retry 继续留在一次 `execute_tool` 内部。

默认硬限制是 `max_rewrite=2`、`max_replan=2`、`max_iterations=5`、`total_tool_budget=36`。达到限制时返回 `AgentLoopResult(status="limit_exhausted")`，保留当前草稿、判定、Plan 和 Results，并通过 `stop_reason` 区分具体限制。`iteration_count` 统计 Verifier 调用；每次真实 Tool attempt（包括 retry）都从同一个线程安全的 loop 级预算控制器扣减。预算耗尽只阻止需要 Tool 的 REPLAN，不阻止仍可只用现有证据完成的 REWRITE。每次 execution round 仍使用 Phase 4.1 的 soft start deadline；它不杀死已经开始的同步 Tool。

Replan 输出是完整替换计划和 `force_rerun_task_ids`。Task ID 与标准化后的完整 Task 定义均未变化、且依赖链也全部可复用时，既有 success 默认复用，empty 也默认允许复用；error 永不复用。Planner 可将 success/empty Task ID 显式放入 `force_rerun_task_ids` 取得新结果。被移出替换计划的旧 Result 会立即丢弃，因此每轮当前 Plan 与 Result ID 始终严格一致。只有 Plan 未变化、Results 未变化并且本轮没有开始任何新 Tool attempt 时才判定 `no_progress`。

Answer Writer 默认使用 `qwen3.7-flash` 和 strict structured output，返回 `DraftAnswer(answer, evidence)`。Evidence 路径与 Result Binding、Verifier 共用同一套安全路径语义：始终从 `ToolResult.data` 开始，只允许公开字段和非负 list index。Writer prompt 不包含 request ID、凭证或 latency。

```python
from financial_agent.agent.retry import RetryPolicy
from financial_agent.loop import LoopPolicy, run_agent_loop

result = run_agent_loop(
    request,
    planner,
    validator,
    registry,
    answer_writer,
    verifier,
    policy=LoopPolicy(max_rewrite=2, max_replan=2, max_iterations=5, total_tool_budget=36),
    retry_policy=RetryPolicy(max_retry=2),
    context=call_context,
)
print(result.answer, result.stop_reason, result.total_tool_attempts)
```

Answer Writer 由 `FINANCIAL_AGENT_ANSWER_MODEL`、`FINANCIAL_AGENT_ANSWER_BASE_URL`、`FINANCIAL_AGENT_ANSWER_TIMEOUT_SECONDS`、`FINANCIAL_AGENT_ANSWER_TEMPERATURE` 配置；循环限制由 `FINANCIAL_AGENT_LOOP_MAX_REWRITE`、`FINANCIAL_AGENT_LOOP_MAX_REPLAN`、`FINANCIAL_AGENT_LOOP_MAX_ITERATIONS`、`FINANCIAL_AGENT_LOOP_TOTAL_TOOL_BUDGET` 配置。显式构造 `LoopPolicy` 时由调用方传入 `run_agent_loop`。

固定 Loop Eval 位于 `eval/loop/loop_cases.jsonl`，完全离线并通过公开 `run_agent_loop` 入口执行。15 条场景覆盖初次 PASS、零 Tool REWRITE、增加/修改 Task 的 REPLAN、empty force rerun、error 恢复、部分复用、Binding 上游变化、五类停止条件、第五次 Verify 才 PASS，以及首次 retry 与后续 replan 共用预算。运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_loop.py
```

2026-09-15 本机闭环验收结果为 15/15 通过；REWRITE audit 确认 Tool 调用数不变且传入 Writer 的 Result snapshot 未变化。

## Context Manager

Phase 5.1 新增 `financial_agent.context.ContextManager`。Planner、Answer Writer 和 Verifier 在构造原有 prompt 前均通过同一接口选择 `UserQuery.history`；`query` 和 `request_id` 原样保留，原请求不会被修改。默认 `full_history` 与 Phase 4 行为兼容，`last_n` 按“一个 user 消息及其后连续 assistant 消息”为一轮保留最近 N 轮，`budgeted_selection` 在各组件独立的 history token budget 内确定性选择完整轮次。历史开头的连续 assistant 消息组成独立首轮，`last_n=0` 可显式丢弃全部历史。

`budgeted_selection` 优先尝试最新一轮；如果该轮本身超预算，会跳过并继续选择其他能装入预算的候选。其余候选综合 recency、与当前 query 的 lexical overlap、明确的 user ID / A 股证券代码 / 日期实体重叠，以及 user role 中的约束、纠正和确认语义排序。中文相关性使用固定 bigram，不把简单正则当作通用公司/机构 NER；公司和机构名称主要依赖 lexical overlap。assistant 的普通“收到/已确认”不会获得约束确认加权。选择以完整轮次为单位，不截断消息；只有 current query 无条件完整保留。

默认 `HeuristicTokenEstimator` 对紧凑 message JSON 的 UTF-8 字节数做稳定估算，它不是具体模型 tokenizer。调用方可通过 `TokenEstimator.estimate_messages()` 协议注入其他估算实现。budget 和 `original_tokens` / `selected_tokens` 只统计 history，不包含 query、system prompt、tools、plan、Tool Results、draft 或 verifier feedback。

```python
from financial_agent.context import ContextManager, ContextPolicy

selection = ContextManager().select(
    request,
    "planner",
    ContextPolicy(strategy="budgeted_selection", budget_tokens=4096, last_n=6),
)
print(selection.request.history, selection.metrics)
```

`ContextSelection.metrics` 包含 component、strategy、budget、original/selected tokens、`selected_tokens / original_tokens` 压缩率和 selected/dropped message count。直接调用 `select()` 可读取结构化 metrics；Planner、Writer、Verifier 的正常路径只记录不含 query、history 或实体值的安全计数日志，不修改 `AgentLoopResult` / `LoopTrace`。

配置项如下；strategy 和 `last_n` 默认共享，三组件 budget 独立：

```text
FINANCIAL_AGENT_CONTEXT_STRATEGY=full_history
FINANCIAL_AGENT_CONTEXT_LAST_N=6
FINANCIAL_AGENT_PLANNER_CONTEXT_BUDGET_TOKENS=4096
FINANCIAL_AGENT_ANSWER_CONTEXT_BUDGET_TOKENS=4096
FINANCIAL_AGENT_VERIFIER_CONTEXT_BUDGET_TOKENS=4096
```

固定 Context Selection Eval 位于 `eval/context/context_cases.jsonl`，包含 12 条至少 8 条 message 的长历史场景，覆盖早期 user ID、否定 Tool 约束、公司指代、日期/证券代码纠正、闲聊中的约束、超长无关最近轮次、显式执行顺序和 user/assistant 确认边界。运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_context.py
```

默认比较 `full_history`、最近 3 轮的 `last_n` 和 130-token `budgeted_selection`，记录关键上下文保留率、history token 压缩率、Planner plan-sufficiency 等价率、关键实体或用户约束丢失率。Planner 指标使用 `deterministic_required_context` 离线 probe：它判断所选上下文是否仍足以形成与 full-history 等价的计划输入，不调用或冒充真实 Qwen Planner。2026-09-16 的 Phase 5.1 baseline 为：

| Strategy | 关键上下文保留率 | history token 压缩率 | Planner 等价率 | 关键实体/约束丢失率 |
| --- | ---: | ---: | ---: | ---: |
| full_history | 100.00% | 100.00% | 100.00% | 0.00% |
| last_n | 43.48% | 61.70% | 33.33% | 55.56% |
| budgeted_selection | 100.00% | 82.27% | 100.00% | 0.00% |

这里的 token 压缩率沿用 `selected_tokens / original_tokens`，数值越低表示裁剪越多。Eval 只建立离线 baseline，不宣称 budgeted selection 必然优于 full history；后续可用同一组 case 对比 Summary、Summary + Retrieval，并另行运行真实 Planner 消融。

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

## Structured Planner + Plan Validator

Phase 3.1 在执行图之外新增 `financial_agent.planner`。`StructuredPlanner` 输入 `UserQuery.query`、经 Context Manager 选择的 `history`、9 个 Tool 的公开 description/input JSON Schema、当前日期和一小组 planning rules。输出固定为：

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_market_history",
      "arguments": {
        "symbol": "600519.SH",
        "start_date": "2026-01-01",
        "end_date": "2026-02-01"
      },
      "dependencies": []
    }
  ]
}
```

`decision` 为 `execute`、`clarify` 或 `no_tool`：execute 必须有 Task；clarify/no_tool 必须没有 Task。缺少 user_id、证券代码或其他业务必填参数时使用 clarify，通用问候/写作或系统无能力支持的请求使用 no_tool。Prompt 分为两条 message：system 包含最小充分计划、不得编造参数、并行/依赖、时间字段、能力边界和禁止结果引用等规则；user 是包含 query、所选 history、current_date 和 tools 的 JSON。Provider 发送 Tool-aware strict JSON Schema：每个 `tool_name` 分支的 `arguments` 直接使用该 Tool 的公开 input schema，杜绝开放对象生成非契约参数键。Tool 参数 schema 明示日期区间为 `[start_date, end_date)`，以及各 RAG `as_of` 的含上界语义。法规检索只用于法规/监管/适当性，业务知识只用于 FAQ/流程/产品说明；研报不是实时新闻；行情 Tool 仅支持已有的 A 股日终/历史行情，不支持指数实时、新闻、汇率或预测。Planner 仅依赖 `PlannerProvider` 协议。默认 adapter 通过 DashScope OpenAI-compatible `/chat/completions` 调用 `qwen3.7-flash-2026-07-15`，temperature 默认为 0.1，关闭 thinking，并优先使用 strict JSON Schema response format。API key 复用 `FINANCIAL_AGENT_QWEN_API_KEY`；model、base URL、timeout、temperature 和 task 上限分别由 `FINANCIAL_AGENT_PLANNER_*` 配置。

`PlanValidator` 不调用模型或 Tool，只保证计划合法可执行。Phase 3.2 的 `bindings` 使用结构化 `{target_parameter, source_task_id, source_path}`；`source_path` 固定从上游 `ToolResult.data` 开始，只允许该 Tool `output_model` 公开 Pydantic 契约中的字段名和非负 list index，不支持 JSONPath 或表达式。因此 Planner 不能读取 `ToolResult.status/error/request_id`，也不能遍历 service/provider 的内部字段。Validator 检查 source task、禁止自身/重复 target、字段白名单、静态类型兼容和包含 binding edge 的 cycle，并把 binding source 自动置为依赖（优先于手写 dependency）。校验成功时转换为现有 Phase 2 `Task`；clarify/no_tool 成功时返回空 Task。旧的 `$t1.result...` 字符串引用会被拒绝。

执行前，Phase 2 dispatcher 在所有上游 dependency 成功后才解析 binding，将真实值写入 arguments，并重新以 Tool input schema 校验；解析失败会产生控制面错误，上游 Tool 失败则下游不会执行。统一入口为 `financial_agent.agent.orchestrator.run_planner_execution(request, planner, validator, registry)`，完整串通 `query/history → Context Manager → Planner → validate/binding compile → graph execution → FinalResult`。第一版只支持标量或完整 list 字段、任意长度的显式 DAG 链；Phase 4.1 已加入 Tool Execution Retry，Phase 4.2 提供独立语义 Verifier，Phase 4.3 加入自动 Rewrite/Replan，Phase 5.1 加入确定性 history budget 和 selection。

固定 Planner Eval Set 位于 `eval/planner/planner_cases.jsonl`，共 40 条独立 JSONL case，不从 prompt、模型输出或源码反推期望。每条都包含 query、完整 history、expected_tools、expected_arguments、expected_dependencies、temporal_expectation、forbidden_tools；可用 `acceptable_plans` 声明多套等价 Tool/参数/依赖 pattern。case 不保存或匹配模型生成的 task ID，依赖只按上游/下游 Tool edge 比较。4 条信息不足或歧义 case 标记为 `abstain`，要求不调用 Tool、不编造参数；它们不进入 executable valid plan rate 的分母。

runner 对每条实际 StructuredPlan 先调用 PlanValidator，再在 canonical 与全部 acceptable patterns 中选择语义得分最高者。指标为 valid plan rate（36 个需要可执行 plan 的 case）、tool selection accuracy（case 级 Tool multiset 精确匹配）、argument accuracy（Tool 实例参数精确匹配）、temporal accuracy（`as_of/start_date/end_date` 子集）、dependency accuracy（Tool 依赖 edge F1）和 unnecessary tool call rate（多余调用数 / 实际调用数）。默认 pytest 使用 Fake Provider；真实 Qwen 评估会依次运行 40 条 case：

```powershell
$env:FINANCIAL_AGENT_QWEN_API_KEY = "<your-key>"
.\.venv\Scripts\python.exe scripts\evaluate_planner.py
.\.venv\Scripts\python.exe -m pytest -q -m live tests\planner\test_live.py
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
| `search_regulatory_knowledge` | `issuer`、`as_of`（省略时仅当前 active；指定时按 publish/effective date 上界） |
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
- Phase 3.1 final optimization（2026-09-11）：Planner policy 明确了提供合法标识时不得 clarify、synthetic 实体原样保留、Tool 域优先级、显式顺序依赖和统一 temporal policy；末尾 checklist 进一步强调多实体合并检索、公告/规则路由及 date/dependency 必须落实。Eval 对语义等价 query 与 query 中保留的 entity filter 评分为 acceptable，同时继续严格比较标识、类别与日期。固定 40 条真实 Eval 的 valid plan rate 1.0000、tool selection accuracy 0.8250、argument accuracy 0.7800、temporal accuracy 0.7500、dependency accuracy 0.9000、unnecessary tool call rate 0.0638；完整逐 case 报告见 `reports/phase3_planner_policy_checklist_40.md`。
- Phase 4.1（2026-09-12）：222 个默认测试通过；Execution Retry 覆盖 retryable/non-retryable 分类、指数退避、全图 attempt 预算、soft deadline、并行预算隔离和 Binding 参数稳定性。
- Phase 4.2（2026-09-13）：265 个默认测试通过；Structured Verifier 覆盖结构化判定、确定性失败清单、共享 Result Path、输入一致性、Provider 错误和四项边界 Eval 指标；真实 Qwen Eval 独立使用 `-m live` 或脚本运行。
- Phase 4.3（2026-09-15）：293 个默认测试通过，15/15 固定 Loop Eval 通过；闭环覆盖 PASS/REWRITE/REPLAN 路由、完整替换计划、success/empty 复用、force rerun、error 不复用、旧结果裁剪、依赖安全失效、跨轮共享 Tool attempt 预算和 no-progress 三条件判定。
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
- Provider 解耦的 Structured Planner、严格 Task DAG schema、Qwen 3.7 Flash adapter 和执行前确定性 PlanValidator。
- 单个 Task 内的有界 Tool Retry、指数退避、全图 attempt 预算和禁止新 attempt 的 soft deadline。
- 结构化 Result Binding、公开输出字段白名单、运行期结果路径解析和 plan → validate → execute 统一入口。
- Provider 解耦的 Structured Verifier、严格 PASS/REWRITE/REPLAN schema、确定性 failed_task_ids 和固定 Verifier Eval Set。
- Verifier 驱动的有界 Agent Loop、证据约束 Answer Writer、完整 Replan、依赖安全结果复用、显式 force rerun 和跨轮 Tool attempt 总预算。
- Planner、Writer、Verifier 共用的 Context Manager、三种 history 策略、独立 token budget、安全选择指标和可注入 token estimator。
- Prompt-independent 的 40 条 Planner Eval Set：单/并行/依赖、三源组合、当前/历史时间、RAG filters、法规、FAQ、无关 Tool 与 abstain 边界，以及六项离线/真实模型通用指标。

## 已知限制

- 闭环已执行 Rewrite/Replan 并支持确定性 history selection，但没有跨进程 checkpoint、人工审批点或 Agentic RL。
- Result Binding 只支持公开结果中的字段名和非负 list index，不支持 JSONPath、表达式或复杂变换。
- clarify/no_tool 只表达 Planner 决策；本阶段仍未生成面向用户的澄清或 no-tool 文案。
- Eval Set 对 Tool/参数/依赖边做固定语义匹配，但尚未覆盖同义 query rewrite、重复同名 Tool 实例的边身份、统计置信区间、成本和延迟。
- Graph 尚未提供持久化 checkpoint、跨进程恢复、任务取消或生产级并发配额。
- 真实 Qwen index 和 live smoke test 依赖宿主通过环境变量提供 API Key；默认测试不会访问外网。
- Knowledge retrieval 尚未实现复杂表格解析、精细版本推理、query rewrite、decomposition、HyDE、GraphRAG 或向量数据库。
- Market Data v0.1 已提供同步 Tushare REST Provider；120 积分下尚无指数、复权、实时行情、分钟线、Level-2、新闻、资金流或指标库。
- 当前为同步本地访问；审计没有多进程并发保证、轮转或防篡改能力。日志不是通用敏感数据脱敏器。
- Context budget 只覆盖 history，尚不覆盖 query、tools、plan、Tool Results、draft 或 feedback；不支持 Summary、history retrieval/embedding、长期 Memory、模型精确 tokenizer 或 KV Cache 优化。
- 依赖只有兼容范围，未锁定全部传递依赖；只在当前 Windows 环境验证。

## 下一步

下一阶段可增加 Summary Compression、完整 prompt/token/cost 预算、循环 trace 持久化、真实 Provider 闭环 Eval，并补充面向用户的 clarify/no_tool 文案。

后续工程约束：外部模型和数据源必须经 adapter/registry，LangGraph node 不直接依赖 provider SDK；API key 仅由环境配置注入；所有用户数据为 synthetic；每个功能补测试，并同步更新 README 的“当前能力 / 已知限制 / 下一步”。
