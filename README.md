# Financial Agent

用于深度学习和求职展示的本地多工具 Agent 工程探索。全部用户数据为 synthetic，不连接真实公司内部系统。

**当前阶段：Phase 1 用户数据子步骤完成，等待验收。** Phase 0 已完成；Phase 1 的 MarketDataProvider 和知识源样本尚未交付。每个阶段/约定子步骤验收后再继续。

## 项目结构与依赖

```text
financial-agent/
├── .env.example / .gitignore / pyproject.toml / README.md
├── src/financial_agent/
│   ├── __init__.py / __main__.py / config.py
│   ├── schemas.py / logging_config.py
│   ├── demo_faults.py           # 仅测试/demo 使用的故障序列
│   ├── tools/                  # ToolResult、ToolSpec、ToolRegistry
│   └── user_data/              # models、fixtures、repository、auth、audit、service、runtime
├── tests/
│   ├── conftest.py / test_config.py / test_schemas.py
│   ├── test_logging.py / test_bootstrap.py
│   └── user_data/              # 数据、权限、Tool、故障、审计、CLI
└── data/                       # 生成的 SQLite 和审计，不提交 Git
```

Python >=3.11；本机验证使用 Python 3.13.5。依赖范围定义在 pyproject.toml。

| 依赖 | 用途 |
| --- | --- |
| langgraph >=1.0,<2 | 后续图编排的基础依赖；当前仅离线冒烟测试 |
| pydantic >=2.10,<3 | schema、输入与结果校验 |
| pydantic-settings >=2.7,<3 | 环境变量和可选 .env 加载 |
| pytest >=8,<10（dev） | 自动化测试 |
| setuptools >=77,<83（build） | 包构建及 editable 安装 |

**Phase 1.1 无新增依赖。** SQLite、Decimal、鉴权比较、JSONL 和计时使用标准库。没有额外安装 provider SDK、ORM 或 RAG 依赖。

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
.\.venv\Scripts\python.exe -m financial_agent seed-user-data
.\.venv\Scripts\python.exe -m financial_agent list-tools
```

默认生成 data/user_data.db。已有数据库时拒绝覆盖；需要重置时显式使用 `seed-user-data --overwrite`，或用 `--path` 指定路径。覆盖前停止使用该数据库的进程；生成器在同目录构建临时数据库再原子发布，首次发布需要文件系统支持 hard link。只读查询不会自动建库。

在当前 PowerShell 进程环境中生成测试 key 和服务端权限映射：

```powershell
$env:FINANCIAL_AGENT_CALLER_API_KEY = 'synthetic-' + [guid]::NewGuid().ToString('N')
$env:FINANCIAL_AGENT_USER_API_KEYS = ConvertTo-Json -Compress -Depth 4 -InputObject @(
    @{
        api_key = $env:FINANCIAL_AGENT_CALLER_API_KEY
        principal_id = 'synthetic-demo'
        user_ids = @('syn-user-001', 'syn-user-002', 'syn-user-003', 'syn-user-004', 'syn-user-005', 'syn-user-006', 'syn-user-999')
        scopes = @('read:profile', 'read:portfolio')
    }
)

.\.venv\Scripts\python.exe -m financial_agent call-tool get_user_profile --user-id syn-user-001
.\.venv\Scripts\python.exe -m financial_agent call-tool get_user_portfolio --user-id syn-user-001
.\.venv\Scripts\python.exe -m financial_agent call-tool get_user_transactions --user-id syn-user-001 --limit 2 --offset 0
.\.venv\Scripts\python.exe -m financial_agent call-tool get_user_transactions --user-id syn-user-001 --start-time '2026-01-02T09:00:00Z' --end-time '2026-01-03T09:00:00Z'
```

FINANCIAL_AGENT_USER_API_KEYS 是宿主维护的 JSON 数组，条目字段为 api_key、principal_id、user_ids、scopes。默认 [] 拒绝全部访问；重复 key、非法配置和未知 scope 被拒绝。FINANCIAL_AGENT_CALLER_API_KEY 仅供 CLI 构造调用上下文，对应未来 HTTP 的 X-API-Key；当前没有 HTTP 服务。scopes 和白名单来自宿主配置，不能由 Agent 参数自报。

路径变量 FINANCIAL_AGENT_USER_DB_PATH、FINANCIAL_AGENT_AUDIT_PATH 默认分别为 data/user_data.db、data/audit.jsonl，相对当前工作目录解析。FINANCIAL_AGENT_MODEL_API_KEY 仍为后续模型 adapter 预留。

Python 直接调用（凭证仍来自环境配置）：

```python
from financial_agent.config import Settings
from financial_agent.user_data.auth import CallContext
from financial_agent.user_data.runtime import build_user_tools

settings = Settings()
registry = build_user_tools(settings)
result = registry.invoke(
    "get_user_portfolio",
    {"user_id": "syn-user-001"},
    context=CallContext(api_key=settings.caller_api_key),
)
print(result.model_dump_json())
```

## 数据样本与访问边界

| 用户 | 场景 |
| --- | --- |
| syn-user-001 | 多账户、多持仓、四笔交易，含相同成交时间与精确小数 |
| syn-user-002 | 存在用户，无账户、持仓、交易 |
| syn-user-003 | 有账户，无持仓、交易 |
| syn-user-004 | 有持仓，无交易历史，模拟期初持仓 |
| syn-user-005 | age_band / risk_level / region 为 null |
| syn-user-006 | 已关闭账户，保留历史交易 |
| syn-user-999 | 不存在；列入示例白名单，用于演示 404 |

身份、区域和证券标识都是 synthetic。users / accounts / holdings / transactions 四张表设置主键、外键和查询索引。金额、价格、数量存为十进制文本，读取为 Decimal，JSON 输出字符串；组合与交易声明 currency: CNY。这是固定快照，不是完整会计账本。

调用路径为 ToolRegistry → UserDataService → UserDataRepository → SQLiteUserDataRepository。repository 使用只读连接和参数绑定，多次读取在同一读事务中完成。repository 是受信任的应用内部接口，不能绕过 service 暴露为 Agent Tool；SQLite 文件不承担操作系统级访问隔离。

## Tool 契约与错误分类

| Tool | 输入 | scope |
| --- | --- | --- |
| get_user_profile | user_id | read:profile |
| get_user_portfolio | user_id | read:portfolio |
| get_user_transactions | user_id、可选 start_time / end_time / limit / offset | read:portfolio |

拒绝未知输入字段与空 user_id。交易时间必须带时区，统一转换为 UTC；范围为 [start_time, end_time)，按 trade_time ASC、transaction_id ASC 排序。limit 默认 20、范围 1–100，offset 默认 0；页数据与 total 分开返回，翻过最后一页仍保留 total。

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

## 本机验证

2026-09-09，Windows / Python 3.13.5：

- Phase 0：15 个测试通过，模块与 CLI 自检成功。
- Phase 1.1：106 个测试通过，包含全部 Phase 0 测试；pip check 无依赖冲突。
- 覆盖三个 Tool 的成功、空数据/缺失值、401、403、404、422、超时、429、503，以及 Decimal、分页、时间边界、只读/外键/SQL 注入、故障顺序、审计与 CLI。
- 测试使用临时 SQLite、临时审计、synthetic 凭证，不使用 live provider。

实际直接依赖版本：LangGraph 1.2.11、Pydantic 2.13.5、pydantic-settings 2.15.0、pytest 9.1.1。记录是本机验证结果，不是锁文件或用户阶段验收。

## 当前能力

- 可安装的 src-layout 包、配置校验、日志、CLI 和 pytest。
- 六用户固定 SQLite fixtures、只读 repository adapter、服务层鉴权及三个独立 Tool。
- 用户白名单与 scopes、统一类型化结果、错误分类、JSONL 审计、可注入故障序列。

## 已知限制

- 尚无应用 LangGraph 工作流、历史压缩、Planner、RAG、RL 或模型质量评估；测试中的单节点图仅验证依赖可用。
- 尚无 MarketDataProvider、知识源样本、live adapter、FastAPI、真实 deadline 或 bounded retry 执行器。
- 当前为同步本地访问；审计没有多进程并发保证、轮转或防篡改能力。日志不是通用敏感数据脱敏器。
- 未定义计划结构或 token budget，模型 key 仅预留；不支持多币种、会计对账和数据库迁移。
- 依赖只有兼容范围，未锁定全部传递依赖；只在当前 Windows 环境验证。

## 下一步

等待用户验收 Phase 1 用户数据子步骤。验收后再继续 Phase 1 的 MarketDataProvider / LocalSnapshotProvider 与三类知识源最小样本；整个 Phase 1 尚未完成。FastAPI、真实 delay/deadline 和 live demo adapter 按后续约定推进。

后续工程约束：外部模型和数据源必须经 adapter/registry，LangGraph node 不直接依赖 provider SDK；API key 仅由环境配置注入；所有用户数据为 synthetic；每个功能补测试，并同步更新 README 的“当前能力 / 已知限制 / 下一步”。
