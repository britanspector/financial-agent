# Phase 1 独立黑盒测试报告

- 测试日期：2026-09-11（Asia/Hong_Kong）
- 测试分支：`phase1`（测试开始时相对 `origin/phase1` ahead 3）
- Qwen live 补测：2026-09-11，使用本地 `phase1` 引用 `417356265808c9331b333c7673411bf2a2680d33`
- 测试方式：仅使用 README 已公开用法、CLI、HTTP/OpenAPI、Tool Registry 的公开 `describe/invoke` 接口，以及公开 Provider/Service 注入点
- 黑盒边界：未读取 `src/`，未读取现有 `tests/`，未运行现有单测，未修改产品代码、仓库配置或产品数据
- 数据：User Data 使用仓库既有 synthetic 数据库，只读访问；本次审计和受控 RAG 索引均定向到系统临时目录并在测试后删除

## 结论

Phase 1 的 User Data、Market Data 主要成功路径和本地 RAG 检索链整体可用。四个 User Data Tool、HTTP API 鉴权、参数校验、Market snapshot/history、日期右开边界、相似 FAQ 区分、RAG 历史过滤和多实体召回均通过。错误请求不会污染后续正常请求，ToolResult 的基本结构稳定。

本轮记录 4 个问题：

| ID | 问题 | 严重程度 | 结论 |
| --- | --- | --- | --- |
| BB-01 | “当前法规”查询中，已废止版本可能排在现行版本之前 | 中 | FAIL |
| BB-02 | HTTP 连接被拒绝与真实读超时均被归类为 `TIMEOUT/504` | 低 | FAIL |
| BB-03 | HTTP API 静默忽略未知 query 参数 | 低 | FAIL |
| BB-04 | Uvicorn access log 原样记录 query 参数值，可能泄露误放在 URL 中的敏感值 | 中 | FAIL |

RAG 首轮因缺少 `FINANCIAL_AGENT_QWEN_API_KEY`，先通过公开注入点使用隔离、确定性的 embedding/reranker 验证过滤、召回范围、Evidence 契约和错误映射。Key 补充后已完成标准 `build_rag_tools(Settings())` 的真实 Qwen live 补测：三个 Tool 均正常返回，首轮结果中的 BB-01 也被真实 Qwen 排序复现。

## 1. User Data

### 1.1 四个业务 Tool 与 ToolResult

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 客户上下文正常查询 | `get_customer_context`, `user_id=syn-user-0001` | exit 0；`status=success`；`source=synthetic_user_db`；返回用户、风险、区域、资产等字段；`error=null`；UUID request_id | PASS |
| 两融账户正常查询 | `get_margin_account`, `user_id=syn-user-0001`, `2026-05-01 <= date < 2026-06-30`, `limit=2` | exit 0；`status=success`；`total=59`；返回 2 条 daily；分页字段完整 | PASS |
| 持仓正常查询 | `get_portfolio_positions`, `user_id=syn-user-0001` | exit 0；`status=success`；返回 `industries/stocks`、币种及快照日期 | PASS |
| 组合分析正常查询 | `get_portfolio_analytics`, `user_id=syn-user-0001` | exit 0；`status=success`；返回 `report/factors/rank` | PASS |
| 翻过最后一页 | `get_margin_account`, `limit=2`, `offset=9999` | exit 0；`status=empty`；`data` 保留合法账户结构和 `total=60`；`error=null` | PASS |
| 未知 Tool | `does_not_exist` | exit 1；`UNKNOWN_TOOL/404/retryable=false`；`data=null` | PASS |

所有成功和失败 ToolResult 均包含 `status/data/source/latency/error/request_id`；request_id 每次不同，错误时 `data=null`，正常及 empty 时 `error=null`。

### 1.2 用户、API Key、scope 与 whitelist

测试服务使用仅存在于测试进程环境中的 4 组凭证：完整权限、仅 customer scope、仅允许另一用户、允许一个不存在用户。凭证没有写入仓库或报告。

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 缺失 API Key | 正常用户，无 `X-API-Key` | CLI exit 1；HTTP 401；`UNAUTHORIZED`，不可重试 | PASS |
| 错误 API Key | 错误 key + 正常用户 | CLI exit 1；HTTP 401；`UNAUTHORIZED`，消息不回显 key | PASS |
| scope 不足 | 仅 customer scope 调用 margin | CLI exit 1；HTTP 403；`FORBIDDEN: Required scope is not granted` | PASS |
| whitelist 不足 | key 只允许 `syn-user-0002`，查询 `syn-user-0001` | CLI exit 1；HTTP 403；`FORBIDDEN: User access is not granted` | PASS |
| whitelist 内用户不存在 | key 允许 `syn-user-9999`，查询该用户 | CLI exit 1；HTTP 404；`NOT_FOUND: Synthetic user not found` | PASS |
| 空 user_id | `user_id=""` | exit 1；`INVALID_ARGUMENT/422` | PASS |
| 超长 user_id | 129 个字符 | exit 1；`INVALID_ARGUMENT/422` | PASS |

HTTP 成功响应为领域模型，错误响应为 `ApiError`，并且响应 header `X-Request-ID` 与 body request_id 一致。

### 1.3 非法参数、日期与分页边界

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 非法日期 | `start_date=not-a-date` | `INVALID_ARGUMENT/422` | PASS |
| 逆序日期 | `2026-06-30` 至 `2026-05-01` | `INVALID_ARGUMENT/422` | PASS |
| 相等日期 | `start_date=end_date=2026-06-01` | HTTP 422 | PASS |
| limit 下界 | `limit=0` | `INVALID_ARGUMENT/422` | PASS |
| 单日右开边界 | `[2026-06-01, 2026-06-02)` | 仅返回 `2026-06-01` 一条 | PASS |
| end_date 排除 | `end_date=2026-06-01` | 返回 5 月 2 日至 5 月 31 日，共 30 条，不含 6 月 1 日 | PASS |
| start_date 包含 | `start_date=2026-06-01` | 返回 6 月 1 日起 30 条 | PASS |
| 全范围之前/之后 | 1900 或 2099 单日窗口 | HTTP 200，合法空集合 | PASS |

### 1.4 HTTP 异常与连接失败

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 数据库不可用 | 服务进程使用明确不存在的临时 DB 路径 | ToolResult `DATA_UNAVAILABLE/503/retryable=true`；无路径或异常原文泄露 | PASS |
| 服务连接被拒绝 | base URL 指向未监听的 `127.0.0.1:18081` | `TIMEOUT/504/retryable=true` | FAIL（BB-02） |
| 真实读超时 | 本地 socket 接受连接后不响应，timeout 50ms | `TIMEOUT/504/retryable=true` | PASS |

### 1.5 HTTP 参数严格性与日志

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 未知 query 参数 | `/context?unexpected=secret-looking-value` | HTTP 200，参数被静默忽略 | FAIL（BB-03） |
| 访问日志敏感值检查 | 同上 | access log 完整打印 `unexpected=secret-looking-value`；API Key header 未被打印 | FAIL（BB-04） |
| 业务错误消息泄露检查 | 错 key、缺 DB、非法参数、连接失败 | 未发现 API Key、SQL、DB 路径、堆栈或第三方响应正文出现在 ToolResult | PASS |

## 2. Market Data

Market Data 通过 README 公开的 `build_market_tools(Settings())` 和 Tool Registry `invoke` 调用；测试时宿主配置可正常访问 Tushare。未读取凭证值。

### 2.1 snapshot、history 与一致性

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| snapshot 标准代码 | `600519.SH` | success；规范化 symbol；`snapshot_kind=daily_close`；as_of `2026-09-10 15:00+08:00`；source=tushare | PASS |
| snapshot 简写代码 | `600519` | success；规范化为 `600519.SH`，与标准代码结果一致 | PASS |
| snapshot 小写交易所 | `600519.sh` | success；规范化为 `600519.SH` | PASS |
| history 正常窗口 | `600519.SH`, `[2026-01-01, 2026-01-10)` | success；返回 1 月 5、6、7、8、9 日 5 根日线 | PASS |
| snapshot/history 最近数据一致 | snapshot 日期 `[2026-09-10, 2026-09-11)` | history 仅一根 9 月 10 日 bar；price/close、open、high、low、previous_close、pct_change、volume、amount 全部一致 | PASS |

### 2.2 日期与非法股票代码

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 周末空窗口 | `[2026-01-10, 2026-01-11)` | `EMPTY_RESULT/404/retryable=false` | PASS |
| end_date 右开 | `[2026-01-09, 2026-01-10)` | 仅返回 1 月 9 日 | PASS |
| 未来空窗口 | `[2099-01-01, 2099-01-02)` | `EMPTY_RESULT/404` | PASS |
| 相等/逆序日期 | 相等或 start > end | `INVALID_ARGUMENT/422` | PASS |
| 非法日期文本 | `start_date=xxx` | `INVALID_ARGUMENT/422` | PASS |
| 非法代码格式 | `BAD`、`AAPL`、`600519.NY` | `INVALID_SYMBOL/422` | PASS |
| 格式合法但不存在 | `999999.SH` | `INVALID_SYMBOL/422` | PASS |
| 未支持额外参数 | `adjustment=qfq` | `INVALID_ARGUMENT/422` | PASS |

### 2.3 provider 异常

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| provider 连接被拒绝 | Tushare base URL 指向未监听本地端口 | `PROVIDER_TIMEOUT/504/retryable=true` | FAIL（BB-02 同类） |
| provider 读超时 | 本地 socket 接受后不响应，timeout 50ms | `PROVIDER_TIMEOUT/504/retryable=true` | PASS |

## 3. RAG

### 3.1 测试条件

- 首轮标准 Qwen Tool 构建：本地索引存在，但当时缺少 Qwen API Key，构建时明确失败；未把该环境缺口计为产品 FAIL。
- 受控黑盒链：使用公开 `EmbeddingProvider`、`RerankerProvider`、`build_rag_index`、`register_rag_tools` 注入确定性的字符 n-gram embedding 和词面重排器；96 个 chunk 的隔离索引写入系统临时目录，测试后删除。
- Qwen live 补测：Key 配置后，标准 `build_rag_tools(Settings())` 在 0.8 秒内加载成功；8 个真实查询均 success，单次 Tool latency 约 293–636ms。
- 补测时当前工作树已切换到 `phase2` 且存在用户未提交改动，直接导入公开包发生循环导入。为避免修改或读取这些改动，测试仅将 Git `phase1` 引用的 `src/` 导出到系统临时目录运行，继续使用当前 `.env` 和既有 RAG 索引；未导出 `tests/`，临时副本在补测后删除。

### 3.2 三个 Tool、Evidence 与查询质量

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 研报广义查询 | “公司投资价值、盈利趋势、估值和主要风险” | success，5 条；全部 `source_type=research_report`；标题、公司、券商、发布日期、报告期齐全 | PASS |
| 单实体研报 | `companies=[青禾医药]`，查询业绩趋势与风险 | success，5 个 Evidence chunk；全部属于青禾医药，Top 结果相关 | PASS |
| 多实体研报 | `companies=[青禾医药, 瀚源新能源]` | success，8 个 Evidence chunk；覆盖两个实体；较单实体 5 条扩大召回范围 | PASS |
| 历史研报 | 青禾医药，`as_of=2025-06-30` | 只返回发布日期 `2025-04-03` 的研报 chunk；没有越过 as_of | PASS |
| 相似 FAQ：预警后怎么办 | query 指向维持担保比例预警，`category=margin` | Top 1 为 `FAQ-MARGIN-MAINTAIN-V1`“低于预警线怎么办” | PASS |
| 相似 FAQ：何时强平 | query 指向强制平仓，`category=margin` | Top 1 为 `FAQ-MARGIN-MAINTAIN-V2`“什么情况下会被强制平仓” | PASS |
| 当前法规 | “当前融资交易最低保证金比例是多少？” | 受控测试与真实 Qwen 均把 2024 superseded 排在 2026 active 之前；Qwen 分数为 0.997985 对 0.996875 | FAIL（BB-01） |
| 历史法规 | 同类问题，`as_of=2025-12-31` | 排除 2026 规则；Top 1 为 2024 版，所有 Evidence 日期不晚于 as_of | PASS |
| Evidence 来源隔离 | 分别调用三个 Tool | 研报只返回 research_report，法规只返回 announcement_policy，业务知识只返回 faq；metadata 中 document_id/title/source_type 一致 | PASS |

同一研究文档可返回多个不同 chunk，因此结果中 document_id 可能重复；本轮没有把不同 chunk 误判为重复 Evidence。

### 3.3 参数与 provider 错误

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 空 query | `query=""` | `INVALID_ARGUMENT/422/retryable=false` | PASS |
| 非法 category | `category=unknown` | `INVALID_ARGUMENT/422` | PASS |
| 非法 as_of | `as_of=not-date` | `INVALID_ARGUMENT/422` | PASS |
| 未知参数 | `top_k=99` | `INVALID_ARGUMENT/422` | PASS |
| 标准 Qwen adapter 不可达 | 有占位 key，embedding URL 指向未监听端口 | ToolResult `PROVIDER_TIMEOUT/504/retryable=true`，source=`synthetic_knowledge_corpus` | PASS |

### 3.4 Qwen live 补测结果

| 测试场景 | 实际结果 | 判定 |
| --- | --- | --- |
| 标准 RAG 构建 | Key 与既有索引可用；0.8 秒完成；公开 Registry 包含三个 RAG Tool | PASS |
| 单实体研报 | 青禾医药查询返回 5 条，全部属于青禾医药；Top 1 为 2026H1E 核心指标 chunk，score=0.921847 | PASS |
| 多实体研报 | 青禾医药 + 瀚源新能源返回 8 条，覆盖两个实体；相对单实体 5 条扩大召回 | PASS |
| 历史研报 | `as_of=2025-06-30` 仅返回 2025-04-03 研报的 2 个 chunk，未越界 | PASS |
| 相似 FAQ：预警 | Top 1 为 `FAQ-MARGIN-MAINTAIN-V1`，score=0.999566 | PASS |
| 相似 FAQ：强平 | Top 1 为 `FAQ-MARGIN-MAINTAIN-V2`，score=1.0 | PASS |
| 历史 FAQ | `as_of=2025-12-31` 排除 2026-03-01 生效的 V2；Top 1 为 V1 | PASS |
| 当前法规 | 2024 superseded 规则 Top 1（0.997985），2026 active 规则 Top 2（0.996875） | FAIL（BB-01） |
| 历史法规 | `as_of=2025-12-31` Top 1 为 2024 规则；所有 Evidence 日期均未越界 | PASS |

Qwen live 返回的 Evidence source 均为 `synthetic_knowledge_corpus`，三个 Tool 的 `source_type` 隔离正确，metadata 与标题/文档 ID 一致，未观察到错误或敏感信息回显。

## 4. 跨模块基本稳定性与 CLI

| 测试场景 | 输入 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| 同一 Registry 连续跨 Tool 调用 | context → margin → positions → analytics | 4 次均 success，request_id 独立 | PASS |
| 错误后恢复 | 上述序列后调用 margin `limit=0`，再调用 context | 中间请求返回 422；后续 context 立即 success | PASS |
| Market 错误后正常请求 | 非法代码后查询合法代码 | 合法请求成功，未受前一错误污染 | PASS |
| CLI 自检 | `python -m financial_agent` | exit 0；输出 `phase=0/status=ready/data_mode=synthetic` | PASS |
| CLI 帮助 | `--help`、`call-tool --help`、`serve-user-data --help` | 均可正常显示 | PASS |
| CLI Tool 发现 | `list-tools` | exit 0，但只列出 4 个 User Data Tool；Market/RAG Tool 只能通过 Python 公开 Registry 使用 | PASS（限制） |
| CLI Tool 错误退出码 | 401/403/404/422/unknown tool | 均 exit 1；success/empty 均 exit 0 | PASS |

## 5. 发现的问题与复现步骤

### BB-01：当前法规可能优先返回已废止版本

- 严重程度：中
- 影响：Agent 若只消费 Top 1 Evidence，可能把 2024 年已废止比例当作当前规则；metadata 虽标记 `superseded`，但排序没有保证 active 优先。
- 实际结果：真实 Qwen 查询“当前融资交易最低保证金比例是多少？”时，`AP-MARGIN-RATIO-202401`（status=superseded，内容为 80%）以 0.997985 排第一；`AP-MARGIN-RATIO-202601`（status=active，内容为 90%）以 0.996875 排第二。
- 复现步骤：
  1. 配置 Qwen Key 和既有索引，通过标准 `build_rag_tools(Settings())` 构建 RAG Registry。
  2. 调用 `search_regulatory_knowledge`，输入 `{"query":"当前融资交易最低保证金比例"}`，不传 `as_of`。
  3. 比较返回 Evidence 的 `metadata.status/effective_date` 和顺序。
- 备注：该问题已同时在受控 reranker 和真实 Qwen reranker 下复现，不再只是待确认风险。

### BB-02：连接被拒绝被误归类为超时

- 严重程度：低
- 影响：监控、告警和重试策略无法区分“端口未监听/立即拒绝”与“服务已连接但响应超时”，会降低诊断准确性。
- 实际结果：User Data 返回 `TIMEOUT/504`；Market Data 返回 `PROVIDER_TIMEOUT/504`。两者都标记 retryable=true，未泄露地址或底层异常。
- 复现步骤：
  1. 确认 `127.0.0.1:18081` 没有监听服务。
  2. User Data：设置进程级 `FINANCIAL_AGENT_USER_DATA_BASE_URL=http://127.0.0.1:18081`，调用任一 User Tool。
  3. Market Data：构造 `Settings(tushare_base_url="http://127.0.0.1:18081")`，调用 `get_market_snapshot`。
  4. 观察错误 code 均为 timeout，而不是 connection/unavailable 类错误。

### BB-03：HTTP API 静默忽略未知 query 参数

- 严重程度：低
- 影响：调用方拼错参数名时仍得到 200，容易误以为过滤条件生效；Tool 层本身会拒绝未知字段，HTTP 与 Tool 严格性不一致。
- 实际结果：`GET /v1/customers/syn-user-0001/context?unexpected=secret-looking-value` 返回 200 和正常领域数据。
- 复现步骤：
  1. 启动 `serve-user-data` 并配置合法 key/whitelist/scope。
  2. 对任一 GET endpoint 添加不存在的 query 参数。
  3. 观察请求仍返回 200，且没有参数错误提示。

### BB-04：access log 原样记录 query 参数值

- 严重程度：中
- 影响：如果上游误把 token、账户号或其他敏感值放入 query，服务访问日志会持久化该值。业务错误正文没有此泄露，但默认 server access log 仍存在风险。
- 实际结果：服务 stderr 出现完整请求行，包含 `?unexpected=secret-looking-value`。
- 复现步骤：
  1. 启动默认 `serve-user-data`。
  2. 请求 `/v1/customers/syn-user-0001/context?unexpected=secret-looking-value`。
  3. 查看服务 stderr/access log，可见完整 query 值。

## 6. 未覆盖与后续建议

- 已完成针对核心场景的 Qwen live RAG 补测；未运行仓库中的完整 `-m live` pytest，以继续遵守“不读取或参考现有 tests”的黑盒限制。
- 建议修复 BB-01 后，使用同一 live 输入回归，要求无 `as_of` 的“当前”查询将 active 版本置于 superseded/expired 版本之前。
- 未测试真实 Tushare 限流、权限拒绝或服务端 5xx；本轮仅覆盖真实正常行情、连接拒绝和受控读超时。
- 未执行任何现有 pytest，以遵守“不参考已有单测实现”的要求。
- 未读取审计正文；仅验证调用结果和服务日志没有直接回显 API Key、SQL、数据库路径或堆栈。
