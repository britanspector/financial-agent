# Phase 3.1 Structured Planner 黑盒测试报告

测试日期：2026-09-11（Asia/Hong_Kong）  
方法：仅使用 README 声明的公开 Python 调用链 `build_agent_tools → build_planner → planner.plan → validator.validate`。未读取 `src/`、`tests/`、Planner prompt、现有 eval cases/答案；未执行任何数据 Tool；未修改代码或配置。`当前`按运行日 2026-09-11 判断。

标记说明：`PASS` 表示 Plan 可通过 Validator 且从业务需求看工具、参数和并行关系合理。`FAIL` 包含不可执行、漏调用、无关调用、编造参数或业务方向错误。输出中的 `:` 前缀是模型实际生成的参数键，不是转写错误。

## 逐例结果

| ID | Query | Planner 输出（实际 Task DAG） | 结论 / 类型 / 严重度 | 不合理之处与正确方向 |
|---|---|---|---|---|
| B01 | 查询 600519.SH 今天的日终收盘行情 | `get_market_snapshot({:symbol:'600519.SH'})`；Validator: INVALID_ARGUMENTS | FAIL；单行情/当前；高 | 参数键错误，缺少 `symbol`。应仅调用 snapshot，`{symbol:'600519.SH'}`。 |
| B02 | 查询 000001.SZ 在 2026-08-01 至 2026-09-01 的日线 | `get_market_history({:symbol,:start_date,:end_date})`；INVALID_ARGUMENTS | FAIL；单行情/历史；高 | 时间和值方向正确但三参数键均错误。应为 history 的标准三字段，区间保持 `[01,01)`。 |
| B03 | 查询 syn-user-0001 当前持仓明细 | `get_portfolio_positions({user_id:'syn-user-0001'})` | PASS；单用户数据/持仓；低 | 工具、参数、无依赖均合理。 |
| B04 | 查询 syn-user-0001 2026-08-01 至 09-01 两融流水 | `get_margin_account({user_id,start_date:'2026-08-01',end_date:'2026-09-01'})` | PASS；两融历史；低 | 正确；Validator 补齐 `limit:60,offset:0`。 |
| B05 | 查询 syn-user-0002 风险偏好与投资期限 | `get_customer_context({user_id:'syn-user-0002'})` | PASS；客户画像；低 | 正确。 |
| B06 | 查询 syn-user-0003 收益、回撤、同组排名 | `get_portfolio_analytics({user_id:'syn-user-0003'})` | PASS；组合分析；低 | 正确。 |
| B07 | 华泰科技最新研报及评级观点 | `search_research_reports({:query:'华泰科技',:companies:['华泰科技']})`；INVALID_ARGUMENTS | FAIL；单RAG/研报；高 | 检索方向正确但参数键错误。应传 `query`，可加 `companies`。 |
| B08 | 证监会现行融资融券监管规则 | `search_regulatory_knowledge({:query,:issuer:'证监会',:as_of:null})`；INVALID_ARGUMENTS | FAIL；单RAG/法规；高 | 参数键错误；当前规则应 `query`，issuer 可选，省略 `as_of` 或为 null。 |
| B09 | 融资融券开通业务说明 | `search_business_knowledge({:query,:category:'margin'})`；INVALID_ARGUMENTS | FAIL；单RAG/业务知识；高 | 参数键错误。应以开户/开通为 query，`category:'margin'`。 |
| B10 | 什么是夏普比率 | `search_business_knowledge({query:'夏普比率定义和计算方法',category:'risk_notice'})` | FAIL；无需工具边界；中 | 通用概念可直接回答；调用内部业务知识且强加风险提示分类无必要。应无 Tool。 |
| B11 | 你好，请介绍能做什么 | `[]`；Validator: EMPTY_PLAN | FAIL；无需工具边界；低 | 业务上不应调用 Tool，但当前结构无法让空计划通过 Validator。应有可验证的 abstain/直接回答表示。 |
| B12 | 帮我查看我的持仓 | `context({user_id:''}) → positions({user_id:'{{get_customer_context.user_id}}'})`；INVALID_ARGUMENTS | FAIL；信息不足；高 | 编造空 ID、使用未支持的动态绑定，且无须先查客户信息。应请求 user_id/身份澄清并不调用 Tool。 |
| B13 | 查贵州茅台现在多少钱 | `snapshot({:symbol:'600519.SH'})`；INVALID_ARGUMENTS | FAIL；信息不足；中 | 虽识别到代码，Planner 未确认“贵州茅台”与受支持证券代码的歧义；且参数错误。应澄清或以明确代码后 snapshot。 |
| B14 | 我应该买股票还是债券 | `get_customer_context({user_id:'unknown'})` | FAIL；信息不足/建议；高 | 编造 `unknown` 用户 ID；缺少客户身份与投资目标。应澄清，不调用 Tool。 |
| B15 | 600519.SH 在 2026-08-15 的收盘价 | `history({:symbol,:start_date:'2026-08-15',:end_date:'2026-08-16'})`；INVALID_ARGUMENTS | FAIL；单日历史；高 | 区间推导正确，参数键错误。 |
| B16 | 000001.SZ 最新日终收盘 | `snapshot({:symbol:'000001.SZ'})`；INVALID_ARGUMENTS | FAIL；最新快照；高 | 仅键名错误。 |
| B17 | syn-user-0001 客户信息和持仓，判断风险适配 | `context, positions, margin_account`（均无依赖） | FAIL；用户组合/并行；中 | 前两项足够；未要求两融且该调用无关。应只并行 customer_context + positions。 |
| B18 | syn-user-0002 持仓集中度与最大回撤 | `positions, portfolio_analytics`（并行） | PASS；用户组合/并行；低 | 两工具都必要且独立。 |
| B19 | syn-user-0003 风险等级及最近30天两融余额变化 | `context, portfolio_analytics, margin_account(2026-08-12,2026-09-11)` | FAIL；用户组合/相对时间；中 | analytics 未被问题要求，是无关调用；30天起点也应为 08-12 或按日历定义明确，但应只用 context + margin。 |
| B20 | syn-user-0001 客户、持仓、分析、两融账户概览 | 四个用户 Tool，全部无依赖 | PASS；四工具并行；低 | 需求完整覆盖；全部可并行。 |
| B21 | 对比两只股票最新收盘 | 两个 `snapshot({:symbol})` 并行；INVALID_ARGUMENTS×2 | FAIL；多实体行情；高 | 分解/并行正确，两个参数键错误。 |
| B22 | 比较两只股票 2026-08 涨跌 | 两个 `history({:symbol,:start_date:'2026-08-01',:end_date:'2026-09-01'})`；INVALID_ARGUMENTS×2 | FAIL；多实体历史；高 | 分解及时间正确，参数键错误。 |
| B23 | 华泰科技、星河能源最新研报 | 两个 `search_research_reports({:query,:companies})` 并行；INVALID_ARGUMENTS×2 | FAIL；多实体研报；高 | 公司分解合理，参数键错误。 |
| B24 | 中信证券、截至 2026-08-31 的华泰科技研报 | `research({:query:'华泰科技',:brokers:['中信证券'],:as_of:'2026-08-31'})`；INVALID_ARGUMENTS | FAIL；研报过滤；高 | 过滤意图和日期正确，参数键错误。 |
| B25 | 截至 2025-12-31 有效的融资融券规定 | `regulatory({:query,:as_of:'2025-12-31'})`；INVALID_ARGUMENTS | FAIL；法规历史；高 | 历史时点正确，参数键错误。 |
| B26 | 当前有效的投资者适当性规则 | `regulatory({:query,:as_of:null})`；INVALID_ARGUMENTS | FAIL；法规当前；高 | 应以 `query` 检索，省略 as_of 即可；实际键错误。 |
| B27 | 截至 2025-12-31 生效的风险测评业务规则 | 重复 `task_id:'search_risk_rules'`；其中一个 `{: : ' '}`，另一个 `business({query,category:'risk_notice',as_of})`；DUPLICATE_TASK_ID/INVALID_ARGUMENTS | FAIL；业务知识历史；高 | 出现重复 ID 和损坏任务；应单一 business 知识查询，且需确认风险测评业务规则而非泛风险提示分类。 |
| B28 | 融资融券的研报、现行法规、开户说明 | `regulatory({:query})`,`business({:query,:category})`,`research({:query,:brokers:['华泰证券']})`；INVALID_ARGUMENTS | FAIL；三RAG并行；高 | 三路并行方向合理，但全部参数键错；研报还将“华泰科技”误作券商“华泰证券”。应以公司 filter 检索华泰科技。 |
| B29 | syn-user-0001 的 600519.SH 数量及最新价 | `positions({user_id})`,`snapshot({symbol:'600519.SH'})` 并行 | PASS；用户+行情；低 | 两项独立且完整。 |
| B30 | syn-user-0002 两融风险：客户、持仓、分析、法规 | `context, margin, positions, analytics, regulatory({query})` 并行 | PASS；用户+法规；低 | 所选信息均与两融风险评估有关；可并行。 |
| B31 | 根据我的持仓找最大仓位再查其今日行情 | `context({})`; `positions({})` 依赖不存在的 `go get_customer_context`；INVALID_ARGUMENTS/MISSING_DEPENDENCY | FAIL；前后依赖边界；高 | 无 user_id，且当前不支持把最大仓位结果绑定到行情参数。应澄清身份，并声明需 result binding 后再做依赖计划。 |
| B32 | 比较组合近期开销与 600519.SH 2026-08 表现 | `margin(2026-08-01,09-01)`,`history(symbol,2026-08-01,09-01)` 并行 | PASS；用户+行情；低 | 费用应取两融账户；时间、工具及并行均正确。 |
| B33 | 2026-07-01 至 07-15 两融记录，第2页每页20条 | `margin({user_id,start:'2026-07-01',end:'2026-07-16',limit:20,offset:20})` | PASS；两融分页/时间；低 | 将含尾日转换为开区间 07-16、第二页 offset=20，合理。 |
| B34 | 今天 A 股市场新闻 | `research({:query:'A股市场新闻'})`；INVALID_ARGUMENTS | FAIL；能力边界；中 | 研报不是“今天新闻”数据源，且参数错误。应说明无新闻 Tool，不调用。 |
| B35 | 上证指数今天实时点位 | `snapshot({:symbol:'000001.SH'})`；INVALID_ARGUMENTS | FAIL；能力边界；高 | 不支持指数和实时行情；不应以股票日终 Tool 代替。 |
| B36 | 100 美元换人民币 | `[]`；EMPTY_PLAN | FAIL；能力边界；低 | 语义上正确地未调用 Tool，但同 B11，空计划无法通过。应有 abstain/能力不足的合法输出。 |
| B37 | syn-user-0001 是否适合两融：客户、持仓、法规 | `context, margin, positions, regulatory` 并行 | PASS；适当性/用户+法规；低 | margin 账户对两融适配判断有直接价值；均相关。 |
| B38 | syn-user-0002 调仓前风险检查：客户、分析、持仓、风险测评和适当性规则 | `context,margin,positions,analytics,business(risk_notice),business(account)` 并行 | FAIL；复杂组合；中 | 多加 margin；“投资者适当性规则”应优先法规知识而非账户业务 FAQ；类别也不可靠。应 context+positions+analytics+风险测评业务知识+适当性法规。 |
| B39 | 分别查两个用户当前持仓 | 两个 `positions({user_id})` 并行 | PASS；多实体用户数据；低 | 正确分解且无依赖。 |
| B40 | 截至 2025-06-30 发布的星河能源研报 | `research({:query,:companies,:as_of:'2025-06-30'})`；INVALID_ARGUMENTS | FAIL；研报历史；高 | 语义及日期正确，参数键错误。 |
| B41 | 截至 2027-01-01 有效的融资融券法规 | `regulatory({:query,:as_of:'2027-01-01'})`；INVALID_ARGUMENTS | FAIL；法规未来时间；中 | 参数错误；此外未来时点应先说明只能检索当前已知材料，不能把未来有效性当作已确定事实。 |
| B42 | 600519.SH [2026-09-01, 2026-09-11) 行情 | `history({:symbol,:start_date:'2026-09-01',:end_date:'2026-09-11'})`；INVALID_ARGUMENTS | FAIL；时间边界；高 | 开闭区间完全正确，三键错误。 |
| B43 | history 给出 600519.SH，查过去5交易日日线 | `history({:symbol,:start_date:'2026-09-04',:end_date:'2026-09-11'})`；INVALID_ARGUMENTS | FAIL；history/相对时间；高 | 正确继承历史代码并给出合理工作日窗口，但参数键错误；“5个交易日”最好由交易日历而非日历回推验证。 |
| B44 | 写风险提醒邮件 | `context({user_id:'placeholder_user_id'})` | FAIL；无需工具/写作；高 | 编造占位用户 ID、无关调用。应直接起草通用邮件，或询问收件人/客户信息。 |
| B45 | 预测 600519.SH 下周涨跌 | `snapshot({:'symbol'}),history({:'symbol'})`；INVALID_ARGUMENTS | FAIL；预测边界；高 | 参数已损坏；且缺少可支持预测的模型/新闻数据，不能把预测包装为普通行情查询。应说明能力限制，若用户要观察可查历史/快照但不承诺预测。 |
| B46 | syn-user-0001 最近一个月两融利息费用变化及费用规则 | `context,margin(2026-08-11,09-11),business({query,category:'margin'})` 并行 | FAIL；用户+业务知识；中 | customer context 未被请求，是无关调用；其余方向和窗口合理。应 margin + business knowledge。 |
| B47 | syn-user-0003 持仓是否超允许风险等级，并查适当性规则 | `context,positions,business({query,category:'risk_notice'})` 并行 | FAIL；适配/用户+法规；中 | 前两项正确；“适当性规则”应使用法规检索而非风险提示业务知识。 |
| B48 | 华泰科技最新研报结合 600519.SH 最新价 | `snapshot({:'dependencies'}), research({query:'华泰科技',companies:['600519.SH'],as_of:'2026-09-11'})`；INVALID_ARGUMENTS | FAIL；研报+行情；高 | 行情参数完全错；研报 company filter 将股票代码误作华泰科技，漏掉正确公司实体。应并行 snapshot(600519.SH) + research(query/companies=华泰科技)。 |
| B49 | 000001.SZ 上周日线 | `history({:'end_date'})`；INVALID_ARGUMENTS | FAIL；相对时间；高 | 丢失代码和时间起点、键和值都错误。应按当前日换算完整上周区间并传三字段。 |
| B50 | 600519.SH 上个月日线 | `history({:symbol,:start_date:'2026-08-01',:end_date:'2026-09-01'})`；INVALID_ARGUMENTS | FAIL；相对时间；高 | 上个月区间推导正确，参数键错误。 |

## 汇总

- 共 50 条；`PASS 13`、`FAIL 37`。Validator 可执行率为 `22/50`（44%）：B10、B14、B17、B19、B38、B44、B46、B47 虽可执行，仍因业务语义失败。
- 最常见 bad case 是**参数序列化损坏**：行情与 RAG 多次把合法字段名生成 `:symbol`、`:query` 等，至少 24 条直接导致 `INVALID_ARGUMENTS`。这是可执行性层面的最高优先级问题。
- 其次是**信息不足时编造**：空字符串、`unknown`、`placeholder_user_id` 和未支持的动态绑定，见 B12/B14/B31/B44；应改为明确 abstain/澄清协议。
- **能力边界误路由**：新闻、指数实时、汇率、预测被错配到研报或行情 Tool（B34/B35/B45）。
- **语义冗余/工具域混淆**：B17/B19/B38/B46 多调用户数据；B38/B47 把适当性法规混同为业务 FAQ。
- **DAG/依赖问题**：多数独立工作能正确并行，但 B31 生成不存在 dependency，且不能解决“上游持仓 → 最大仓位代码 → 行情”的结果绑定需求。
- **空计划与 Validator 语义冲突**：B11/B36 的业务决策是不调用 Tool，但 Validator 将其判为 `EMPTY_PLAN`；应定义可验证的无工具/澄清结果，而非把它当成坏计划。

本报告仅记录 Planner 计划与 Validator 结果；未对代码作出修改或实施修复。
