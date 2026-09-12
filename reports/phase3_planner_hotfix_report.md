# Phase 3.1 Planner Hotfix 验证报告

测试日期：2026-09-11（Asia/Hong_Kong）  
范围：仅 Structured Planner、Provider structured output、Prompt、Plan Validator 与 Planner Eval；未实现 Result Binding、Retry 或 Verifier/Replan。

## 1. 参数键损坏：证据与根因

hotfix 前，用生产 Qwen adapter 对公开 Planner 调用链采样，Provider 解码后的原始 structured-output object 分别为：

```json
{"tasks":[{"tool_name":"get_market_snapshot","arguments":{":symbol":"600519.SH"}}]}
{"tasks":[{"tool_name":"search_research_reports","arguments":{":query":"华泰科技",":brokers":["华泰证券"]}}]}
{"tasks":[{"tool_name":"search_business_knowledge","arguments":{":query":"融资融券开通业务说明",":category":"margin"}}]}
```

Qwen adapter 仅将 `choices[0].message.content` 作 `json.loads`，随后才由 `StructuredPlan.model_validate` 和 `PlanValidator` 消费；没有任何参数键转换逻辑。因此 `:symbol` / `:query` 是模型原始输出，不是 Provider 或 Pydantic 改写。

当时的 `StructuredPlan` JSON Schema 将 `PlannedTask.arguments` 描述为开放 object（`additionalProperties: true`），没有按 `tool_name` 约束到 Tool input schema。这是根因。

### 修复

- Provider request schema 改为 Tool-aware `oneOf`：每个 task 的 `tool_name` 为常量，`arguments` 直接复制对应公开 Tool input schema。
- object schema 关闭额外属性；严格 decoder 所需的声明字段全部显式出现，nullable filter 用 `null`、空 list filter 用 `[]` 表示无 filter。
- Provider 保留仅当前 adapter 实例可见的 `last_raw_response`，便于比较原始输出与 Planner parse；不写日志，避免泄露 query/history。
- fixture 覆盖 `get_market_history`、`search_research_reports`、`search_regulatory_knowledge` 和 `search_business_knowledge`，验证 schema 参数名、Planner 参数和 Validator 均一致。
- Qwen live regression 覆盖 Market 与三个 RAG Tool，检查实际执行任务不会出现 `:` 前缀参数名。

## 2. 合法的非执行决策

`StructuredPlan.decision` 为：

| decision | tasks | Validator 行为 |
| --- | --- | --- |
| `execute` | 非空 | 校验并转换为 Phase 2 Task |
| `clarify` | 空 | 合法，返回空 Task |
| `no_tool` | 空 | 合法，返回空 Task |

缺少 user_id、股票代码或其它必填业务参数时，Planner 必须选择 `clarify`，不能填入 `unknown`、placeholder 或空字符串。请求无需 Tool、或所有公开 Tool 都不支持的能力时使用 `no_tool`。

## 3. 通用业务边界

- regulatory knowledge：法规、监管、适当性规则。
- business knowledge：FAQ、业务流程、产品说明。
- research reports：研究报告证据，不是实时新闻；“最新研报”表示最新可用报告。
- market tools：仅现有 A 股股票日终/历史行情；不支持指数实时、新闻、汇率或价格预测。
- 只使用完成请求所需的 Tool；不编造业务参数。

## 4. 真实 Qwen Eval

运行使用固定 `eval/planner/planner_cases.jsonl` 的 40 条 case，按原文件顺序拆为四个连续 10-case batch（避免长连续调用的终端输出捕获问题）。所有 batch 均在最终 hotfix prompt/schema 后调用配置的 Qwen Provider；未执行任何业务 Tool。

| Case 范围 | valid plan rate | tool selection | argument | temporal | dependency | unnecessary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–10 | 1.0000 | 0.9000 | 0.6667 | 1.0000 | 1.0000 | 0.0000 |
| 11–20 | 1.0000 | 0.6000 | 0.3000 | 0.6000 | 1.0000 | 0.1250 |
| 21–30 | 1.0000 | 0.6000 | 0.3077 | 0.0000 | 1.0000 | 0.1000 |
| 31–40 | 1.0000 | 0.8000 | 0.4667 | 1.0000 | 0.6000 | 0.0714 |
| **40-case aggregate** | **1.0000** | **0.7250** | **0.4400** | **0.5000** | **0.9000** | **0.0682** |

聚合的 executable valid-plan 分母为 36；argument 和 temporal 使用 Eval 的对应期望项分母；dependency 为 case 平均；unnecessary 为全部实际 Planner task 的加权比率。

### 与 hotfix 前对比

| 指标 | 修复前 | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| valid plan rate | 0.5000 | 1.0000 | +0.5000 |
| tool selection accuracy | 0.6750 | 0.7250 | +0.0500 |
| argument accuracy | 0.4600 | 0.4400 | -0.0200 |
| temporal accuracy | 0.3333 | 0.5000 | +0.1667 |
| dependency accuracy | 0.8750 | 0.9000 | +0.0250 |
| unnecessary tool call rate | 0.3030 | 0.0682 | -0.2348 |

参数精确率的小幅下降不表示 `:symbol` / `:query` 回归：这些非法键在最终 live regression 中未再出现。新 strict schema 会要求模型显式输出一些带默认值的 optional filter（`null` / `[]`），而旧 Eval 的部分 canonical pattern 省略这些字段，仍按原始 arguments 字典精确比较，因而计为不相等。

## 5. 验证

- 默认测试：`python -m pytest -q` → `199 passed, 12 deselected`。
- Planner live：`python -m pytest -q -m live tests/planner/test_live.py` → `5 passed`。
- `git diff --check` 通过。
- 敏感信息扫描未发现凭证；扫描中出现的 `RISK-*` 为测试/知识库标识，不是 secret。

## 6. 仍存在的 bad cases / 限制

- Qwen 对一部分“最新研报”类自然语言请求仍可能选择 `clarify`，尽管研究报告检索能力和 query 已足够；这属于模型的 Tool selection 偏差，不是参数键损坏。
- 历史区间推导、可选 filter 选择与复杂多 Tool 路由仍造成 argument/temporal accuracy 损失。
- 上游结果派生下游参数（例如从持仓取最大仓位后查价格）仍不能规划为 execute；需未来 Result Binding，明确不在本 hotfix 范围内。
- `clarify` / `no_tool` 目前只表达结构化 Planner 决策，尚无面向用户的澄清文案生成。
