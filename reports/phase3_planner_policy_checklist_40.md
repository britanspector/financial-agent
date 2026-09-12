# Planner Eval Case Report

## Metrics

```json
{
  "case_count": 40,
  "executable_case_count": 36,
  "abstention_case_count": 4,
  "valid_plan_rate": 1.0,
  "tool_selection_accuracy": 0.825,
  "argument_accuracy": 0.78,
  "temporal_accuracy": 0.75,
  "dependency_accuracy": 0.9,
  "unnecessary_tool_call_rate": 0.06382978723404255
}
```

## Cases

### user_context

Query: 查询 syn-user-0001 的客户资料

Expected plan:

```json
{
  "expected_tools": [
    "get_customer_context"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### user_positions

Query: 查询 syn-user-0002 的当前持仓

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_positions"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0002"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### user_analytics_from_history

Query: 查看他的组合分析

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_analytics"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_analytics",
      "arguments": {
        "user_id": "syn-user-0003"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_portfolio_analytics",
      "arguments": {
        "user_id": "syn-user-0003"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### margin_current

Query: 查询 syn-user-0001 的两融账户

Expected plan:

```json
{
  "expected_tools": [
    "get_margin_account"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0001",
        "start_date": null,
        "end_date": null,
        "limit": 60,
        "offset": 0
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `True` / `True` / `True`
- reasons: tool selection differs from selected expected plan; 1 unnecessary Tool call(s)


### margin_history_dates

Query: 查询 syn-user-0001 在 2026-05-01 到 2026-06-01 的两融历史

Expected plan:

```json
{
  "expected_tools": [
    "get_margin_account"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0001",
        "start_date": "2026-05-01",
        "end_date": "2026-06-01"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "get_margin_account",
      "arguments": {
        "start_date": "2026-05-01",
        "end_date": "2026-06-01"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "1",
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0001",
        "start_date": "2026-05-01",
        "end_date": "2026-06-01",
        "limit": 366,
        "offset": 0
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `False` / `True` / `True`
- reasons: one or more required values or non-default arguments differ

  - argument `get_margin_account` (incorrect): required value or non-default argument differs; expected={'user_id': 'syn-user-0001', 'start_date': '2026-05-01', 'end_date': '2026-06-01'}; actual={'user_id': 'syn-user-0001', 'start_date': '2026-05-01', 'end_date': '2026-06-01', 'limit': 366, 'offset': 0}

### all_user_data_parallel

Query: 并行给出 syn-user-0002 的资料、持仓、组合分析和两融账户

Expected plan:

```json
{
  "expected_tools": [
    "get_customer_context",
    "get_portfolio_positions",
    "get_portfolio_analytics",
    "get_margin_account"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    },
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    },
    {
      "tool_name": "get_portfolio_analytics",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    },
    {
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0002"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0002"
      },
      "dependencies": []
    },
    {
      "task_id": "t3",
      "tool_name": "get_portfolio_analytics",
      "arguments": {
        "user_id": "syn-user-0002"
      },
      "dependencies": []
    },
    {
      "task_id": "t4",
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0002",
        "start_date": null,
        "end_date": null,
        "limit": 60,
        "offset": 0
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### user_id_from_history

Query: 再看一下持仓

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_positions"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### missing_user_id_abstain

Query: 查询我的持仓

Expected plan:

```json
{
  "expected_tools": [],
  "expected_arguments": [],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### market_snapshot_maotai

Query: 查询 600519.SH 当前行情

Expected plan:

```json
{
  "expected_tools": [
    "get_market_snapshot"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### market_snapshot_pingan

Query: 查询 000001.SZ 当前收盘快照

Expected plan:

```json
{
  "expected_tools": [
    "get_market_snapshot"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "000001.SZ"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "000001.SZ"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### market_history_half_open

Query: 查询 600519.SH 在 [2026-01-01, 2026-02-01) 的历史行情

Expected plan:

```json
{
  "expected_tools": [
    "get_market_history"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_market_history",
      "arguments": {
        "symbol": "600519.SH",
        "start_date": "2026-01-01",
        "end_date": "2026-02-01"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "get_market_history",
      "arguments": {
        "start_date": "2026-01-01",
        "end_date": "2026-02-01"
      }
    }
  ]
}
```

Actual plan:

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

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### market_history_inclusive_end

Query: 查询 000001.SZ 在 2026年3月1日至2026年3月31日（含首尾）的历史行情

Expected plan:

```json
{
  "expected_tools": [
    "get_market_history"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_market_history",
      "arguments": {
        "symbol": "000001.SZ",
        "start_date": "2026-03-01",
        "end_date": "2026-04-01"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "get_market_history",
      "arguments": {
        "start_date": "2026-03-01",
        "end_date": "2026-04-01"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_market_history",
      "arguments": {
        "symbol": "000001.SZ",
        "start_date": "2026-03-01",
        "end_date": "2026-04-01"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### market_only_forbidden_tools

Query: 只查询 600519.SH 当前行情，不要用户数据或知识库

Expected plan:

```json
{
  "expected_tools": [
    "get_market_snapshot"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### research_single_company

Query: 检索澄海智造的经营展望研报

Expected plan:

```json
{
  "expected_tools": [
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "澄海智造的经营展望研报",
        "companies": [
          "澄海智造"
        ]
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "澄海智造的经营展望研报",
        "companies": [
          "澄海智造"
        ],
        "brokers": [],
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### research_broker_as_of

Query: 查甲辰证券研究所截至 2026-06-30 对青禾医药的研报

Expected plan:

```json
{
  "expected_tools": [
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "甲辰证券研究所截至 2026-06-30 对青禾医药的研报",
        "companies": [
          "青禾医药"
        ],
        "brokers": [
          "甲辰证券研究所"
        ],
        "as_of": "2026-06-30"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "as_of": "2026-06-30"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "甲辰证券研究所对青禾医药的研报",
        "companies": [
          "青禾医药"
        ],
        "brokers": [
          "甲辰证券研究所"
        ],
        "as_of": "2026-06-30"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct

  - argument `search_research_reports` (acceptable): query wording is semantically equivalent; expected={'query': '甲辰证券研究所截至 2026-06-30 对青禾医药的研报', 'companies': ['青禾医药'], 'brokers': ['甲辰证券研究所'], 'as_of': '2026-06-30'}; actual={'query': '甲辰证券研究所对青禾医药的研报', 'companies': ['青禾医药'], 'brokers': ['甲辰证券研究所'], 'as_of': '2026-06-30'}

### research_multi_company

Query: 比较曜石半导体和云帆软件的最新经营研报

Expected plan:

```json
{
  "expected_tools": [
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "比较曜石半导体和云帆软件的最新经营研报",
        "companies": [
          "曜石半导体",
          "云帆软件"
        ]
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "比较曜石半导体和云帆软件的最新经营研报",
        "companies": [
          "曜石半导体",
          "云帆软件"
        ],
        "brokers": [],
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### research_multi_entity_historical

Query: 找衡川证券研究所截至 2026-05-01 对瀚源新能源和远陆物流的研报

Expected plan:

```json
{
  "expected_tools": [
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "找衡川证券研究所截至 2026-05-01 对瀚源新能源和远陆物流的研报",
        "companies": [
          "瀚源新能源",
          "远陆物流"
        ],
        "brokers": [
          "衡川证券研究所"
        ],
        "as_of": "2026-05-01"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "as_of": "2026-05-01"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "衡川证券研究所对瀚源新能源和远陆物流的研报",
        "companies": [
          "Hanyuan New Energy",
          "Yuanlu Logistics"
        ],
        "brokers": [
          "Hengchuan Securities Research Institute"
        ],
        "as_of": "2026-05-01"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `False` / `True` / `True`
- reasons: one or more required values or non-default arguments differ

  - argument `search_research_reports` (incorrect): required value or non-default argument differs; expected={'query': '找衡川证券研究所截至 2026-05-01 对瀚源新能源和远陆物流的研报', 'companies': ['瀚源新能源', '远陆物流'], 'brokers': ['衡川证券研究所'], 'as_of': '2026-05-01'}; actual={'query': '衡川证券研究所对瀚源新能源和远陆物流的研报', 'companies': ['Hanyuan New Energy', 'Yuanlu Logistics'], 'brokers': ['Hengchuan Securities Research Institute'], 'as_of': '2026-05-01'}

### research_unfiltered_topic

Query: 检索银行资本补充相关的研究报告

Expected plan:

```json
{
  "expected_tools": [
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "银行资本补充相关的研究报告"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "银行资本补充相关的研究报告",
        "companies": [],
        "brokers": [],
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### research_company_from_history

Query: 查看截至 2026-04-30 的该公司研报

Expected plan:

```json
{
  "expected_tools": [
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "查看截至 2026-04-30 的该公司研报",
        "companies": [
          "沧澜消费"
        ],
        "as_of": "2026-04-30"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "as_of": "2026-04-30"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "查看截至 2026-04-30 的该公司研报",
        "companies": [
          "Canglan Consumer"
        ],
        "brokers": [],
        "as_of": "2026-04-30"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `False` / `True` / `True`
- reasons: one or more required values or non-default arguments differ

  - argument `search_research_reports` (incorrect): required value or non-default argument differs; expected={'query': '查看截至 2026-04-30 的该公司研报', 'companies': ['沧澜消费'], 'as_of': '2026-04-30'}; actual={'query': '查看截至 2026-04-30 的该公司研报', 'companies': ['Canglan Consumer'], 'brokers': [], 'as_of': '2026-04-30'}

### regulation_current_margin

Query: 查询中证融资服务中心当前有效的最低保证金比例规则

Expected plan:

```json
{
  "expected_tools": [
    "search_regulatory_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "当前有效的最低保证金比例规则",
        "issuer": "中证融资服务中心"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "中证融资服务中心当前有效的最低保证金比例规则",
        "issuer": null,
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct

  - argument `search_regulatory_knowledge` (acceptable): entity filter omitted but preserved verbatim in query; expected={'query': '当前有效的最低保证金比例规则', 'issuer': '中证融资服务中心'}; actual={'query': '中证融资服务中心当前有效的最低保证金比例规则', 'issuer': None, 'as_of': None}

### regulation_historical_margin

Query: 查询中证融资服务中心截至 2025-12-31 的最低保证金比例规则

Expected plan:

```json
{
  "expected_tools": [
    "search_regulatory_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "截至 2025-12-31 的最低保证金比例规则",
        "issuer": "中证融资服务中心",
        "as_of": "2025-12-31"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "as_of": "2025-12-31"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "中证融资服务中心最低保证金比例规则",
        "issuer": "中证融资服务中心",
        "as_of": "2025-12-31"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct

  - argument `search_regulatory_knowledge` (acceptable): query wording is semantically equivalent; expected={'query': '截至 2025-12-31 的最低保证金比例规则', 'issuer': '中证融资服务中心', 'as_of': '2025-12-31'}; actual={'query': '中证融资服务中心最低保证金比例规则', 'issuer': '中证融资服务中心', 'as_of': '2025-12-31'}

### regulation_current_exchange

Query: 查询东衡证券交易所当前有效的盘后固定价格交易细则

Expected plan:

```json
{
  "expected_tools": [
    "search_regulatory_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "当前有效的盘后固定价格交易细则",
        "issuer": "东衡证券交易所"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "盘后固定价格交易细则",
        "issuer": null,
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `False` / `True` / `True`
- reasons: one or more required values or non-default arguments differ

  - argument `search_regulatory_knowledge` (incorrect): required value or non-default argument differs; expected={'query': '当前有效的盘后固定价格交易细则', 'issuer': '东衡证券交易所'}; actual={'query': '盘后固定价格交易细则', 'issuer': None, 'as_of': None}

### regulation_historical_company

Query: 检索澄海智造截至 2025-07-20 的业绩预告公告

Expected plan:

```json
{
  "expected_tools": [
    "search_regulatory_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "截至 2025-07-20 的业绩预告公告",
        "issuer": "澄海智造",
        "as_of": "2025-07-20"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "as_of": "2025-07-20"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "澄海智造业绩预告公告",
        "companies": [
          "澄海智造"
        ],
        "brokers": [],
        "as_of": "2025-07-20"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `False` / `False` / `True`
- reasons: tool selection differs from selected expected plan; one or more required values or non-default arguments differ; temporal expectation differs; 1 unnecessary Tool call(s)

  - argument `search_regulatory_knowledge` (incorrect): missing matching Tool instance; expected={'query': '截至 2025-07-20 的业绩预告公告', 'issuer': '澄海智造', 'as_of': '2025-07-20'}; actual=None

### faq_account

Query: 开户需要哪些基本条件？

Expected plan:

```json
{
  "expected_tools": [
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "开户需要哪些基本条件？",
        "category": "account"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "开户需要哪些基本条件？",
        "category": "account",
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### faq_trading_historical

Query: 截至 2025-12-31，卖出股票后资金何时可以转出？

Expected plan:

```json
{
  "expected_tools": [
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "卖出股票后资金何时可以转出？",
        "category": "trading",
        "as_of": "2025-12-31"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "as_of": "2025-12-31"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `False` / `False` / `True`
- reasons: tool selection differs from selected expected plan; one or more required values or non-default arguments differ; temporal expectation differs

  - argument `search_business_knowledge` (incorrect): missing matching Tool instance; expected={'query': '卖出股票后资金何时可以转出？', 'category': 'trading', 'as_of': '2025-12-31'}; actual=None

### faq_margin

Query: 两融账户维持担保比例低于预警线怎么办？

Expected plan:

```json
{
  "expected_tools": [
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "两融账户维持担保比例低于预警线怎么办？",
        "category": "margin"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `False` / `True` / `True`
- reasons: tool selection differs from selected expected plan; one or more required values or non-default arguments differ

  - argument `search_business_knowledge` (incorrect): missing matching Tool instance; expected={'query': '两融账户维持担保比例低于预警线怎么办？', 'category': 'margin'}; actual=None

### faq_product_feature

Query: 新版条件单支持哪些有效期？

Expected plan:

```json
{
  "expected_tools": [
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "新版条件单支持哪些有效期？",
        "category": "product_feature"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "新版条件单支持哪些有效期？",
        "category": "product_feature",
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### faq_risk_historical

Query: 截至 2025-06-30，融资交易为什么会放大亏损？

Expected plan:

```json
{
  "expected_tools": [
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "融资交易为什么会放大亏损？",
        "category": "risk_notice",
        "as_of": "2025-06-30"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "as_of": "2025-06-30"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `False` / `False` / `True`
- reasons: tool selection differs from selected expected plan; one or more required values or non-default arguments differ; temporal expectation differs

  - argument `search_business_knowledge` (incorrect): missing matching Tool instance; expected={'query': '融资交易为什么会放大亏损？', 'category': 'risk_notice', 'as_of': '2025-06-30'}; actual=None

### user_market_parallel

Query: 并行查询 syn-user-0001 的持仓和 600519.SH 当前行情

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_positions",
    "get_market_snapshot"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    },
    {
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### user_market_rag_parallel

Query: 同时获取 syn-user-0002 的组合分析、000001.SZ 当前行情和安岳银行研报

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_analytics",
    "get_market_snapshot",
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_analytics",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    },
    {
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "000001.SZ"
      }
    },
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "安岳银行研报",
        "companies": [
          "安岳银行"
        ]
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_portfolio_analytics",
      "arguments": {
        "user_id": "syn-user-0002"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "000001.SZ"
      },
      "dependencies": []
    },
    {
      "task_id": "t3",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "安岳银行研报",
        "companies": [
          "安岳银行"
        ],
        "brokers": [],
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### profile_then_faq_dependency

Query: 先查询 syn-user-0001 的客户资料，再查询风险等级不匹配时能否购买产品的说明

Expected plan:

```json
{
  "expected_tools": [
    "get_customer_context",
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    },
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "风险等级不匹配时能否购买产品的说明",
        "category": "risk_notice"
      }
    }
  ],
  "expected_dependencies": [
    {
      "upstream_tool": "get_customer_context",
      "downstream_tool": "search_business_knowledge"
    }
  ],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "风险等级不匹配时能否购买产品的说明",
        "category": "risk_notice",
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `False`
- reasons: dependency edges differ


### margin_and_regulation_parallel

Query: 并行查询 syn-user-0003 的两融账户与当前最低保证金比例规则

Expected plan:

```json
{
  "expected_tools": [
    "get_margin_account",
    "search_regulatory_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0003"
      }
    },
    {
      "tool_name": "search_regulatory_knowledge",
      "arguments": {
        "query": "当前最低保证金比例规则",
        "issuer": "中证融资服务中心"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_margin_account",
      "arguments": {
        "user_id": "syn-user-0003",
        "start_date": null,
        "end_date": null,
        "limit": 60,
        "offset": 0
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "当前最低保证金比例规则",
        "category": "margin",
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `False` / `True` / `True`
- reasons: tool selection differs from selected expected plan; one or more required values or non-default arguments differ; 1 unnecessary Tool call(s)

  - argument `search_regulatory_knowledge` (incorrect): missing matching Tool instance; expected={'query': '当前最低保证金比例规则', 'issuer': '中证融资服务中心'}; actual=None

### positions_then_research_dependency

Query: 先查 syn-user-0001 的持仓，再查看截至 2026-06-30 的澄海智造研报

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_positions",
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    },
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "截至 2026-06-30 的澄海智造研报",
        "companies": [
          "澄海智造"
        ],
        "as_of": "2026-06-30"
      }
    }
  ],
  "expected_dependencies": [
    {
      "upstream_tool": "get_portfolio_positions",
      "downstream_tool": "search_research_reports"
    }
  ],
  "temporal_expectation": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "as_of": "2026-06-30"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "task_1",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "task_2",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "澄海智造研报",
        "companies": [
          "澄海智造"
        ],
        "brokers": [],
        "as_of": "2026-06-30"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `False`
- reasons: dependency edges differ

  - argument `search_research_reports` (acceptable): query wording is semantically equivalent; expected={'query': '截至 2026-06-30 的澄海智造研报', 'companies': ['澄海智造'], 'as_of': '2026-06-30'}; actual={'query': '澄海智造研报', 'companies': ['澄海智造'], 'brokers': [], 'as_of': '2026-06-30'}

### three_source_parallel

Query: 并行查询 syn-user-0001 的持仓、600519.SH 当前行情和截至 2026-05-01 的澄海智造研报

Expected plan:

```json
{
  "expected_tools": [
    "get_portfolio_positions",
    "get_market_snapshot",
    "search_research_reports"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    },
    {
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      }
    },
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "截至 2026-05-01 的澄海智造研报",
        "companies": [
          "澄海智造"
        ],
        "as_of": "2026-05-01"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": [
    {
      "tool_name": "search_research_reports",
      "arguments": {
        "as_of": "2026-05-01"
      }
    }
  ]
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "get_market_snapshot",
      "arguments": {
        "symbol": "600519.SH"
      },
      "dependencies": []
    },
    {
      "task_id": "t3",
      "tool_name": "search_research_reports",
      "arguments": {
        "query": "澄海智造研报",
        "companies": [
          "澄海智造"
        ],
        "brokers": [],
        "as_of": "2026-05-01"
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct

  - argument `search_research_reports` (acceptable): query wording is semantically equivalent; expected={'query': '截至 2026-05-01 的澄海智造研报', 'companies': ['澄海智造'], 'as_of': '2026-05-01'}; actual={'query': '澄海智造研报', 'companies': ['澄海智造'], 'brokers': [], 'as_of': '2026-05-01'}

### faq_only_forbidden_market

Query: 只解释为什么市价委托不一定按显示价格成交，不要查行情

Expected plan:

```json
{
  "expected_tools": [
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "为什么市价委托不一定按显示价格成交",
        "category": "trading"
      }
    }
  ],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `False` / `False` / `True` / `True`
- reasons: tool selection differs from selected expected plan; one or more required values or non-default arguments differ

  - argument `search_business_knowledge` (incorrect): missing matching Tool instance; expected={'query': '为什么市价委托不一定按显示价格成交', 'category': 'trading'}; actual=None

### missing_symbol_abstain

Query: 查询当前股价

Expected plan:

```json
{
  "expected_tools": [],
  "expected_arguments": [],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### conflicting_user_history_abstain

Query: 查询他的组合分析

Expected plan:

```json
{
  "expected_tools": [],
  "expected_arguments": [],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### underspecified_analysis_abstain

Query: 帮我分析一下最近表现

Expected plan:

```json
{
  "expected_tools": [],
  "expected_arguments": [],
  "expected_dependencies": [],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "clarify",
  "tasks": []
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `True`
- reasons: all scored dimensions correct


### dynamic_context_dependency

Query: 先查询 syn-user-0002 的客户资料，再给出融资交易风险提示；不要把上游结果写入参数

Expected plan:

```json
{
  "expected_tools": [
    "get_customer_context",
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0002"
      }
    },
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "融资交易风险提示",
        "category": "risk_notice"
      }
    }
  ],
  "expected_dependencies": [
    {
      "upstream_tool": "get_customer_context",
      "downstream_tool": "search_business_knowledge"
    }
  ],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0002"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "融资交易风险提示",
        "category": "risk_notice",
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `True` / `True` / `False`
- reasons: dependency edges differ


### acceptable_dependency_variants

Query: 查询 syn-user-0001 的客户资料和持仓后，再说明风险等级不匹配时能否购买产品

Expected plan:

```json
{
  "expected_tools": [
    "get_customer_context",
    "get_portfolio_positions",
    "search_business_knowledge"
  ],
  "expected_arguments": [
    {
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    },
    {
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      }
    },
    {
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "风险等级不匹配时能否购买产品",
        "category": "risk_notice"
      }
    }
  ],
  "expected_dependencies": [
    {
      "upstream_tool": "get_customer_context",
      "downstream_tool": "search_business_knowledge"
    },
    {
      "upstream_tool": "get_portfolio_positions",
      "downstream_tool": "search_business_knowledge"
    }
  ],
  "temporal_expectation": []
}
```

Actual plan:

```json
{
  "decision": "execute",
  "tasks": [
    {
      "task_id": "t1",
      "tool_name": "get_customer_context",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "t2",
      "tool_name": "get_portfolio_positions",
      "arguments": {
        "user_id": "syn-user-0001"
      },
      "dependencies": []
    },
    {
      "task_id": "t3",
      "tool_name": "search_business_knowledge",
      "arguments": {
        "query": "风险等级不匹配时能否购买产品",
        "category": "product_feature",
        "as_of": null
      },
      "dependencies": []
    }
  ]
}
```

- validation: `True`
- tool / argument / temporal / dependency: `True` / `False` / `True` / `False`
- reasons: one or more required values or non-default arguments differ; dependency edges differ

  - argument `search_business_knowledge` (incorrect): required value or non-default argument differs; expected={'query': '风险等级不匹配时能否购买产品', 'category': 'risk_notice'}; actual={'query': '风险等级不匹配时能否购买产品', 'category': 'product_feature', 'as_of': None}
