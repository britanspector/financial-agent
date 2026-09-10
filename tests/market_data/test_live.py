from datetime import date, timedelta

import pytest

from financial_agent.config import Settings
from financial_agent.market_data.runtime import build_market_tools
from financial_agent.user_data.auth import CallContext


def live_registry():
    settings = Settings()
    if settings.tushare_token is None:
        pytest.skip("FINANCIAL_AGENT_TUSHARE_TOKEN is not configured")
    return build_market_tools(settings)


@pytest.mark.live
def test_live_tushare_daily_close_snapshot():
    result = live_registry().invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert result.status == "success"
    assert result.data.source == "tushare"
    assert result.data.snapshot_kind == "daily_close"


@pytest.mark.live
def test_live_tushare_unadjusted_stock_history():
    end = date.today()
    result = live_registry().invoke("get_market_history", {
        "symbol": "600519.SH", "start_date": end - timedelta(days=30), "end_date": end,
    }, context=CallContext())
    assert result.status == "success"
    assert result.data.source == "tushare"
    assert result.data.asset_type == "stock"
    assert result.data.adjustment == "none"
    assert result.data.bars
