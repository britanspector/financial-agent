import json
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
import pytest

from financial_agent.market_data.models import MarketHistoryInput
from financial_agent.market_data.normalizer import InvalidSymbolError, normalize
from financial_agent.market_data.provider import EmptyResultError, ProviderTimeoutError
from financial_agent.market_data.runtime import register_market_tools
from financial_agent.market_data.service import MarketDataService
from financial_agent.market_data.tushare_provider import TushareProvider
from financial_agent.user_data.auth import CallContext


FIELDS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]


def response(items, *, code=0, msg=None, fields=FIELDS):
    return {"code": code, "msg": msg, "data": {"fields": fields, "items": items}}


def provider(handler, *, token="unit-test-secret", clock=None):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TushareProvider(token, client=client, clock=clock)


@pytest.mark.parametrize("value,asset_type,expected", [
    ("600519.SH", "stock", "600519.SH"),
    ("sh600519", "stock", "600519.SH"),
    ("000001", "stock", "000001.SZ"),
    ("000001", "index", "000001.SH"),
])
def test_symbol_normalization_uses_asset_type(value, asset_type, expected):
    assert normalize(value, asset_type) == expected


@pytest.mark.parametrize("value,asset_type", [("000001.SH", "stock"), ("600519.SZ", "stock"), ("bad", "stock")])
def test_invalid_symbols_are_rejected(value, asset_type):
    with pytest.raises(InvalidSymbolError):
        normalize(value, asset_type)


def test_snapshot_posts_rest_payload_and_uses_latest_daily_row():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=response([
            ["600519.SH", "20260130", 10, 12, 9, 11, 10, 10, 2, 3],
            ["600519.SH", "20260131", 11, 13, 10, 12, 11, 9.09, 4, 5],
        ]))

    clock = lambda: datetime(2026, 2, 1, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    snapshot = provider(handler, clock=clock).get_snapshot("600519.SH")
    assert captured["api_name"] == "daily"
    assert captured["token"] == "unit-test-secret"
    assert captured["params"] == {"ts_code": "600519.SH", "start_date": "20251103", "end_date": "20260201"}
    assert captured["fields"] == ",".join(FIELDS)
    assert snapshot.snapshot_kind == "daily_close"
    assert snapshot.price == Decimal("12")
    assert snapshot.volume == Decimal("400")
    assert snapshot.amount == Decimal("5000")
    assert snapshot.as_of.hour == 15
    assert snapshot.as_of.utcoffset().total_seconds() == 8 * 3600
    assert snapshot.source == "tushare"
    assert "name" not in snapshot.model_dump()


def test_history_is_half_open_sorted_and_converts_units():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content)["params"])
        return httpx.Response(200, json=response([
            ["000001.SZ", "20240201", 12, 13, 11, 12, 11, 9, 9, 9],
            ["000001.SZ", "20240104", 11, 12, 10, 11, 10, 10, 2.5, 3.5],
            ["000001.SZ", "20240103", 10, 11, 9, 10, 9, 11.11, 1.5, 2.5],
        ]))

    history = provider(handler).get_history("000001.SZ", date(2024, 1, 1), date(2024, 2, 1))
    assert captured["end_date"] == "20240131"
    assert [bar.trade_date for bar in history.bars] == [date(2024, 1, 3), date(2024, 1, 4)]
    assert history.bars[0].previous_close == Decimal("9")
    assert history.bars[0].volume == Decimal("150")
    assert history.bars[0].amount == Decimal("2500")
    assert history.asset_type == "stock"
    assert history.adjustment == "none"


@pytest.mark.parametrize("body,code", [
    ({"code": 2002, "msg": "permission denied", "data": None}, "PROVIDER_PERMISSION_DENIED"),
    ({"code": 9999, "msg": "unknown", "data": None}, "PROVIDER_ERROR"),
])
def test_provider_codes_are_mapped_strictly(body, code):
    registry = register_market_tools(MarketDataService(provider(lambda request: httpx.Response(200, json=body))))
    result = registry.invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert result.error.code == code


def test_provider_message_redacts_token():
    secret = "a-secret-that-must-not-leak"
    market = provider(
        lambda request: httpx.Response(200, json={"code": 9, "msg": f"bad {secret}", "data": None}), token=secret,
    )
    with pytest.raises(Exception) as caught:
        market.get_snapshot("600519.SH")
    assert secret not in str(caught.value)


@pytest.mark.parametrize("body", [
    {"code": 0, "msg": None, "data": {"fields": FIELDS, "items": [["too-short"]]}},
    {"code": 0, "msg": None, "data": {"fields": ["trade_date"], "items": []}},
    {"code": 0, "msg": None, "data": None},
])
def test_malformed_success_response_becomes_provider_error(body):
    registry = register_market_tools(MarketDataService(provider(lambda request: httpx.Response(200, json=body))))
    result = registry.invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert result.error.code == "PROVIDER_ERROR"
    assert result.error.http_status == 502


def test_invalid_json_becomes_provider_error():
    registry = register_market_tools(MarketDataService(provider(lambda request: httpx.Response(200, content=b"not-json"))))
    result = registry.invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert result.error.code == "PROVIDER_ERROR"


@pytest.mark.parametrize("exception,code,retryable", [
    (httpx.ReadTimeout("slow"), "PROVIDER_TIMEOUT", True),
    (httpx.ConnectError("offline"), "PROVIDER_UNAVAILABLE", True),
])
def test_transport_errors_are_mapped_without_retry(exception, code, retryable):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise exception

    registry = register_market_tools(MarketDataService(provider(handler)))
    result = registry.invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert result.error.code == code
    assert result.error.retryable is retryable
    assert calls == 1


def test_empty_result_and_old_history_contract_are_rejected():
    registry = register_market_tools(MarketDataService(provider(lambda request: httpx.Response(200, json=response([])))))
    empty = registry.invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert empty.error.code == "EMPTY_RESULT"
    assert empty.error.http_status == 404
    invalid = registry.invoke("get_market_history", {
        "symbol": "600519.SH", "start_date": "2024-01-01", "end_date": "2024-02-01",
        "asset_type": "stock", "adjustment": "none",
    }, context=CallContext())
    assert invalid.error.code == "INVALID_ARGUMENT"
    assert set(MarketHistoryInput.model_json_schema()["properties"]) == {"symbol", "start_date", "end_date"}


def test_service_maps_typed_provider_errors():
    class Provider:
        def get_snapshot(self, symbol):
            raise ProviderTimeoutError()

        def get_history(self, *args):
            raise EmptyResultError()

    registry = register_market_tools(MarketDataService(Provider()))
    result = registry.invoke("get_market_snapshot", {"symbol": "600519.SH"}, context=CallContext())
    assert result.error.code == "PROVIDER_TIMEOUT"
    assert result.error.retryable is True
