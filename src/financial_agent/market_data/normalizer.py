"""Normalize user-facing Mainland China stock/index symbols."""

import re

from financial_agent.market_data.models import AssetType, CanonicalSymbol


class InvalidSymbolError(ValueError):
    pass


_CANONICAL = re.compile(r"^(\d{6})\.(SH|SZ)$")
_PREFIXED = re.compile(r"^(SH|SZ)(\d{6})$")


def normalize(symbol: str, asset_type: AssetType) -> CanonicalSymbol:
    """Return a canonical six-digit code with an exchange suffix."""
    if not isinstance(symbol, str):
        raise InvalidSymbolError("Symbol must be a string")
    value = symbol.strip().upper()
    match = _CANONICAL.fullmatch(value)
    if match:
        code, suffix = match.groups()
    else:
        match = _PREFIXED.fullmatch(value)
        if match:
            suffix, code = match.groups()
        elif re.fullmatch(r"\d{6}", value):
            code = value
            suffix = _infer_suffix(code, asset_type)
        else:
            raise InvalidSymbolError("Unsupported symbol format")

    expected = _infer_suffix(code, asset_type)
    if suffix != expected:
        raise InvalidSymbolError("Symbol exchange does not match asset type")
    return f"{code}.{suffix}"


def _infer_suffix(code: str, asset_type: AssetType) -> str:
    if asset_type == "stock":
        if code.startswith(("600", "601", "603", "605", "688")):
            return "SH"
        if code.startswith(("000", "001", "002", "003", "300", "301")):
            return "SZ"
    else:
        if code.startswith("000"):
            return "SH"
        if code.startswith("399"):
            return "SZ"
    raise InvalidSymbolError("Cannot infer exchange for symbol")
