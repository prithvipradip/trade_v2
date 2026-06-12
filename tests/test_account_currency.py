"""Currency normalization in IBKRClient.get_account_values (2026-06-12).

A CAD-base paper account reported NetLiquidation only in CAD, which the
risk layer then compared against USD option costs — skewing every limit
~37% generous. get_account_values now converts base-currency totals to USD
via IBKR's ExchangeRate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ait.broker.ibkr_client import IBKRClient


def _av(tag, value, currency):
    return SimpleNamespace(tag=tag, value=str(value), currency=currency)


def _client(rows):
    c = IBKRClient.__new__(IBKRClient)
    c.ensure_connected = AsyncMock(return_value=True)
    c._ib = SimpleNamespace(accountValues=lambda: rows)
    return c


# Real shape observed on the live CAD-base paper account (DUN603821).
CAD_ROWS = [
    _av("ExchangeRate", "1.00", "BASE"),
    _av("ExchangeRate", "1.00", "CAD"),
    _av("ExchangeRate", "1.3975776", "USD"),
    _av("NetLiquidation", "278229.23", "CAD"),
    _av("NetLiquidationByCurrency", "278229.2335", "BASE"),
    _av("NetLiquidationByCurrency", "251536.94", "CAD"),
    _av("NetLiquidationByCurrency", "19106.32", "USD"),
    _av("BuyingPower", "927398.04", "CAD"),
    _av("CashBalance", "278108.7655", "BASE"),
    _av("CashBalance", "251426.00", "CAD"),
    _av("CashBalance", "19106.53", "USD"),
    _av("MaintMarginReq", "0.00", "CAD"),
]


class TestCurrencyNormalization:
    @pytest.mark.asyncio
    async def test_cad_nlv_converted_to_usd(self):
        values = await _client(CAD_ROWS).get_account_values()
        # 278229.23 CAD / 1.3975776 = 199079.49 USD
        assert float(values["NetLiquidation"]) == pytest.approx(199079.49, abs=0.5)

    @pytest.mark.asyncio
    async def test_buying_power_converted(self):
        values = await _client(CAD_ROWS).get_account_values()
        assert float(values["BuyingPower"]) == pytest.approx(927398.04 / 1.3975776, abs=1.0)

    @pytest.mark.asyncio
    async def test_cash_balance_prefers_base_total_not_usd_partial(self):
        # BASE row (278108 total) must win over the USD partial (19106).
        values = await _client(CAD_ROWS).get_account_values()
        assert float(values["CashBalance"]) == pytest.approx(278108.7655 / 1.3975776, abs=1.0)

    @pytest.mark.asyncio
    async def test_usd_base_account_unchanged(self):
        rows = [
            _av("ExchangeRate", "1.00", "BASE"),
            _av("ExchangeRate", "1.00", "USD"),
            _av("NetLiquidation", "199079.49", "USD"),
            _av("BuyingPower", "663598.28", "USD"),
        ]
        values = await _client(rows).get_account_values()
        assert float(values["NetLiquidation"]) == pytest.approx(199079.49, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_exchange_rate_falls_back_safely(self):
        # No ExchangeRate rows: rate defaults to 1.0, USD rows pass through.
        rows = [_av("NetLiquidation", "50000.00", "USD")]
        values = await _client(rows).get_account_values()
        assert float(values["NetLiquidation"]) == pytest.approx(50000.0, abs=0.01)
