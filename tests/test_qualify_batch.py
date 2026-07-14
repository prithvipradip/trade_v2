"""R13: qualify_contracts_batch must return N results for N inputs.

Installed ib_insync 0.9.86 OMITS failed contracts from qualifyContractsAsync's
return value (the old wrapper comment claimed conId==0 in place — wrong). The
old wrapper mapped over that shorter list, so a partial qualification failure
returned a SHORTER list with no Nones, and the executor's positional
`signal.legs[i]` pairing silently misaligned conId-to-action across combo
legs. These tests pin the N-for-N/None-in-place contract of the wrapper
against 0.9.86's drop-failures behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

from ait.broker.ibkr_client import IBKRClient


def _client(qualify_behavior):
    c = IBKRClient.__new__(IBKRClient)
    c._ib = SimpleNamespace(qualifyContractsAsync=qualify_behavior)

    async def _connected():
        return True
    c.ensure_connected = _connected
    return c


def _contract(strike: float) -> SimpleNamespace:
    return SimpleNamespace(conId=0, strike=strike)


async def test_partial_failure_returns_none_in_place_not_shorter_list():
    legs = [_contract(95.0), _contract(98.0), _contract(102.0), _contract(105.0)]

    async def qualify(*contracts):
        # 0.9.86 behavior: mutate successes in place, DROP failures from the
        # return value. Leg index 1 (the 98 strike) fails.
        out = []
        for i, ct in enumerate(contracts):
            if ct.strike != 98.0:
                ct.conId = 1000 + i
                out.append(ct)
        return out

    result = await _client(qualify).qualify_contracts_batch(legs)
    assert len(result) == 4, "must be N-for-N even when the lib drops failures"
    assert result[0] is legs[0] and result[0].conId == 1000
    assert result[1] is None, "failed leg must be None IN PLACE, not dropped"
    assert result[2] is legs[2] and result[2].conId == 1002
    assert result[3] is legs[3] and result[3].conId == 1003


async def test_all_qualified_passthrough():
    legs = [_contract(95.0), _contract(105.0)]

    async def qualify(*contracts):
        for i, ct in enumerate(contracts):
            ct.conId = 2000 + i
        return list(contracts)

    result = await _client(qualify).qualify_contracts_batch(legs)
    assert result == legs


async def test_total_failure_returns_all_none():
    legs = [_contract(95.0), _contract(105.0)]

    async def qualify(*contracts):
        return []  # nothing qualified, nothing mutated

    result = await _client(qualify).qualify_contracts_batch(legs)
    assert result == [None, None]


async def test_exception_returns_all_none():
    legs = [_contract(95.0), _contract(105.0)]

    async def qualify(*contracts):
        raise RuntimeError("farm down")

    result = await _client(qualify).qualify_contracts_batch(legs)
    assert result == [None, None]
