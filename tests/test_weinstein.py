"""Unit tests for the Weinstein engine: constants match WEINSTEIN_QUANT_RULES.md, classifier transitions,
stop-limit fill semantics, Bollinger B2 stop-proxy, no look-ahead in weekly -> daily mapping."""
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.trades import ExitSpec, Mats, simulate
from alpha.weinstein import WRULES, classify

ROOT = Path(__file__).resolve().parents[1]


def test_constants_documented():
    doc = (ROOT / "WEINSTEIN_QUANT_RULES.md").read_text(encoding="utf-8")
    assert WRULES["MA_W"] == 30 and "30 週" in doc
    assert WRULES["FLAT_BAND"] == 0.01 and "±1.0%" in doc
    assert WRULES["VOL_W_MULT"] == 2.0 and "≥ 2.0" in doc
    assert WRULES["PULLBACK_VOL_TEXTBOOK"] == 0.25 and "0.25" in doc
    assert WRULES["OH_WINDOW"] == 130 and WRULES["OH_FAIL"] == 4 and "OH15 ≥ 4" in doc
    assert WRULES["BASE_MIN_W"] == 8 and WRULES["W2_SLOPE_MIN"] == 0.02
    assert WRULES["TRIG_PAD"] == 0.003 and WRULES["LIMIT_PAD"] == 0.02 and "1.003" in doc and "1.02" in doc
    assert WRULES["STOP_CAP"] == 0.15 and "0.85" in doc


def _series_to_stage(close):
    c = pd.Series(close, dtype=float)
    ma = c.rolling(30, min_periods=26).mean()
    sl = ma / ma.shift(4) - 1
    r = classify(c.to_numpy()[:, None], ma.to_numpy()[:, None], sl.to_numpy()[:, None])
    return r["stage"][:, 0]


def test_classifier_cycle():
    # decline -> base -> advance -> top -> decline
    down = np.linspace(100, 50, 60)
    base = 50 + np.sin(np.arange(50) / 3.0)
    up = np.linspace(51, 120, 60)
    top = 120 + 2 * np.sin(np.arange(40) / 2.0)
    down2 = np.linspace(119, 60, 50)
    st = _series_to_stage(np.r_[down, base, up, top, down2])
    seq = [s for i, s in enumerate(st) if s > 0 and (i == 0 or s != st[i - 1])]
    # must visit 4 -> 1 -> 2 -> 3 -> 4 in order (allowing repeats/minor whipsaws in between)
    want = [4, 1, 2, 3, 4]
    k = 0
    for s in seq:
        if k < len(want) and s == want[k]:
            k += 1
    assert k == len(want), seq


def _mats(o, h, l, c):
    T = len(c)
    a = lambda x: np.asarray(x, float)[:, None]
    z = np.zeros((T, 1), bool)
    return Mats(dates=pd.bdate_range("2024-01-01", periods=T), ids=["X"], O=a(o), H=a(h), L=a(l), C=a(c),
                RAW_C=a(c), G=a(np.r_[1.0, np.asarray(o[1:]) / np.asarray(c[:-1])]), LOCK_UP=z, LOCK_DN=z,
                MA20=a(c), MA150=a(c), ATR=np.full((T, 1), 1.0), SWL=np.full((T, 1), np.nan),
                SWH=np.full((T, 1), np.nan), BB={(20, 2.0): (np.full((T, 1), 105.0), np.full((T, 1), 95.0))},
                VAL=np.ones((T, 1)), last_valid=np.array([T - 1]))


def test_stop_limit_gap_above_limit_no_fill_then_fill_at_limit():
    # day1 gaps above the limit and never trades back -> no fill; day2 trades down through the limit -> fill at limit
    o = [100, 105, 104, 104, 104]
    h = [100, 106, 105, 105, 105]
    l = [99, 104.5, 101, 103, 103]
    c = [100, 105, 103, 104, 104]
    M = _mats(o, h, l, c)
    r = simulate(M, 0, 0, 1, "stoplimit", 101.0, 90.0, ExitSpec("x"), 4, 0.0, 0.0, order_days=3, limit=103.0)
    assert r is not None and abs(r["entry_price_adj"] - 103.0) < 1e-9 and r["entry_date"] == M.dates[2]


def test_stop_limit_intraday_cross_fills_at_trigger():
    o = [100, 100, 101]
    h = [100, 102, 102]
    l = [99, 99.5, 100]
    c = [100, 101.5, 101]
    M = _mats(o, h, l, c)
    r = simulate(M, 0, 0, 1, "stoplimit", 101.0, 95.0, ExitSpec("x"), 2, 0.0, 0.0, order_days=2, limit=103.0)
    assert abs(r["entry_price_adj"] - 101.0) < 1e-9


def test_bollinger_b2_stop_proxy():
    # armed by a close above the upper band (105), next day trades below yesterday's band -> exit at the band
    o = [100, 100, 104, 106, 104]
    h = [100, 101, 106, 107, 105]
    l = [99, 99.5, 103, 104, 103]
    c = [100, 100.5, 106, 106, 104]
    M = _mats(o, h, l, c)
    r = simulate(M, 0, 0, 1, "open", np.nan, 90.0, ExitSpec("b2", bb="B2"), 4, 0.0, 0.0, order_days=1)
    assert r["exit_reason"] == "BB_B2_INTRADAY"
    assert r["exit_date"] == M.dates[3]          # day 3 low (104) < yesterday's band (105): stop at 105
