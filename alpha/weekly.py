"""Weekly bars and 30-week moving averages for Weinstein stage analysis.

BOOK-DERIVED (Ch.1 pp.13-14, 25-26): weekly bars (Mon-Fri), 30-week MA = average of the last 30 Friday
closes (investors), 10-week MA (traders), Mansfield plots a WEIGHTED 30-week MA; RS = stock / market average.
RESEARCH-DERIVED: the zero-line normalisation of the RS line (book shows a zero line on Mansfield charts but the
read pages do not define it) -> we use RS / SMA52(RS) - 1 ("Mansfield-style"), flagged as research-derived.

PIT: a weekly value becomes usable at the close of the LAST trading day of that week; mid-week days carry the
previous completed week's value (see `to_daily`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def week_end_index(dates: pd.DatetimeIndex) -> pd.Series:
    """Map each trading day to the last trading day of its (Mon-Fri) week."""
    wk = dates.to_period("W-FRI")
    s = pd.Series(dates, index=dates)
    return s.groupby(wk).transform("max")


def weekly_bars(p) -> dict[str, pd.DataFrame]:
    we = week_end_index(p.dates)
    key = we.to_numpy()
    grp = lambda df, how: getattr(df.groupby(key), how)()
    W = {
        "close": grp(p.c, "last"),
        "high": grp(p.h, "max"),
        "low": grp(p.l, "min"),
        "open": grp(p.o, "first"),
        "volume": grp(p.vol.where(p.c.notna()), "sum").replace(0, np.nan),
        "value": grp(p.val.where(p.c.notna()), "sum").replace(0, np.nan),
    }
    W["close"] = W["close"].where(W["volume"].notna())
    m = p.market["adj_close"].groupby(key).last()
    W["mkt_close"] = m
    for k in W:
        W[k].index = pd.DatetimeIndex(W[k].index)
    return W


def wma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linearly weighted MA (newest weight n), vectorised via shifts."""
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    acc = None
    for i in range(n):                       # i = 0 oldest ... n-1 newest
        term = x.shift(n - 1 - i) * w[i]
        acc = term if acc is None else acc + term
    return acc


def weekly_indicators(W: dict) -> dict[str, pd.DataFrame]:
    c = W["close"]
    out = {}
    out["ma30w"] = c.rolling(30, min_periods=26).mean()
    out["ma10w"] = c.rolling(10, min_periods=9).mean()
    # Mansfield-style weighted MA (book: weighted, reacts faster). Linear weights are an assumption.
    out["wma30w"] = wma(c, 30)
    for k in ("ma30w", "wma30w"):
        out[f"{k}_slope4"] = out[k] / out[k].shift(4) - 1
        out[f"{k}_slope8"] = out[k] / out[k].shift(8) - 1
    rs = c.div(W["mkt_close"], axis=0)
    out["rs_w"] = rs
    out["mrs"] = rs / rs.rolling(52, min_periods=40).mean() - 1        # RESEARCH-DERIVED zero line
    out["rs_slope4"] = rs / rs.shift(4) - 1
    out["vol_avg10w"] = W["volume"].shift(1).rolling(10, min_periods=6).mean()
    out["vol_ratio_w"] = W["volume"] / out["vol_avg10w"]
    return out


def to_daily(x: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekly frame indexed by week-end trading days -> daily frame (value usable from week-end close on)."""
    return x.reindex(x.index.union(dates)).ffill().reindex(dates)
