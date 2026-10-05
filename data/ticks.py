"""Tick data -> 1-second bars (Part 21-23).

FinMind `TaiwanStockPriceTick` (trade prints with microsecond timestamps) is cached per stock-day under
data/cache/raw/TaiwanStockPriceTick/<stock>_<date>.parquet; TAIEX 5-second index under
data/cache/raw/TaiwanVariousIndicators5Seconds/TAIEX5S_<date>.parquet.

Only trade prints are used (price, size, time). TickType (aggressor side) is NOT used — no order book /
order-flow information is assumed. After-hours fixed-price prints (14:30) are dropped.

    python -m data.ticks --list outputs/intraday/tick_requests.csv     # download a request list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.finmind_client import FinMindCache, FinMindClient, FinMindError  # noqa: E402

TICK_DS = "TaiwanStockPriceTick"
IDX_DS = "TaiwanVariousIndicators5Seconds"
SESSION_START = 9 * 3600
SESSION_END = 13 * 3600 + 30 * 60          # 13:30:00 closing auction print
N_SEC = SESSION_END - SESSION_START + 1     # 16201 one-second slots


def tick_key(sid: str, d: str) -> str:
    return f"{sid}_{d}"


def download(requests: pd.DataFrame, log=print, index_too: bool = True) -> dict:
    """requests: DataFrame with columns stock_id, date (YYYY-MM-DD)."""
    cache = FinMindCache(FinMindClient())
    n_new = n_empty = n_err = 0
    req = requests.drop_duplicates(["stock_id", "date"]).sort_values(["date", "stock_id"])
    for i, r in enumerate(req.itertuples()):
        key = tick_key(r.stock_id, r.date)
        if cache.path(TICK_DS, key).exists():
            continue
        try:
            df = cache.client.fetch(TICK_DS, data_id=r.stock_id, start_date=r.date, end_date=r.date, validate=False)
        except FinMindError as e:
            n_err += 1
            log(f"[tick] {key}: {e}")
            continue
        if df.empty:
            n_empty += 1
            df = pd.DataFrame(columns=["date", "stock_id", "deal_price", "volume", "Time"])
        else:
            df = df[["date", "stock_id", "deal_price", "volume", "Time"]]
        cache.write(TICK_DS, key, df)
        n_new += 1
        if n_new % 200 == 0:
            cache.save_manifest()
            log(f"[tick] {i + 1}/{len(req)} new={n_new} empty={n_empty} err={n_err}")
    if index_too:
        for d in sorted(req["date"].unique()):
            key = f"TAIEX5S_{d}"
            if cache.path(IDX_DS, key).exists():
                continue
            try:
                df = cache.client.fetch(IDX_DS, start_date=d, end_date=d, validate=False)
            except FinMindError as e:
                log(f"[idx5s] {d}: {e}")
                continue
            cache.write(IDX_DS, key, df if not df.empty else pd.DataFrame(columns=["date", "TAIEX"]))
    cache.save_manifest()
    return {"new": n_new, "empty": n_empty, "errors": n_err}


def _sec_of_day(t: pd.Series) -> np.ndarray:
    s = t.astype(str).str.slice(0, 8)
    hh = s.str.slice(0, 2).astype(int)
    mm = s.str.slice(3, 5).astype(int)
    ss = s.str.slice(6, 8).astype(int)
    return (hh * 3600 + mm * 60 + ss).to_numpy()


def bars_1s(sid: str, d: str, cache: FinMindCache | None = None) -> pd.DataFrame | None:
    """Full 1-second grid 09:00:00..13:30:00. close is forward-filled; volume 0 when no trade.
    Columns: sec (0..16200), open, high, low, close, volume (lots), n, has_trade."""
    cache = cache or FinMindCache(client=None)
    tk = cache.read(TICK_DS, tick_key(sid, d))
    if tk is None or tk.empty:
        return None
    sec = _sec_of_day(tk["Time"])
    m = (sec >= SESSION_START) & (sec <= SESSION_END)
    if m.sum() == 0:
        return None
    px = tk["deal_price"].to_numpy(float)[m]
    vv = tk["volume"].to_numpy(float)[m]
    s = sec[m] - SESSION_START
    df = pd.DataFrame({"s": s, "p": px, "v": vv})
    g = df.groupby("s")
    b = pd.DataFrame({"open": g["p"].first(), "high": g["p"].max(), "low": g["p"].min(), "close": g["p"].last(),
                      "volume": g["v"].sum(), "n": g["p"].size()})
    full = b.reindex(np.arange(N_SEC))
    full["has_trade"] = full["n"].notna()
    full["close"] = full["close"].ffill()
    for col in ("open", "high", "low"):
        full[col] = full[col].fillna(full["close"])
    full["volume"] = full["volume"].fillna(0.0)
    full["n"] = full["n"].fillna(0)
    full.index.name = "sec"
    return full.reset_index()


def index_1s(d: str, cache: FinMindCache | None = None) -> pd.Series | None:
    """TAIEX on the 1-second grid (5-second prints forward-filled)."""
    cache = cache or FinMindCache(client=None)
    ix = cache.read(IDX_DS, f"TAIEX5S_{d}")
    if ix is None or ix.empty:
        return None
    t = pd.to_datetime(ix["date"].astype(str))
    sec = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy() - SESSION_START
    s = pd.Series(ix["TAIEX"].to_numpy(float), index=sec)
    s = s[(s.index >= 0) & (s.index < N_SEC)]
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(np.arange(N_SEC)).ffill()


def sec_to_time(sec: int) -> str:
    t = SESSION_START + int(sec)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    a = ap.parse_args()
    req = pd.read_csv(a.list, dtype={"stock_id": str})
    print(download(req))


if __name__ == "__main__":
    main()
