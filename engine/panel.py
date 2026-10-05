"""Raw FinMind cache -> clean PIT panel (wide matrices dates x stocks).

Key decisions (documented in FINMIND_DATA_AUDIT.md):
* Adjustment: FinMind `spread` = close - reference price (exchange-published reference
  already reflects ex-dividend / ex-rights / capital reduction). Daily total return
  r_t = close_t / (close_t - spread_t) - 1. Adjusted prices are rebuilt by compounding
  r_t, so a corporate-action gap is never mistaken for a price move. Ratios of adjusted
  prices are identical to ratios of a PIT series -> no look-ahead from adjustment.
* No-trade days (volume 0 / price 0) -> NaN (not tradable).
* Universe is decided each day only with data up to that day (listed days, price,
  20D traded value, common stock, no disposition for entries).
* Sector = FinMind industry_category (current snapshot; reclassification history is
  not available -> PIT_RISK noted). Delisted names absent from TaiwanStockInfo -> UNKNOWN.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from data.download import NON_EQUITY_INDUSTRIES, is_common_stock_id  # noqa: E402
from data.finmind_client import FinMindCache  # noqa: E402

GENERIC_SECTORS = {"電子工業", "其他", "其他業", "UNKNOWN"}
SECTOR_ALIASES = {"其他電子類": "其他電子業", "觀光事業": "觀光餐旅", "金融業": "金融保險", "金融保險業": "金融保險",
                  "建材營造業": "建材營造", "數位雲端類": "數位雲端", "綠能環保類": "綠能環保", "居家生活類": "居家生活",
                  "運動休閒類": "運動休閒", "其他業": "其他"}


def normalize_sector(s: str) -> str:
    s = str(s).strip()
    return SECTOR_ALIASES.get(s, s)


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    ids: list[str]
    o: pd.DataFrame          # adjusted OHLC
    h: pd.DataFrame
    l: pd.DataFrame
    c: pd.DataFrame
    raw_c: pd.DataFrame      # unadjusted close (for display / price filter)
    raw_o: pd.DataFrame
    r: pd.DataFrame          # daily total return (spread based)
    vol: pd.DataFrame        # shares
    val: pd.DataFrame        # traded value TWD
    trades: pd.DataFrame
    listed_days: pd.DataFrame
    universe: pd.DataFrame   # bool
    disp: pd.DataFrame       # bool: under disposition today
    disp_next: pd.DataFrame  # bool: disposition covers next trading day (known today)
    disp_minutes: pd.DataFrame
    market: pd.DataFrame     # TAIEX OHLC + ret
    market2: pd.DataFrame    # TPEx OHLC + ret (may be empty)
    sector: pd.Series        # stock_id -> sector
    names: pd.Series
    stock_type: pd.Series
    delisted: pd.Series      # stock_id -> delist date
    mktval: pd.DataFrame | None = None   # market value (month-end snapshots, forward-filled = PIT)
    meta: dict = field(default_factory=dict)


def _load_prices(cache: FinMindCache) -> pd.DataFrame:
    d = cache.raw_dir / "TaiwanStockPrice"
    files = [f for f in sorted(d.glob("*.parquet")) if not f.stem.startswith("IDX_")]
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        frames.append(df[df["stock_id"].map(is_common_stock_id)])
    if not frames:
        raise RuntimeError(f"本地快取沒有 TaiwanStockPrice 資料: {d}. 請先執行 python -m data.download")
    px = pd.concat(frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    return px.drop_duplicates(["date", "stock_id"], keep="last")


def _index_frame(cache: FinMindCache, key: str) -> pd.DataFrame:
    df = cache.read("TaiwanStockPrice", key)
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.rename(columns={"max": "high", "min": "low"})
    df = df[(df["close"] > 0)]
    ref = df["close"] - df["spread"] if "spread" in df.columns else df["close"].shift(1)
    raw_r = df["close"] / df["close"].shift(1) - 1
    r = df["close"] / ref - 1
    r = r.where(r.abs() < 0.2, raw_r)
    df["ret"] = r.fillna(0.0)
    df["adj_close"] = (1 + df["ret"]).cumprod() * df["close"].iloc[0]
    f = df["adj_close"] / df["close"]
    for col in ("open", "high", "low"):
        df["adj_" + col] = df[col] * f
    return df[["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "ret"]]


def _market_frames(cache: FinMindCache) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    taiex = _index_frame(cache, "IDX_TAIEX")
    src = "TaiwanStockPrice:TAIEX (OHLC)"
    if taiex.empty:
        taiex = _index_frame(cache, "IDX_0050")
        src = "FALLBACK 0050 ETF OHLC (TAIEX OHLC 不可得)"
    if taiex.empty:
        tri = cache.read("TaiwanStockTotalReturnIndex", "TAIEX")
        if not tri.empty:
            tri["date"] = pd.to_datetime(tri["date"])
            tri = tri.sort_values("date").set_index("date")
            p = tri["price"].astype(float)
            taiex = pd.DataFrame({"open": p.shift(1), "high": p, "low": p, "close": p})
            taiex["ret"] = p.pct_change().fillna(0)
            for col in ("open", "high", "low", "close"):
                taiex["adj_" + col] = taiex[col]
            src = "FALLBACK TotalReturnIndex (無 OHLC，open→close 特徵不可用)"
    tpex = _index_frame(cache, "IDX_TPEx")
    if tpex.empty:
        tpex = _index_frame(cache, "IDX_006201")
    if taiex.empty:
        raise RuntimeError("找不到任何大盤資料 (TAIEX / 0050 / TotalReturnIndex)")
    return taiex, tpex, src


def _sector_map(info: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    if info.empty:
        return pd.Series(dtype=str), pd.Series(dtype=str), pd.Series(dtype=str)
    df = info[info["stock_id"].map(is_common_stock_id)].copy()
    df = df[~df["industry_category"].isin(NON_EQUITY_INDUSTRIES)]
    df["industry_category"] = df["industry_category"].map(normalize_sector)
    # a stock can appear under several categories -> prefer the most specific one
    df["generic"] = df["industry_category"].isin(GENERIC_SECTORS)
    df = df.sort_values(["stock_id", "generic"])
    first = df.groupby("stock_id").first()
    # market type: prefer the current listing row (twse/tpex) over an older 興櫃 (emerging) row.
    # NOTE: the industry above is kept exactly as frozen for V1 (most specific category of any row);
    # for 27 stocks this differs from the current-listing row -> documented data issue for V2.
    if "type" in df.columns:
        df["nonmain"] = ~df["type"].isin(["twse", "tpex"])
        typ = df.sort_values(["stock_id", "nonmain"]).groupby("stock_id")["type"].first()
    else:
        typ = pd.Series(dtype=str)
    return first["industry_category"], first["stock_name"], typ


_MIN_PAT = [("六十分鐘", 60), ("一小時", 60), ("二十五分鐘", 25), ("二十分鐘", 20), ("十分鐘", 10), ("五分鐘", 5),
            ("60分鐘", 60), ("25分鐘", 25), ("20分鐘", 20), ("10分鐘", 10), ("5分鐘", 5)]


def parse_matching_minutes(text: str) -> int:
    """Matching interval in minutes. TWSE/TPEx defaults: 1st disposition ~5 min, 2nd ~20 min."""
    t = str(text or "")
    for pat, m in _MIN_PAT:
        if pat in t:
            return m
    m = re.search(r"每\s*(\d+)\s*分", t)
    if m:
        return int(m.group(1))
    if "第二次" in t or "再次" in t:
        return 20
    return 5


def disposition_table(cache: FinMindCache) -> pd.DataFrame:
    dp = cache.read_all("TaiwanStockDispositionSecuritiesPeriod")
    if dp.empty:
        return pd.DataFrame(columns=["announce_date", "stock_id", "start", "end", "minutes", "measure", "condition"])
    dp = dp[dp["stock_id"].map(is_common_stock_id)].copy()
    out = pd.DataFrame({
        "announce_date": pd.to_datetime(dp["date"]),
        "stock_id": dp["stock_id"],
        "start": pd.to_datetime(dp.get("period_start"), errors="coerce"),
        "end": pd.to_datetime(dp.get("period_end"), errors="coerce"),
        "measure": dp.get("measure", ""),
        "condition": dp.get("condition", ""),
    })
    out["minutes"] = out["measure"].map(parse_matching_minutes)
    out["restrictions"] = out["measure"].astype(str).map(
        lambda s: ";".join(k for k in ("預收款券", "人工管制", "分盤") if k in s) or "分盤撮合")
    out["start"] = out["start"].fillna(out["announce_date"] + pd.Timedelta(days=1))
    out["end"] = out["end"].fillna(out["start"] + pd.Timedelta(days=14))
    return out.dropna(subset=["start", "end"]).drop_duplicates()


def _load_adj(cache: FinMindCache, ids: list, dates: pd.DatetimeIndex, end: str | None) -> pd.DataFrame | None:
    d = cache.raw_dir / "TaiwanStockPriceAdj"
    if not d.exists():
        return None
    frames = []
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f)
        if len(df) and {"date", "stock_id", "close", "Trading_Volume"} <= set(df.columns):
            frames.append(df[["date", "stock_id", "close", "Trading_Volume"]])
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    a["date"] = pd.to_datetime(a["date"])
    if end:
        a = a[a["date"] <= pd.Timestamp(end)]
    a = a[(a["close"] > 0) & (a["Trading_Volume"] > 0)].drop_duplicates(["date", "stock_id"], keep="last")
    w = a.pivot(index="date", columns="stock_id", values="close")
    return w.reindex(index=dates, columns=ids)


def build_panel(cache: FinMindCache | None = None, end: str | None = None, log=print) -> Panel:
    cache = cache or FinMindCache(client=None)
    px = _load_prices(cache)
    if end:
        px = px[px["date"] <= pd.Timestamp(end)]
    taiex, tpex, mkt_src = _market_frames(cache)
    if end:
        taiex = taiex[taiex.index <= pd.Timestamp(end)]
        tpex = tpex[tpex.index <= pd.Timestamp(end)] if not tpex.empty else tpex
    dates = pd.DatetimeIndex(sorted(set(taiex.index) | set(px["date"].unique())))
    dates = dates[dates >= pd.Timestamp(config.DATA_START)]
    taiex = taiex.reindex(dates)

    px = px.rename(columns={"max": "high", "min": "low"})
    bad = (px["Trading_Volume"] <= 0) | (px["close"] <= 0) | px["close"].isna()
    for col in ("open", "high", "low", "close"):
        px.loc[bad, col] = np.nan

    def wide(col):
        return px.pivot(index="date", columns="stock_id", values=col).reindex(dates)

    raw_o, raw_h, raw_l, raw_c = wide("open"), wide("high"), wide("low"), wide("close")
    spread = wide("spread")
    vol, val, trades = wide("Trading_Volume"), wide("Trading_money"), wide("Trading_turnover")
    ids = list(raw_c.columns)

    # --- daily total return ------------------------------------------------------------
    # 1) FinMind TaiwanStockPriceAdj close-to-close ratio (correct on ex-dividend days)
    # 2) fallback: exchange reference price from `spread` (close - spread); FinMind sets spread=0 on
    #    ex-rights/dividend days, so spread==0 with a changed close is treated as "unknown"
    # 3) fallback: raw close-to-close when within the +-10% price limit
    prev_c = raw_c.ffill().shift(1)
    raw_r = raw_c / prev_c - 1
    ref = raw_c - spread
    r_ref = raw_c / ref - 1
    ok = r_ref.notna() & raw_r.notna() & (raw_r.abs() > 0.003) & (raw_r.abs() < 0.095) & (spread != 0)
    agree = (np.sign(r_ref[ok]) == np.sign(raw_r[ok])).sum().sum() / max(ok.sum().sum(), 1)
    if agree < 0.9:
        log(f"[panel] spread 似乎無正負號 (agree={agree:.2f}) -> 使用 raw move 的方向")
        r_ref = np.sign(raw_r) * r_ref.abs()
    lim = config.PRICE_LIMIT + 0.006
    ld0 = raw_c.notna().cumsum()
    no_limit = ld0 <= 6          # newly listed: no daily price limit for the first 5 sessions
    spread_unknown = (spread == 0) & (raw_c != prev_c)
    r_spread = r_ref.where((r_ref.abs() <= lim) & ref.gt(0) & ~spread_unknown)
    adj_px = _load_adj(cache, ids, dates, end)
    if adj_px is not None:
        adj_px = adj_px.where(raw_c.notna())
        r_adj = adj_px / adj_px.ffill().shift(1) - 1
        r_adj = r_adj.where((r_adj.abs() <= lim) | (no_limit & (r_adj.abs() < 3)))
    else:
        r_adj = pd.DataFrame(np.nan, index=dates, columns=ids)
    r = r_adj.copy()
    src_adj = r.notna() & raw_c.notna()
    r = r.fillna(r_spread)
    src_spread = r.notna() & ~src_adj & raw_c.notna()
    r = r.fillna(raw_r.where((raw_r.abs() <= lim) | (no_limit & (raw_r.abs() < 3))))
    src_raw = r.notna() & ~src_adj & ~src_spread & raw_c.notna()
    n_corp = int((spread_unknown & raw_c.notna()).sum().sum())
    unresolved = int((r.isna() & raw_c.notna() & prev_c.notna()).sum().sum())
    r = r.where(raw_c.notna())
    first_valid = raw_c.notna().cumsum() == 1
    r = r.mask(first_valid, 0.0)

    growth = (1 + r.fillna(0)).cumprod()
    base = raw_c.bfill().iloc[0]
    adj_c = (growth * base).where(raw_c.notna())
    f = adj_c / raw_c
    o, h, l = raw_o * f, raw_h * f, raw_l * f

    listed_days = raw_c.notna().cumsum()
    val20 = val.where(raw_c.notna()).rolling(20, min_periods=15).mean()

    sector, names, stype = _sector_map(cache.read("TaiwanStockInfo", "_all"))
    sector = sector.reindex(ids).fillna("UNKNOWN")
    names = names.reindex(ids)
    stype = stype.reindex(ids)
    delist = cache.read("TaiwanStockDelisting", "_all")
    delisted = pd.Series(dtype="datetime64[ns]")
    if not delist.empty:
        delist = delist[delist["stock_id"].isin(ids)]
        delisted = pd.to_datetime(delist.set_index("stock_id")["date"])
        delisted = delisted[~delisted.index.duplicated()]
        names = names.fillna(delist.set_index("stock_id")["stock_name"].reindex(ids)
                             if "stock_name" in delist.columns else names)
    names = names.fillna(pd.Series(ids, index=ids))

    # --- disposition --------------------------------------------------------------
    dtab = disposition_table(cache)
    disp = pd.DataFrame(False, index=dates, columns=ids)
    disp_min = pd.DataFrame(0, index=dates, columns=ids, dtype=np.int16)
    for row in dtab.itertuples():
        if row.stock_id not in disp.columns:
            continue
        m = (dates >= row.start) & (dates <= row.end)
        disp.loc[m, row.stock_id] = True
        disp_min.loc[m, row.stock_id] = np.maximum(disp_min.loc[m, row.stock_id].to_numpy(), row.minutes)
    # "disposition covers the next trading day" must be decidable at t without seeing t+1:
    # known once announced, from shortly before the period starts until the day before it ends.
    disp_next = pd.DataFrame(False, index=dates, columns=ids)
    for row in dtab.itertuples():
        if row.stock_id in disp_next.columns:
            m = (dates >= row.announce_date) & (dates < row.end) & (dates >= row.start - pd.Timedelta(days=7))
            disp_next.loc[m, row.stock_id] = True

    universe = (raw_c.notna() & (listed_days >= config.MIN_LISTED_DAYS) & (raw_c >= config.MIN_PRICE)
                & (val20 >= config.MIN_AVG_VALUE_20D))

    mv = cache.read_all("TaiwanStockMarketValue")
    mktval = None
    if not mv.empty:
        mv["date"] = pd.to_datetime(mv["date"])
        mv = mv[mv["stock_id"].isin(ids)].drop_duplicates(["date", "stock_id"], keep="last")
        mw = mv.pivot(index="date", columns="stock_id", values="market_value")
        mktval = mw.reindex(dates.union(mw.index)).ffill(limit=70).reindex(dates).reindex(columns=ids)

    market = taiex.copy()
    p = Panel(dates=dates, ids=ids, o=o, h=h, l=l, c=adj_c, raw_c=raw_c, raw_o=raw_o, r=r, vol=vol, val=val,
              trades=trades, listed_days=listed_days, universe=universe, disp=disp, disp_next=disp_next,
              disp_minutes=disp_min, market=market, market2=tpex.reindex(dates) if not tpex.empty else tpex,
              sector=sector, names=names, stock_type=stype, delisted=delisted, mktval=mktval,
              meta={"market_source": mkt_src, "spread_sign_agree": float(agree), "corp_action_gaps": n_corp,
                    "ret_src_adj": int(src_adj.sum().sum()), "ret_src_spread": int(src_spread.sum().sum()),
                    "ret_src_raw": int(src_raw.sum().sum()), "ret_unresolved": unresolved,
                    "n_ids": len(ids), "n_dates": len(dates), "first_date": str(dates[0].date()),
                    "last_date": str(dates[-1].date()), "disposition_events": int(len(dtab))})
    p.meta["disposition_table"] = dtab
    log(f"[panel] {len(ids)} stocks x {len(dates)} dates; market={mkt_src}; return source adj/spread/raw="
        f"{p.meta['ret_src_adj']}/{p.meta['ret_src_spread']}/{p.meta['ret_src_raw']} unresolved={unresolved}")
    return p
