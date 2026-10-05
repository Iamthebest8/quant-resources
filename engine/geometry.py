"""HIGH R/R GEOMETRY features (strategy-neutral, PIT).

Everything at date t uses data <= t only. Pivot highs/lows are confirmed `half` days after
they print, so a pivot formed at k is only usable from k + half.

Families
  RISK      structural stop candidates, stop distance % / ATR
  UPSIDE    distance to 60/120/250D/historical high, nearest major resistance, base-height projection
  OVERHEAD  resistance density (days) and supply density (volume-at-price) above price
  RS-LEAD   RS line breaks its 60/120D high before price does
  COMPRESS  ATR5/ATR20, ATR10/ATR60, Bollinger width percentile, Range5/Range20, RV10/RV60
  VOLUME    Vol5/Vol20, Vol10/Vol60, breakout-volume ratio, volume percentile
  EXTEND    distance to MA20/MA60 in ATR, distance to 30-week MA (MA150), days since stage-2 start
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from engine.panel import Panel

EPS = 1e-12


def _wide(arr, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(arr, index=like.index, columns=like.columns)


def pivots(x: np.ndarray, half: int, kind: str = "high") -> np.ndarray:
    """Boolean array: True at k if x[k] is the max (min) of x[k-half : k+half]. Confirmed at k+half."""
    T = len(x)
    out = np.zeros(T, bool)
    if T < 2 * half + 1:
        return out
    w = sliding_window_view(np.where(np.isfinite(x), x, -np.inf if kind == "high" else np.inf), 2 * half + 1)
    ext = w.max(axis=1) if kind == "high" else w.min(axis=1)
    mid = x[half:T - half]
    out[half:T - half] = np.isfinite(mid) & (mid == ext)
    return out


def nearest_resistance(c: np.ndarray, h: np.ndarray, half: int = 10, lookback: int = 750) -> tuple:
    """Distance from close to the nearest CONFIRMED pivot high above price (within lookback days)
    and the count of confirmed pivot highs within +5/+10/+15% above price."""
    T = len(c)
    piv = np.where(pivots(h, half, "high"))[0]
    dist = np.full(T, np.nan)
    n5 = np.zeros(T)
    n10 = np.zeros(T)
    n15 = np.zeros(T)
    if len(piv) == 0:
        return dist, n5, n10, n15
    pv = h[piv]
    known = piv + half                       # first day the pivot is known
    for t in range(T):
        if not np.isfinite(c[t]):
            continue
        m = (known <= t) & (piv >= t - lookback)
        if not m.any():
            dist[t] = np.inf
            continue
        above = pv[m][pv[m] > c[t]]
        if len(above) == 0:
            dist[t] = np.inf                 # "blue sky": no confirmed pivot high above
        else:
            r = above / c[t] - 1
            dist[t] = r.min()
            n5[t], n10[t], n15[t] = (r <= 0.05).sum(), (r <= 0.10).sum(), (r <= 0.15).sum()
    return dist, n5, n10, n15


def supply_density(tp: np.ndarray, vol: np.ndarray, c: np.ndarray, window: int = 250,
                   bands=(0.05, 0.10, 0.15)) -> dict:
    """Share of the past `window` days' volume traded at typical price in (C, C*(1+b)] — volume-at-price
    approximation of overhead supply. Uses days t-window .. t-1."""
    T = len(c)
    out = {b: np.full(T, np.nan) for b in bands}
    days = {b: np.full(T, np.nan) for b in bands}
    if T <= window:
        return {"vol": out, "days": days}
    tpw = sliding_window_view(np.nan_to_num(tp, nan=0.0), window)[:-1]     # windows ending t-1
    vw = sliding_window_view(np.nan_to_num(vol, nan=0.0), window)[:-1]
    cc = c[window:]
    tot = vw.sum(axis=1)
    ratio = tpw / np.where(cc > 0, cc, np.nan)[:, None]
    for b in bands:
        m = (ratio > 1.0) & (ratio <= 1.0 + b)
        out[b][window:] = (vw * m).sum(axis=1) / np.where(tot > 0, tot, np.nan)
        days[b][window:] = m.sum(axis=1) / window
    return {"vol": out, "days": days}


def rolling_rank_last(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Percentile of today's value within its trailing n-day window (PIT)."""
    return x.rolling(n, min_periods=int(n * 0.6)).rank(pct=True)


def compute_geometry(p: Panel, F: dict, log=print) -> dict[str, pd.DataFrame]:
    G: dict[str, pd.DataFrame] = {}
    c, h, l, o = p.c, p.h, p.l, p.o
    vol = p.vol.where(c.notna())
    atr14 = F["atr"]
    # ---------------- 30-week MA equivalent & trend ------------------------------------------
    ma150 = c.rolling(150, min_periods=120).mean()
    G["ma150"] = ma150
    G["ma150_slope20"] = ma150 / ma150.shift(20) - 1
    G["ma150_slope_atr"] = (ma150 - ma150.shift(20)) / (atr14 + EPS)
    G["dist_ma150"] = c / ma150 - 1
    # ---------------- ATR family / compression ------------------------------------------------
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    for n in (5, 10, 20, 60):
        G[f"atr{n}"] = tr.rolling(n, min_periods=max(3, int(n * 0.7))).mean()
    G["atr5_20"] = G["atr5"] / (G["atr20"] + EPS)
    G["atr10_60"] = G["atr10"] / (G["atr60"] + EPS)
    ma20 = c.rolling(20, min_periods=15).mean()
    sd20 = c.rolling(20, min_periods=15).std()
    G["bbw20"] = 4 * sd20 / (ma20 + EPS)
    G["bbw_pct120"] = rolling_rank_last(G["bbw20"], 120)
    rng = lambda n: h.rolling(n, min_periods=int(n * 0.8)).max() - l.rolling(n, min_periods=int(n * 0.8)).min()
    G["range5_20"] = rng(5) / (rng(20) + EPS)
    r = p.r
    G["rv10_60"] = r.rolling(10, min_periods=8).std() / (r.rolling(60, min_periods=40).std() + EPS)
    # ---------------- volume pattern ----------------------------------------------------------
    v5, v10, v20, v60 = (vol.rolling(n, min_periods=int(n * 0.7)).mean() for n in (5, 10, 20, 60))
    G["vol5_20"] = v5 / (v20 + EPS)
    G["vol10_60"] = v10 / (v60 + EPS)
    G["brk_vol"] = vol / (vol.shift(1).rolling(20, min_periods=15).mean() + EPS)
    G["vol_pct250"] = rolling_rank_last(vol, 250)
    nc = p.disp | p.disp.shift(1, fill_value=False)
    for k in ("vol5_20", "vol10_60", "brk_vol", "vol_pct250"):
        G[k] = G[k].mask(nc)                  # NOT_COMPARABLE under disposition
    # ---------------- upside geometry -----------------------------------------------------------
    for n in (60, 120, 250):
        G[f"room_h{n}"] = h.rolling(n, min_periods=int(n * 0.8)).max() / c - 1
    G["room_hist"] = h.cummax() / c - 1
    # ---------------- RS leads price -----------------------------------------------------------
    rsl = F["rsl"]
    for n in (60, 120):
        rs_hi = (rsl >= rsl.rolling(n, min_periods=int(n * 0.8)).max() - 1e-9)
        px_gap = c / c.rolling(n, min_periods=int(n * 0.8)).max() - 1
        G[f"rs_newhigh{n}"] = rs_hi.astype(float).where(c.notna())
        G[f"rs_leads{n}"] = (rs_hi & (px_gap <= -0.03)).astype(float).where(c.notna())
        G[f"rs_leads{n}_10d"] = G[f"rs_leads{n}"].rolling(10, min_periods=5).max()
    # ---------------- trend extension ------------------------------------------------------------
    G["ext_ma20_atr"] = (c - ma20) / (atr14 + EPS)
    G["ext_ma60_atr"] = (c - c.rolling(60, min_periods=45).mean()) / (atr14 + EPS)
    above = (c > ma150) & (G["ma150_slope20"] > 0)
    cs = above.astype(int).cumsum()
    G["stage2_age"] = (cs - cs.where(~above).ffill().fillna(0)).where(c.notna())   # consecutive days price>rising MA150
    start_px = c.where(above & ~above.shift(1, fill_value=False)).ffill()
    G["ret_since_stage2"] = (c / start_px - 1).where(above)
    # ---------------- per-stock loops: resistance pivots & supply density ---------------------------
    tp = (h + l + c) / 3
    keys = ["res_dist", "res_n5", "res_n10", "res_n15", "supply5", "supply10", "supply15",
            "resdays5", "resdays10", "resdays15", "swing_low10", "swing_low20", "swing_high10", "swing_high20"]
    out = {k: np.full(c.shape, np.nan, dtype=np.float32) for k in keys}
    C, H, L, TP, V = c.to_numpy(), h.to_numpy(), l.to_numpy(), tp.to_numpy(), vol.to_numpy()
    for j in range(C.shape[1]):
        cj = C[:, j]
        if np.isfinite(cj).sum() < 260:
            continue
        d, n5, n10, n15 = nearest_resistance(cj, H[:, j])
        out["res_dist"][:, j], out["res_n5"][:, j], out["res_n10"][:, j], out["res_n15"][:, j] = d, n5, n10, n15
        sd = supply_density(TP[:, j], V[:, j], cj)
        for b, k in ((0.05, "5"), (0.10, "10"), (0.15, "15")):
            out["supply" + k][:, j] = sd["vol"][b]
            out["resdays" + k][:, j] = sd["days"][b]
        # most recent CONFIRMED swing low (pivot low with 5-day half window), PIT
        for half, key, arr, kind in ((5, "swing_low10", L, "low"), (10, "swing_low20", L, "low"),
                                     (5, "swing_high10", H, "high"), (10, "swing_high20", H, "high")):
            pl = pivots(arr[:, j], half, kind)
            val = np.where(pl, arr[:, j], np.nan)
            out[key][:, j] = pd.Series(val).shift(half).ffill().to_numpy()     # known `half` days later
        if (j + 1) % 300 == 0:
            log(f"[geometry] {j + 1}/{C.shape[1]} stocks")
    for k, a in out.items():
        G[k] = _wide(a, c)
    G["res_dist_capped"] = G["res_dist"].clip(upper=1.0)     # inf (blue sky) -> 100%
    G["blue_sky"] = (G["res_dist"] == np.inf).astype(float).where(c.notna())
    for k, v in list(G.items()):
        if isinstance(v, pd.DataFrame) and v.dtypes.iloc[0] == np.float64:
            G[k] = v.astype(np.float32)
    log(f"[geometry] {len(G)} geometry frames")
    return G
