"""Intraday execution research on 1-second bars (Parts 21-27).

WHAT to trade comes from daily/weekly alpha (a candidate with a trigger level and a structural stop);
this module only studies WHEN to enter on the candidate day.

For each candidate day we evaluate decision points every `step` seconds. Features at decision second t use
1-second bars <= t only. Entry is at the NEXT second's price (t+1, the close of that second, i.e. the last
trade printed by then) plus slippage. Outcomes after entry: returns at +1/5/15/30/60 min, EOD; MFE/MAE to EOD;
whether the daily structural stop or a micro stop is hit intraday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.ticks import N_SEC, bars_1s, index_1s, sec_to_time

HORIZONS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600}
LAST_DECISION = N_SEC - 1 - 10 * 60          # no new entries in the last 10 minutes (closing auction risk)


def _ret(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[lag:] = x[lag:] / x[:-lag] - 1
    return out


def _roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    cs = np.cumsum(np.r_[0.0, x])
    out = cs[n:] - cs[:-n]
    return np.r_[np.full(n - 1, np.nan), out]


def _roll_max(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(n, min_periods=1).max().to_numpy()


def _roll_min(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).min().to_numpy()


def day_dataset(sid: str, d: str, side: int, trigger: float, stop: float, atr: float, prev_close: float,
                adv_lots: float, next_close: float | None = None, step: int = 15, ctx: dict | None = None,
                cache=None) -> pd.DataFrame | None:
    """Decision-point dataset for one candidate day.

    side +1 long / -1 short; trigger = daily trigger level (breakout / breakdown); stop = daily structural stop;
    all prices are UNADJUSTED (same basis as ticks); adv_lots = average daily volume in lots (1000 shares).
    """
    b = bars_1s(sid, d, cache)
    if b is None or b["has_trade"].sum() < 30:
        return None
    first = int(np.argmax(b["has_trade"].to_numpy()))
    px = b["close"].to_numpy(float).copy()
    hi = b["high"].to_numpy(float).copy()
    lo = b["low"].to_numpy(float).copy()
    vol = b["volume"].to_numpy(float).copy()
    px[:first] = np.nan
    hi[:first] = np.nan
    lo[:first] = np.nan
    mk = index_1s(d, cache)
    mkt = mk.to_numpy(float) if mk is not None else np.full(N_SEC, np.nan)
    open_px = px[first]
    # running session stats (<= t)
    cum_v = np.cumsum(vol)
    cum_pv = np.cumsum(vol * np.nan_to_num(px))
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), np.nan)
    hod = np.fmax.accumulate(np.nan_to_num(hi, nan=-np.inf))
    lod = np.fmin.accumulate(np.nan_to_num(lo, nan=np.inf))
    above_vwap = (px >= vwap).astype(float)
    above_vwap[:first] = np.nan
    t_above = np.nancumsum(np.nan_to_num(above_vwap)) / np.maximum(np.arange(N_SEC) - first + 1, 1)
    # trigger crossing (long: price >= trigger)
    sgn = side
    crossed = (sgn * (px - trigger) >= 0)
    first_cross = int(np.argmax(crossed)) if crossed.any() else -1
    # micro structure on 1-minute bars
    m1h = _roll_max(np.nan_to_num(hi, nan=-np.inf), 60)
    m1l = _roll_min(np.nan_to_num(lo, nan=np.inf), 60)
    m1h_prev = np.r_[np.full(60, np.nan), m1h[:-60]]
    m1l_prev = np.r_[np.full(60, np.nan), m1l[:-60]]
    m1h_prev2 = np.r_[np.full(120, np.nan), m1h[:-120]]
    m1l_prev2 = np.r_[np.full(120, np.nan), m1l[:-120]]
    hh = (m1h > m1h_prev) & (m1h_prev > m1h_prev2)
    hl = (m1l > m1l_prev) & (m1l_prev > m1l_prev2)
    or5_h = np.nanmax(hi[first:first + 300]) if N_SEC > first + 300 else np.nan
    or5_l = np.nanmin(lo[first:first + 300]) if N_SEC > first + 300 else np.nan
    # forward-looking helpers (labels only): suffix extremes and forward-window extremes
    hi_f = np.nan_to_num(hi, nan=-np.inf)
    lo_f = np.nan_to_num(lo, nan=np.inf)
    suf_max = np.fmax.accumulate(hi_f[::-1])[::-1]
    suf_min = np.fmin.accumulate(lo_f[::-1])[::-1]
    fwd_max = {nm: pd.Series(hi_f[::-1]).rolling(hs + 1, min_periods=1).max().to_numpy()[::-1]
               for nm, hs in HORIZONS.items()}
    fwd_min = {nm: pd.Series(lo_f[::-1]).rolling(hs + 1, min_periods=1).min().to_numpy()[::-1]
               for nm, hs in HORIZONS.items()}
    v60 = _roll_sum(vol, 60)
    v300 = _roll_sum(vol, 300)
    exp_per_sec = adv_lots / N_SEC if adv_lots and adv_lots > 0 else np.nan
    rows = []
    start = first                      # first decision = right after the opening print (OPEN policy)
    for t in range(start, LAST_DECISION, step):
        e = t + 1
        if not np.isfinite(px[e]):
            continue
        r = {"sec": t, "time": sec_to_time(t), "since_open_min": (t - first) / 60}
        p = px[t]
        for nm, lag in (("1s", 1), ("5s", 5), ("15s", 15), ("30s", 30), ("1m", 60), ("3m", 180), ("5m", 300)):
            r[f"r_{nm}"] = p / px[t - lag] - 1 if t - lag >= first and np.isfinite(px[t - lag]) else np.nan
        r["accel_1m"] = r["r_1m"] - (px[t - 60] / px[t - 120] - 1 if t - 120 >= first else np.nan)
        seg = px[max(first, t - 300):t + 1]
        dr = np.diff(np.log(seg[np.isfinite(seg)])) if np.isfinite(seg).sum() > 2 else np.array([np.nan])
        r["rv_5m"] = float(np.nanstd(dr) * np.sqrt(300)) if len(dr) > 1 else np.nan
        r["ret_open"] = p / open_px - 1
        r["ret_prevclose"] = p / prev_close - 1 if prev_close > 0 else np.nan
        r["dist_hod"] = p / hod[t] - 1
        r["dist_lod"] = p / lod[t] - 1
        r["new_hod_30s"] = float(hod[t] > hod[max(t - 30, 0)])
        r["micro_hh"] = float(hh[t])
        r["micro_hl"] = float(hl[t])
        r["above_or5"] = float(p > or5_h) if t >= first + 300 else np.nan
        r["below_or5"] = float(p < or5_l) if t >= first + 300 else np.nan
        r["dist_trigger"] = sgn * (p / trigger - 1)
        r["dist_trigger_atr"] = sgn * (p - trigger) / atr if atr > 0 else np.nan
        r["crossed"] = float(first_cross >= 0 and t >= first_cross)
        r["secs_since_cross"] = (t - first_cross) if first_cross >= 0 and t >= first_cross else np.nan
        # retest: after crossing, price came back within 0.5% of trigger and is now back beyond it
        if first_cross >= 0 and t > first_cross + 5:
            after = px[first_cross:t + 1]
            near = np.nanmin(sgn * (after / trigger - 1))
            r["retest_depth"] = near
            r["retest_hold"] = float(near <= 0.005 and near >= -0.01 and sgn * (p / trigger - 1) > 0.002)
        else:
            r["retest_depth"] = np.nan
            r["retest_hold"] = 0.0
        r["dist_vwap"] = p / vwap[t] - 1 if np.isfinite(vwap[t]) else np.nan
        r["dist_vwap_atr"] = (p - vwap[t]) / atr if atr > 0 and np.isfinite(vwap[t]) else np.nan
        r["vwap_slope_5m"] = vwap[t] / vwap[t - 300] - 1 if t - 300 >= first and np.isfinite(vwap[t - 300]) else np.nan
        r["time_above_vwap"] = t_above[t]
        was_below = np.nanmin(px[max(first, t - 180):t - 30] - vwap[max(first, t - 180):t - 30]) < 0 if t - 30 > first else False
        r["vwap_reclaim"] = float(bool(was_below) and p > vwap[t])
        if np.isfinite(mkt[t]) and np.isfinite(mkt[first]):
            mret = mkt[t] / mkt[first] - 1
            r["rel_open"] = r["ret_open"] - mret
            r["mkt_r_5m"] = mkt[t] / mkt[t - 300] - 1 if t - 300 >= 0 and np.isfinite(mkt[t - 300]) else np.nan
            r["rel_r_5m"] = (r["r_5m"] - r["mkt_r_5m"]) if np.isfinite(r.get("mkt_r_5m", np.nan)) else np.nan
            r["rel_r_1m"] = r["r_1m"] - (mkt[t] / mkt[t - 60] - 1 if np.isfinite(mkt[t - 60]) else np.nan)
        else:
            r["rel_open"] = r["mkt_r_5m"] = r["rel_r_5m"] = r["rel_r_1m"] = np.nan
        r["vol_int_1m"] = v60[t] / (60 * exp_per_sec) if exp_per_sec and np.isfinite(v60[t]) else np.nan
        r["vol_int_5m"] = v300[t] / (300 * exp_per_sec) if exp_per_sec and np.isfinite(v300[t]) else np.nan
        r["vol_accel"] = (v60[t] / 60) / (v300[t] / 300) if np.isfinite(v300[t]) and v300[t] > 0 else np.nan
        r["cum_vol_ratio"] = cum_v[t] / (exp_per_sec * (t - first + 1)) if exp_per_sec else np.nan
        # micro stops available at t (PIT)
        micro_low = np.nanmin(lo[max(first, t - 120):t + 1]) if sgn > 0 else np.nanmax(hi[max(first, t - 120):t + 1])
        r["micro_stop"] = micro_low
        # ---- outcomes (future, research labels only) ----
        ep = px[e]
        r["entry_price"] = ep
        fh = suf_max[e]
        fl = suf_min[e]
        r["mfe_eod"] = sgn * ((fh if sgn > 0 else fl) / ep - 1)
        r["mae_eod"] = sgn * ((fl if sgn > 0 else fh) / ep - 1)
        for nm, hs in HORIZONS.items():
            k = min(e + hs, N_SEC - 1)
            r[f"fwd_{nm}"] = sgn * (px[k] / ep - 1)
            r[f"mfe_{nm}"] = sgn * (((fwd_max[nm][e]) if sgn > 0 else fwd_min[nm][e]) / ep - 1)
            r[f"mae_{nm}"] = sgn * (((fwd_min[nm][e]) if sgn > 0 else fwd_max[nm][e]) / ep - 1)
        r["fwd_eod"] = sgn * (px[N_SEC - 1] / ep - 1)
        r["fwd_next"] = sgn * (next_close / ep - 1) if next_close and next_close > 0 else np.nan
        r["daily_stop_hit"] = float((fl <= stop) if sgn > 0 else (fh >= stop))
        r["stop_dist_daily"] = sgn * (ep - stop) / ep
        r["stop_dist_daily_atr"] = sgn * (ep - stop) / atr if atr > 0 else np.nan
        r["stop_dist_micro"] = sgn * (ep - micro_low) / ep
        r["stop_dist_micro_atr"] = sgn * (ep - micro_low) / atr if atr > 0 else np.nan
        r["micro_stop_hit"] = float((fl <= micro_low) if sgn > 0 else (fh >= micro_low))
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.insert(0, "date", d)
    df.insert(0, "stock_id", sid)
    df["side"] = side
    df["trigger"] = trigger
    df["stop"] = stop
    df["first_cross_sec"] = first_cross
    df["open_price"] = open_px
    if ctx:
        for k, v in ctx.items():
            df[k] = v
    return df


FEATURES = ["since_open_min", "r_1s", "r_5s", "r_15s", "r_30s", "r_1m", "r_3m", "r_5m", "accel_1m", "rv_5m",
            "ret_open", "ret_prevclose", "dist_hod", "dist_lod", "new_hod_30s", "micro_hh", "micro_hl", "above_or5",
            "below_or5", "dist_trigger", "dist_trigger_atr", "crossed", "secs_since_cross", "retest_depth",
            "retest_hold", "dist_vwap", "dist_vwap_atr", "vwap_slope_5m", "time_above_vwap", "vwap_reclaim",
            "rel_open", "mkt_r_5m", "rel_r_5m", "rel_r_1m", "vol_int_1m", "vol_int_5m", "vol_accel", "cum_vol_ratio"]


def utility(df: pd.DataFrame, lam: float = 1.0, cost: float = 0.0) -> pd.Series:
    """FutureUtility = MFE(EOD) - lam * |MAE(EOD)| - cost."""
    return df["mfe_eod"] - lam * df["mae_eod"].abs() - cost
