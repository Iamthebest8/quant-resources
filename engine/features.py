"""PIT feature engine (Parts 7-20).

Every feature at date t uses only data with timestamp <= t (rolling windows,
cross-sectional ranks within the same date). Trades based on a feature at t
are executed at the open of t+1.  tests/test_pit.py verifies this by
recomputing features on data truncated at t and checking equality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.panel import Panel

EPS = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rolling_slope(y: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """OLS slope of y on time over a trailing window of n rows (per column)."""
    T = len(y)
    idx = np.arange(T, dtype=float)
    if isinstance(y, pd.DataFrame):
        idx_b = pd.DataFrame(np.repeat(idx[:, None], y.shape[1], axis=1), index=y.index, columns=y.columns)
    else:
        idx_b = pd.Series(idx, index=y.index)
    sy = y.rolling(n, min_periods=n).sum()
    sty = (y * idx_b).rolling(n, min_periods=n).sum()
    start = idx_b - n + 1
    s_rel = sty - start * sy
    st = n * (n - 1) / 2.0
    stt = (n - 1) * n * (2 * n - 1) / 6.0
    return (n * s_rel - st * sy) / (n * stt - st * st)


def cs_pct(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    x = df.where(mask) if mask is not None else df
    return x.rank(axis=1, pct=True)


def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    if isinstance(tr, np.ndarray):
        tr = pd.DataFrame(tr, index=c.index, columns=c.columns)
    return tr.rolling(n, min_periods=n // 2).mean()


def clv(h, l, c):
    rng = (h - l)
    return ((c - l) / rng.where(rng > EPS)).fillna(0.5)


def compound(r: pd.DataFrame | pd.Series, n: int):
    return np.exp(np.log1p(r.fillna(0)).rolling(n, min_periods=max(2, int(n * 0.8))).sum()) - 1


# ---------------------------------------------------------------------------
# market regime (Part 7)
# ---------------------------------------------------------------------------
def market_features(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=m.index)
    c, o, h, l = m["adj_close"], m["adj_open"], m["adj_high"], m["adj_low"]
    r = m["ret"].fillna(0)
    out["m_ret"] = r
    out["m_oc"] = (m["close"] / m["open"] - 1).where(m["open"] > 0)
    out["m_clv"] = clv(h, l, c)
    out["m_lc"] = c / l - 1
    out["m_range"] = (h - l) / c.shift(1)
    out["m_ma20"] = c.rolling(20).mean()
    out["m_ma60"] = c.rolling(60).mean()
    out["m_ma20_slope"] = out["m_ma20"] / out["m_ma20"].shift(5) - 1
    out["m_ret5"] = c / c.shift(5) - 1
    out["m_ret20"] = c / c.shift(20) - 1
    vol20 = r.rolling(20).std()
    out["m_vol20"] = vol20
    out["m_tstat20"] = np.log1p(out["m_ret20"]) / (vol20 * np.sqrt(20) + EPS)
    out["m_er20"] = (c - c.shift(20)).abs() / (c.diff().abs().rolling(20).sum() + EPS)
    out["m_oc10"] = out["m_oc"].rolling(10).mean()
    hh = (h.rolling(5).max() > h.shift(5).rolling(5).max()).astype(float)
    hl = (l.rolling(5).min() > l.shift(5).rolling(5).min()).astype(float)
    out["m_struct"] = hh + hl - 1    # +1 HH&HL, -1 LH&LL
    out["m_at_high20"] = (c >= c.rolling(20).max() - EPS).astype(float)
    up = (out["m_tstat20"] >= 1.0) & (c > out["m_ma20"]) & (out["m_ma20_slope"] > 0)
    dn = (out["m_tstat20"] <= -1.0) & (c < out["m_ma20"]) & (out["m_ma20_slope"] < 0)
    reg = pd.Series("MARKET_SIDEWAYS", index=m.index)
    reg[up] = "MARKET_UP"
    reg[dn] = "MARKET_DOWN"
    reg[out["m_ma20"].isna()] = "UNKNOWN"
    out["regime"] = reg
    day = pd.Series("FLAT", index=m.index)
    day[r >= 0.005] = "UP"
    day[r <= -0.005] = "DOWN"
    out["m_day"] = day
    return out


# ---------------------------------------------------------------------------
# stock features
# ---------------------------------------------------------------------------
FEATURE_FAMILIES = {
    # family -> candidate features (all oriented: higher = stronger, unless listed in NEGATIVE)
    "MKT_REL": ["ex_1", "cex_5", "resid_5", "oc_rel5", "clv_rel5", "lc_rel5"],
    "PERSIST": ["outp_5", "outp_10", "cex_10", "rs_slope_10"],
    "ACCEL": ["rs_accel"],
    "SECTOR_REL": ["srs_5", "srs_10", "sec_outp_10"],
    "INDEP": ["is_cnt5", "is_cnt10"],
    "RESIL": ["dr60", "dcap60"],
    "UPPART": ["up60", "ucap60"],
    "ASYM": ["asym60"],
    "STRUCT": ["dist_h20", "dist_h60", "progress_atr", "hl_flag", "relhigh_cnt20", "tight10"],
    "VOLUME": ["vol_ratio", "val_accel", "val_pct", "turnover_pct"],
    "TRAD_RS": ["rs_pct_20", "rs_pct_40", "rs_pct_60", "rs_pct_120"],
}
NEGATIVE = {"dcap60", "tight10"}   # lower = better
CONTROLS = ["beta", "atr_pct", "hv20", "val20", "val_pct"]
# features allowed into the Discovery score (Part 15: Probe must not wait for mature RS60)
DISCOVERY_FAMILIES = ["MKT_REL", "PERSIST", "ACCEL", "SECTOR_REL", "INDEP", "RESIL", "UPPART", "ASYM", "STRUCT",
                      "VOLUME"]


def compute_features(p: Panel, log=print) -> dict[str, pd.DataFrame]:
    """Return dict of wide (dates x ids) feature frames + 'mkt' market frame."""
    F: dict[str, pd.DataFrame] = {}
    mk = market_features(p.market)
    F["mkt"] = mk
    c, o, h, l, r = p.c, p.o, p.h, p.l, p.r
    U = p.universe
    m_ret = mk["m_ret"]
    m_close = p.market["adj_close"]

    def bc(s: pd.Series) -> pd.DataFrame:  # broadcast market series
        return pd.DataFrame(np.repeat(s.to_numpy()[:, None], len(p.ids), axis=1), index=p.dates, columns=p.ids)

    M = bc(m_ret)
    # --- beta & vol controls (Part 18) ---------------------------------------------
    rr = r.fillna(0)
    cov = (rr * M).rolling(120, min_periods=60).mean() - rr.rolling(120, min_periods=60).mean() * M.rolling(
        120, min_periods=60).mean()
    var = M.rolling(120, min_periods=60).var(ddof=0)
    beta_raw = cov / (var + EPS)
    F["beta"] = (0.7 * beta_raw + 0.3).clip(-1, 4)
    a14 = atr(h, l, c, 14)
    F["atr"] = a14
    F["atr_pct"] = a14 / c
    F["hv20"] = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    val = p.val.where(c.notna())
    F["val20"] = val.rolling(20, min_periods=15).mean()
    F["val_pct"] = cs_pct(F["val20"], U)

    # --- market relative (Part 8) ------------------------------------------------------
    ex1 = r - M
    F["ret_1"] = r
    F["ex_1"] = ex1
    F["resid_1"] = r - F["beta"] * M
    F["cex_5"] = compound(r, 5) - compound(M, 5)
    F["resid_5"] = F["resid_1"].rolling(5, min_periods=4).sum()
    oc = (p.raw_c / p.raw_o - 1)
    F["oc"] = oc
    F["oc_rel"] = oc - bc(mk["m_oc"])
    F["oc_rel5"] = F["oc_rel"].rolling(5, min_periods=4).mean()
    s_clv = clv(h, l, c)
    F["clv"] = s_clv
    F["clv_rel"] = s_clv - bc(mk["m_clv"])
    F["clv_rel5"] = F["clv_rel"].rolling(5, min_periods=4).mean()
    lc = c / l - 1
    F["lc_rel"] = lc - bc(mk["m_lc"])
    F["lc_rel5"] = F["lc_rel"].rolling(5, min_periods=4).mean()
    rsl = np.log(c.div(m_close, axis=0))
    F["rsl"] = rsl
    F["rsl_high20"] = (rsl >= rsl.rolling(20, min_periods=15).max() - 1e-9).astype(float).where(c.notna())
    F["rsl_high40"] = (rsl >= rsl.rolling(40, min_periods=30).max() - 1e-9).astype(float).where(c.notna())
    c_hi20 = c.rolling(20, min_periods=15).max()
    m_at_hi = bc(mk["m_at_high20"])
    F["px_high20"] = (c >= c_hi20 - 1e-9).astype(float).where(c.notna())
    F["relhigh"] = ((F["px_high20"] > 0) & (m_at_hi < 1)).astype(float).where(c.notna())
    F["relhigh_cnt20"] = F["relhigh"].rolling(20, min_periods=10).sum()

    # --- persistence (Part 14) ---------------------------------------------------------------
    for n in (3, 5, 10):
        F[f"outp_{n}"] = (ex1 > 0).astype(float).where(r.notna()).rolling(n, min_periods=n).mean()
    F["cex_3"] = compound(r, 3) - compound(M, 3)
    F["cex_10"] = compound(r, 10) - compound(M, 10)
    for n in (5, 10, 20):
        F[f"rs_slope_{n}"] = rolling_slope(rsl, n)
    F["rs_accel"] = F["rs_slope_5"] - F["rs_slope_20"]

    # --- traditional RS (Part 15) -----------------------------------------------------------
    for n in (10, 20, 40, 60, 120):
        rs = (c / c.shift(n)).div(m_close / m_close.shift(n), axis=0) - 1
        F[f"rs_{n}"] = rs
        F[f"rs_pct_{n}"] = cs_pct(rs, U)
    F["ret_10"] = c / c.shift(10) - 1
    F["ret_20"] = c / c.shift(20) - 1
    F["ret_60"] = c / c.shift(60) - 1
    F["cs_rank20"] = cs_pct(F["ret_20"], U)

    # --- sector relative (Part 13): leave-one-out equal-weight sector return -------------
    sec = p.sector.reindex(p.ids)
    rU = r.where(U.shift(1, fill_value=False))   # members = universe as of yesterday
    sec_ret = pd.DataFrame(index=p.dates, columns=p.ids, dtype=float)
    sec_n = pd.DataFrame(index=p.dates, columns=p.ids, dtype=float)
    for s_name, cols in sec.groupby(sec).groups.items():
        cols = list(cols)
        sub = rU[cols]
        tot = sub.sum(axis=1, min_count=1)
        cnt = sub.notna().sum(axis=1)
        own = sub.fillna(0)
        own_n = sub.notna().astype(int)
        loo_n = cnt.to_numpy()[:, None] - own_n.to_numpy()
        loo = (tot.to_numpy()[:, None] - own.to_numpy()) / np.where(loo_n > 0, loo_n, np.nan)
        if s_name == "UNKNOWN":
            loo = np.full_like(loo, np.nan)
        sec_ret[cols] = loo
        sec_n[cols] = loo_n
    thin = sec_n < 4
    sec_ret = sec_ret.mask(thin)
    sec_ret = sec_ret.fillna(M)            # thin / unknown sector -> market
    F["sec_ret"] = sec_ret.astype(float)
    F["sec_n"] = sec_n
    F["sec_ex_1"] = r - F["sec_ret"]
    for n in (5, 10, 20):
        F[f"srs_{n}"] = compound(r, n) - compound(F["sec_ret"], n)
        F[f"secmkt_{n}"] = compound(F["sec_ret"], n) - compound(M, n)
    F["sec_outp_10"] = (F["sec_ex_1"] > 0).astype(float).where(r.notna()).rolling(10, min_periods=10).mean()
    sec_lvl = np.exp(np.log1p(F["sec_ret"].fillna(0)).cumsum())
    F["sec_level"] = sec_lvl

    # --- independent strength event (Part 9) -------------------------------------------
    hl1 = (l >= l.shift(1)).astype(float)
    mkt_not_up = bc((mk["m_ret"] <= 0.003).astype(float))
    F["mkt_not_up"] = mkt_not_up
    F["is_event"] = ((mkt_not_up > 0) & (ex1 >= 0.02) & (s_clv >= 0.7) & (hl1 > 0)).astype(float).where(r.notna())
    F["is_cnt5"] = F["is_event"].rolling(5, min_periods=3).sum()
    F["is_cnt10"] = F["is_event"].rolling(10, min_periods=5).sum()

    # --- downside resilience / upside participation (Parts 10-12) ------------------------
    dn_day = bc((mk["m_ret"] <= -0.008).astype(float))
    up_day = bc((mk["m_ret"] >= 0.008).astype(float))
    win = 60
    n_dn = dn_day.rolling(win).sum()
    n_up = up_day.rolling(win).sum()
    F["dr60"] = (ex1.fillna(0) * dn_day).rolling(win).sum() / n_dn.where(n_dn >= 3)
    F["up60"] = (ex1.fillna(0) * up_day).rolling(win).sum() / n_up.where(n_up >= 3)
    F["dcap60"] = (rr * dn_day).rolling(win).sum() / (M * dn_day).rolling(win).sum().where(n_dn >= 3)
    F["ucap60"] = (rr * up_day).rolling(win).sum() / (M * up_day).rolling(win).sum().where(n_up >= 3)
    F["asym60"] = F["ucap60"] - F["dcap60"]
    for k in ("dr60", "up60", "dcap60", "ucap60", "asym60"):
        F[k] = F[k].where(c.notna())

    # --- price structure (Part 16) ------------------------------------------------------
    for n in (5, 10, 20, 60):
        F[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
    F["ma20_slope"] = F["ma20"] / F["ma20"].shift(5) - 1
    F["ma60_slope"] = F["ma60"] / F["ma60"].shift(10) - 1
    for n in (20, 40, 60, 250):
        F[f"dist_h{n}"] = c / h.rolling(n, min_periods=int(n * 0.8)).max() - 1
    F["dist_hist"] = c / h.cummax() - 1
    F["hl_flag"] = (l.rolling(5).min() > l.shift(5).rolling(5).min()).astype(float).where(c.notna())
    F["hh_flag"] = (h.rolling(5).max() > h.shift(5).rolling(5).max()).astype(float).where(c.notna())
    F["tight10"] = (h.rolling(10).max() - l.rolling(10).min()) / (a14 + EPS)
    F["range5_atr"] = (h.rolling(5).max() - l.rolling(5).min()) / (a14 + EPS)
    prior_hi20 = h.shift(1).rolling(20, min_periods=15).max()
    F["prior_hi20"] = prior_hi20
    F["bo_attempt"] = (h > prior_hi20).astype(float).where(c.notna())
    F["bo_close"] = (c > prior_hi20).astype(float).where(c.notna())
    below = (c < F["ma20"]).astype(float)
    F["reclaim"] = ((c > F["ma20"]) & (below.shift(1).rolling(5).max() > 0)).astype(float).where(c.notna())
    F["progress_atr"] = (c - c.shift(10)) / (a14 + EPS)
    F["ext_ma20"] = c / F["ma20"] - 1
    F["trend_ok"] = ((c > F["ma20"]) & (F["ma20_slope"] > 0) & (F["hh_flag"] > 0) & (F["hl_flag"] > 0)
                     ).astype(float).where(c.notna())
    hv_d = r.rolling(20, min_periods=15).std()
    F["cap_vel"] = F["cex_10"] / (hv_d * np.sqrt(10) + EPS)
    F["low3"] = l.rolling(3, min_periods=2).min()

    # --- volume / turnover (Part 19), not comparable under disposition (Part 20) ----------
    vol = p.vol.where(c.notna())
    nc = p.disp | p.disp.shift(1, fill_value=False)
    vma20 = vol.rolling(20, min_periods=15).mean()
    F["vol_ratio"] = (vol / vma20).mask(nc)
    F["vol_accel"] = (vol.rolling(5).mean() / vma20).mask(nc)
    F["val_accel"] = (val.rolling(5).mean() / F["val20"]).mask(nc)
    F["val_pctile_ts"] = val.rolling(120, min_periods=60).rank(pct=True).mask(nc)
    F["vol_comparable"] = (~nc).astype(float)
    if p.mktval is not None:
        F["turnover"] = (F["val20"] / p.mktval).where(p.mktval > 0)          # avg daily turnover (fraction of cap)
        F["turnover_pct"] = cs_pct(F["turnover"], U)
        F["turnover_accel"] = (val.rolling(5).mean() / p.mktval).where(p.mktval > 0).mask(nc) / F["turnover"]
    else:
        F["turnover_pct"] = F["val_pct"]

    # --- cross-sectional percentile versions of candidate features (for scores) ----------------
    for fam, feats in FEATURE_FAMILIES.items():
        for f in feats:
            if f.startswith("rs_pct_") or f in ("val_pct", "turnover_pct") or f not in F:
                continue
            x = -F[f] if f in NEGATIVE else F[f]
            F[f"pct_{f}"] = cs_pct(x, U)
    for k, v in list(F.items()):
        if k != "mkt" and isinstance(v, pd.DataFrame) and v.dtypes.iloc[0] == np.float64:
            F[k] = v.astype(np.float32)
    log(f"[features] {len(F)} feature frames computed")
    return F


RANK_FEATURES = ["outp_10", "srs_10", "rs_accel", "progress_atr", "relhigh_cnt20", "cap_vel"]


def ranking_score(F: dict, U: pd.DataFrame) -> pd.DataFrame:
    """Part 23 probe ranking: 'who looks most like a leader in formation' (fixed, equal-weight)."""
    parts = [cs_pct(F[f], U) for f in RANK_FEATURES]
    return sum(parts) / len(parts)
