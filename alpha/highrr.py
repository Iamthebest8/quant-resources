"""HIGH REWARD/RISK research (Parts on PRE_TRADE_RR, overhead resistance, RS-leads-price, resilience, sector,
compression, volume, base quality, trend extension).

Method (no indicator soup, no grid search):
  1. Event panel = every universe stock at every week-end close that is in weekly stage 1 or 2 (long side).
     Features are PIT at the week-end close; entry = next open; outcomes over 40 trading days:
       fwd40      close(t+40)/open(t+1)-1
       mfe40/mae40
       stop_hit   low <= structural stop (swing low 20d * 0.99) within 40 days
       ret_s40    stop-aware return: -stop distance if the stop is hit first, else fwd40
       r_s40      ret_s40 / stop distance (realised R)
       ge20       fwd40 >= 20%  (right-tail hit)
  2. Univariate deciles + Spearman per window (PRE 2020-22, DISCOVERY 2023-24 = selection, STRICT_OOS shown only).
  3. A small rank-based HIGH_RR score: <= 1 feature per family, chosen ONLY from DISCOVERY decile monotonicity
     (|Spearman of decile means| >= 0.6 and sign agreeing in PRE), equal-weight average of signed percentiles.
     Frozen to outputs/frozen/HIGH_RR_SCORE_V1.json before any OOS evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H = 40
FAMILIES = {
    "RISK": ["stop_dist_pct", "stop_dist_atr"],
    "UPSIDE": ["room_h60", "room_h120", "room_h250", "res_dist_capped", "blue_sky", "base_height"],
    "PRE_TRADE_RR": ["rr_res", "rr_h250", "rr_base"],
    "OVERHEAD": ["res_n5", "res_n10", "res_n15", "supply5", "supply10", "supply15", "resdays10"],
    "RS": ["rs10", "rs20", "rs40", "rs60", "rs120", "rs120_pct", "rs_slope20", "rs_accel", "mrs", "rs_slope13"],
    "RS_LEADS": ["rs_leads60_10d", "rs_leads120_10d", "rs_newhigh120"],
    "RESILIENCE": ["dcap60", "dr60", "ucap60", "up60", "asym60"],
    "SECTOR": ["srs_20", "secmkt_20", "grp_stage2_pct"],
    "COMPRESSION": ["atr5_20", "atr10_60", "bbw_pct120", "range5_20", "rv10_60"],
    "VOLUME": ["vol5_20", "vol10_60", "vol_pct250", "brk_vol"],
    "BASE": ["base_weeks", "base_depth", "base_tight10w"],
    "EXTENSION": ["ext_ma20_atr", "ext_ma60_atr", "dist_ma150", "stage2_age", "ret_since_stage2", "stage_w"],
}
ALL_FEATURES = [f for v in FAMILIES.values() for f in v]


def forward_outcomes(p, stop_mult: float = 0.99, h: int = H) -> dict[str, np.ndarray]:
    """Daily (T x N) forward outcome matrices for an entry at the NEXT open after day t (path-ordered)."""
    O, Hh, L, C = (p.o.to_numpy(), p.h.to_numpy(), p.l.to_numpy(), p.c.to_numpy())
    T, N = C.shape
    entry = np.vstack([O[1:], np.full((1, N), np.nan)])
    stop = p.l.rolling(20, min_periods=15).min().to_numpy() * stop_mult
    mx = np.full((T, N), -np.inf)
    mn = np.full((T, N), np.inf)
    first_hit = np.full((T, N), h + 1)
    first_up = np.full((T, N), h + 1)
    up_lvl = entry * 1.2
    for k in range(1, h + 1):
        hk = np.full((T, N), np.nan)
        lk = np.full((T, N), np.nan)
        hk[:T - k] = Hh[k:]
        lk[:T - k] = L[k:]
        mx = np.fmax(mx, hk)
        mn = np.fmin(mn, lk)
        with np.errstate(invalid="ignore"):
            hit = (lk <= stop) & (first_hit > h)
            up = (hk >= up_lvl) & (first_up > h)
        first_hit[hit] = k
        first_up[up] = k
    fwd = np.full((T, N), np.nan)
    fwd[:T - h] = C[h:] / entry[:T - h] - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mfe = np.where(np.isfinite(mx), mx / entry - 1, np.nan)
        mae = np.where(np.isfinite(mn), mn / entry - 1, np.nan)
        stop_dist = 1 - stop / entry
    stop_hit = first_hit <= h
    ret_s = np.where(stop_hit, -stop_dist, fwd)
    ret_s[~np.isfinite(fwd) & ~stop_hit] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r_s = ret_s / stop_dist
    return {"fwd40": fwd, "mfe40": mfe, "mae40": mae, "stop_dist_pct": stop_dist, "stop_hit": stop_hit.astype(float),
            "ret_s40": ret_s, "r_s40": r_s, "ge20": (fwd >= 0.2).astype(float),
            "up20_before_stop": (first_up < first_hit).astype(float), "entry_open": entry}


def event_panel(p, F, G, ctx, log=print) -> pd.DataFrame:
    """Week-end panel of stage-1/2 universe stocks with features + forward outcomes."""
    from alpha.weekly import week_end_index
    we = week_end_index(p.dates)
    is_we = (we.to_numpy() == p.dates.to_numpy())
    ti = np.nonzero(is_we)[0]
    wk = ctx.wd.week_end
    wpos = p.dates.get_indexer(wk)
    assert (np.sort(wpos) == ti).all()
    st = ctx.S["stage"].to_numpy()
    U = p.universe.to_numpy()[ti]
    nodisp = (~(p.disp | p.disp_next)).to_numpy()[ti]
    sel = U & nodisp & np.isin(st, [1, 2])
    log(f"[highrr] event panel candidates: {int(sel.sum()):,}")
    out = forward_outcomes(p)
    c = p.c
    mk = p.market["adj_close"]
    cols = {}
    wi, js = np.nonzero(sel)
    rows_t = ti[wi]

    def take(mat):
        a = mat.to_numpy() if isinstance(mat, pd.DataFrame) else mat
        return a[rows_t, js]

    def takew(frame):
        a = frame.to_numpy() if isinstance(frame, pd.DataFrame) else frame
        return a[wi, js]

    cols["date"] = p.dates[rows_t]
    cols["stock_id"] = np.array(p.ids)[js]
    cols["t"] = rows_t
    cols["j"] = js
    cols["stage_w"] = takew(st).astype(float)
    for k, v in out.items():
        if k != "entry_open":
            cols[k] = take(v)
    atr = F["atr"].to_numpy()
    cols["stop_dist_atr"] = cols["stop_dist_pct"] * take(out["entry_open"]) / take(atr)
    for k in ("room_h60", "room_h120", "room_h250", "res_dist_capped", "blue_sky", "res_n5", "res_n10", "res_n15",
              "supply5", "supply10", "supply15", "resdays10", "rs_leads60_10d", "rs_leads120_10d", "rs_newhigh120",
              "atr5_20", "atr10_60", "bbw_pct120", "range5_20", "rv10_60", "vol5_20", "vol10_60", "vol_pct250",
              "brk_vol", "ext_ma20_atr", "ext_ma60_atr", "dist_ma150", "stage2_age", "ret_since_stage2"):
        cols[k] = take(G[k]) if k in G else np.full(len(rows_t), np.nan)
    for n in (10, 20, 40, 60, 120):
        ex = (c / c.shift(n) - 1).sub(mk / mk.shift(n) - 1, axis=0)
        cols[f"rs{n}"] = take(ex)
        if n == 120:
            cols["rs120_pct"] = take(ex.where(p.universe).rank(axis=1, pct=True))
    for k in ("rs_slope_20", "rs_accel", "dcap60", "dr60", "ucap60", "up60", "asym60", "srs_20", "secmkt_20"):
        cols[k.replace("rs_slope_20", "rs_slope20")] = take(F[k]) if k in F else np.full(len(rows_t), np.nan)
    cols["mrs"] = takew(ctx.mrs)
    cols["rs_slope13"] = takew(ctx.rs_slope13)
    cols["grp_stage2_pct"] = takew(ctx.grp_stage2_pct)
    # base geometry from the weekly stage episode (stage-1 base or current stage-2 leg)
    hi52 = ctx.wd.high.rolling(52, min_periods=1).max()
    lo52 = ctx.wd.low.rolling(52, min_periods=1).min()
    Rb = np.fmin(ctx.S["ep_maxh"].to_numpy(), hi52.to_numpy())
    Sb = np.fmax(ctx.S["ep_minl"].to_numpy(), lo52.to_numpy())
    cols["base_weeks"] = takew(ctx.S["ep_len"]).astype(float)
    cols["base_depth"] = takew(Rb / Sb - 1)
    tight = ctx.wd.high.rolling(10, min_periods=8).max() / ctx.wd.low.rolling(10, min_periods=8).min() - 1
    cols["base_tight10w"] = takew(tight)
    cols["base_height"] = takew((Rb - Sb) / ctx.wd.close.to_numpy())
    df = pd.DataFrame(cols)
    sd = df["stop_dist_pct"].where(df["stop_dist_pct"] > 0.005)
    room_res = df["res_dist_capped"].fillna(1.0).clip(upper=1.0)
    df["rr_res"] = room_res / sd
    df["rr_h250"] = df["room_h250"].where(df["room_h250"] > 0.005, room_res) / sd
    df["rr_base"] = df["base_height"] / sd
    df = df[np.isfinite(df["ret_s40"]) & (df["stop_dist_pct"] > 0)]
    for k in df.columns:
        if df[k].dtype == np.float64:
            df[k] = df[k].astype(np.float32)
    log(f"[highrr] event panel rows: {len(df):,}")
    return df


def window_masks(dates: pd.Series, windows: dict) -> dict:
    return {w: (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)) for w, (a, b) in windows.items()}


def decile_table(df: pd.DataFrame, feat: str, windows: dict, q: int = 10, outcomes=("ret_s40", "r_s40", "ge20",
                 "stop_hit", "mfe40", "mae40", "up20_before_stop")) -> pd.DataFrame:
    """Deciles are cut with DISCOVERY-window breakpoints (frozen) and applied to every window."""
    x = df[feat].astype(float)
    disc = window_masks(df["date"], {"D": windows["DISCOVERY"]})["D"]
    xd = x[disc & np.isfinite(x)]
    if xd.nunique() < 3:
        return pd.DataFrame()
    edges = np.unique(np.nanquantile(xd, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        return pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(x, edges, labels=False, include_lowest=True)
    rows = []
    for w, m in window_masks(df["date"], windows).items():
        sub = df[m & b.notna()]
        bb = b[m & b.notna()]
        for k, g in sub.groupby(bb):
            r = {"feature": feat, "window": w, "bin": int(k), "n": len(g),
                 "x_lo": float(edges[int(k)]), "x_hi": float(edges[int(k) + 1]), "x_med": float(g[feat].median())}
            for o in outcomes:
                r[o] = float(g[o].mean())
            pos = g["ret_s40"][g["ret_s40"] > 0]
            neg = g["ret_s40"][g["ret_s40"] < 0]
            r["payoff"] = float(pos.mean() / -neg.mean()) if len(pos) and len(neg) else np.nan
            r["pf"] = float(pos.sum() / -neg.sum()) if len(neg) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def monotonicity(tab: pd.DataFrame, window: str, outcome: str = "ret_s40") -> float:
    t = tab[tab["window"] == window].sort_values("bin")
    if len(t) < 4:
        return np.nan
    return float(pd.Series(t[outcome].to_numpy()).corr(pd.Series(np.arange(len(t))), method="spearman"))


def spearman_ic(df: pd.DataFrame, feat: str, outcome: str = "ret_s40") -> float:
    """Mean weekly cross-sectional Spearman IC."""
    g = df[["date", feat, outcome]].dropna()
    if len(g) < 100:
        return np.nan
    ics = g.groupby("date").apply(lambda x: x[feat].rank().corr(x[outcome].rank()) if len(x) >= 20 else np.nan,
                                  include_groups=False)
    return float(ics.mean())


def select_score(summary: pd.DataFrame, min_mono: float = 0.6) -> dict:
    """Pick at most one feature per family from DISCOVERY monotonicity, requiring PRE sign agreement."""
    chosen = {}
    for fam, feats in FAMILIES.items():
        if fam in ("PRE_TRADE_RR", "RISK"):
            # RISK / PRE_TRADE_RR are mechanically coupled to the stop-aware outcome (ret_s40 = -stop distance when
            # stopped); they are tested as a separate hypothesis, never auto-selected into the score
            continue
        cand = summary[(summary["family"] == fam)]
        cand = cand[(cand["mono_DISCOVERY"].abs() >= min_mono)
                    & (np.sign(cand["mono_DISCOVERY"]) == np.sign(cand["mono_PRE_2020_2022"]))]
        if len(cand) == 0:
            continue
        best = cand.iloc[cand["mono_DISCOVERY"].abs().argmax()]
        chosen[fam] = {"feature": best["feature"], "sign": int(np.sign(best["mono_DISCOVERY"])),
                       "mono_discovery": float(best["mono_DISCOVERY"]), "mono_pre": float(best["mono_PRE_2020_2022"])}
    return chosen


def score_frame(df: pd.DataFrame, spec: dict) -> pd.Series:
    """Equal-weight mean of signed cross-sectional percentiles (within each week)."""
    parts = []
    for v in spec.values():
        f, s = v["feature"], v["sign"]
        pct = df.groupby("date")[f].rank(pct=True)
        parts.append((pct if s > 0 else 1 - pct).fillna(0.5))
    if not parts:
        return pd.Series(0.5, index=df.index)
    return sum(parts) / len(parts)


def score_at(p, F, G, ctx, spec: dict, rows: pd.DataFrame, ref_panel: pd.DataFrame) -> np.ndarray:
    """HIGH_RR score for arbitrary (t, j) rows (e.g. strategy setups): percentile of each feature vs the
    event-panel cross-section of the SAME week (last week-end <= t), signed, averaged."""
    out = np.full(len(rows), np.nan)
    if not spec or len(rows) == 0:
        return out
    feats = [v["feature"] for v in spec.values()]
    refs = {d: g for d, g in ref_panel.groupby("date")}
    week_dates = np.array(sorted(refs))
    # feature values for the rows: look them up in the panel when the row is in it, else compute from G/F at t
    key = ref_panel.set_index(["t", "j"])
    vals = {}
    for f in feats:
        vals[f] = key[f] if f in key else None
    for i, r in enumerate(rows.itertuples()):
        k = np.searchsorted(week_dates, np.datetime64(p.dates[r.t]), side="right") - 1
        if k < 0:
            continue
        ref = refs[week_dates[k]]
        acc, n = 0.0, 0
        for v in spec.values():
            f, s = v["feature"], v["sign"]
            x = _feature_value(p, F, G, ctx, f, r.t, r.j, key)
            col = ref[f].to_numpy(float)
            col = col[np.isfinite(col)]
            if not np.isfinite(x) or len(col) < 20:
                acc += 0.5
            else:
                pc = (col < x).mean() + 0.5 * (col == x).mean()
                acc += pc if s > 0 else 1 - pc
            n += 1
        out[i] = acc / max(n, 1)
    return out


def _feature_value(p, F, G, ctx, f, t, j, key) -> float:
    try:
        return float(key.at[(t, j), f])
    except KeyError:
        pass
    if f in ("mrs", "rs_slope13", "rs_slope4", "grp_stage2_pct"):
        frame = {"mrs": ctx.mrs, "rs_slope13": ctx.rs_slope13, "rs_slope4": ctx.rs_slope4,
                 "grp_stage2_pct": ctx.grp_stage2_pct}[f]
        k = ctx.wd.week_end.searchsorted(p.dates[t], side="right") - 1
        return float(frame.iat[k, j]) if k >= 0 else np.nan
    if f in G:
        return float(G[f].iat[t, j])
    if f in F:
        return float(F[f].iat[t, j])
    if f.startswith("rs") and f[2:].isdigit():
        n = int(f[2:])
        c, mk = p.c, p.market["adj_close"]
        if t - n < 0:
            return np.nan
        return float(c.iat[t, j] / c.iat[t - n, j] - 1 - (mk.iat[t] / mk.iat[t - n] - 1))
    return np.nan
