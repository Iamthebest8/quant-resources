"""Weinstein stage analysis engine: weekly stage classifier, group / market stages, and setup generators
for W1 (stage 1->2 breakout), W2 (stage-2 continuation), W3 (post-breakout pullback), S1 (stage 3->4
breakdown short) and S2 (light-volume rebound short).

Every constant lives in WRULES and is documented in WEINSTEIN_QUANT_RULES.md with a BOOK / RESEARCH tag.
PIT: weekly values are computed at the close of the week's last trading day (t); orders live during the
following week (t+1 ...). Nothing at t uses data after t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.weekly import week_end_index

WRULES = {
    # stage classifier
    "MA_W": 30, "MA_MIN": 26, "SLOPE_LAG": 4, "FLAT_BAND": 0.01,
    # RS
    "MRS_BASE": 52, "RS_SLOPE_LONG": 13,
    # volume (book p.104 / p.105-106)
    "VOL_W_MULT": 2.0, "VOL_W_PRIOR": 4, "VOL_BUILD_RECENT": 3, "VOL_BUILD_PRIOR": 8, "VOL_D_MULT": 2.0,
    "PULLBACK_VOL_TEXTBOOK": 0.25, "PULLBACK_VOL_MODERN": 0.50, "REBOUND_VOL_SHORT": 0.50,
    # overhead
    "OH_WINDOW": 130, "OH_BAND": 0.15, "OH_FAIL": 4,
    # W1
    "BASE_MIN_W": 8, "BASE_CAP_W": 52, "W1_SLOPE_MIN": -0.01,
    # W2
    "W2_AGE_MIN": 8, "W2_SLOPE_MIN": 0.02, "W2_WIN": 6, "W2_RANGE_MAX": 0.25, "W2_LO_MIN": 0.97, "W2_LO_MAX": 1.10,
    # W3 / S2 follow-ups
    "PB_START": 3, "PB_END": 60, "PB_ZONE": 1.05, "RB_ZONE": 0.97,
    # S1
    "TOP_MIN_W": 6, "S1_SLOPE_MAX": 0.01,
    # orders / stops
    "TRIG_PAD": 0.003, "LIMIT_PAD": 0.02, "STOP_PAD": 0.01, "STOP_CAP": 0.15, "LOW_LOOK_W": 4,
    # modernized
    "MOD_RS_PCT": 0.70, "MOD_SUPPLY15": 0.15, "MOD_STOP_ATR_MAX": 3.0, "MOD_STOP_ATR": 2.5, "MOD_W3_STOP_ATR": 2.0,
    "MOD_BREADTH": 0.50, "MOD_BE_R": 2.0,
}


# ---------------------------------------------------------------------------------------------
# weekly data
# ---------------------------------------------------------------------------------------------
@dataclass
class WeeklyData:
    week_end: pd.DatetimeIndex          # week-end trading days
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    vol: pd.DataFrame                   # average daily volume in the week (disposition days excluded)
    mkt: pd.Series


def weekly_data(p) -> WeeklyData:
    we = week_end_index(p.dates)
    key = we.to_numpy()
    c = p.c.groupby(key).last()
    h = p.h.groupby(key).max()
    lo = p.l.groupby(key).min()
    v = p.vol.where(p.c.notna() & ~p.disp).groupby(key).mean()
    c = c.where(h.notna())
    mkt = p.market["adj_close"].groupby(key).last()
    idx = pd.DatetimeIndex(c.index)
    for x in (c, h, lo, v, mkt):
        x.index = idx
    return WeeklyData(idx, c, h, lo, v, mkt)


# ---------------------------------------------------------------------------------------------
# stage classifier
# ---------------------------------------------------------------------------------------------
def classify(close: np.ndarray, ma: np.ndarray, slope: np.ndarray, high: np.ndarray | None = None,
             low: np.ndarray | None = None, band: float = WRULES["FLAT_BAND"]) -> dict[str, np.ndarray]:
    """close/ma/slope: (weeks x N). Returns stage (0 unknown,1..4), sub (True = 2B / 4B), ep_len,
    episode max/min close and max high / min low (INCLUDING the current week)."""
    W, N = close.shape
    high = close if high is None else high
    low = close if low is None else low
    stage = np.zeros((W, N), np.int8)
    sub = np.zeros((W, N), bool)
    ep_len = np.zeros((W, N), np.int16)
    ep_maxc = np.full((W, N), np.nan)
    ep_minc = np.full((W, N), np.nan)
    ep_maxh = np.full((W, N), np.nan)
    ep_minl = np.full((W, N), np.nan)
    st = np.zeros(N, np.int8)
    ln = np.zeros(N, np.int16)
    mxc = np.full(N, np.nan)
    mnc = np.full(N, np.nan)
    mxh = np.full(N, np.nan)
    mnl = np.full(N, np.nan)
    for i in range(W):
        c, m, s = close[i], ma[i], slope[i]
        ok = np.isfinite(c) & np.isfinite(m) & np.isfinite(s)
        rising, falling = s > band, s < -band
        flat = ~rising & ~falling
        above, below = c > m, c < m
        new = st.copy()
        # unknown -> first classification
        u = ok & (st == 0)
        new[u & rising & above] = 2
        new[u & falling & below] = 4
        new[u & ~(rising & above) & ~(falling & below)] = 1
        # from 1
        a = ok & (st == 1)
        new[a & rising & above & (c > mxc)] = 2
        new[a & falling & below & (c < mnc)] = 4
        # from 2
        a = ok & (st == 2)
        new[a & (flat | falling)] = 3
        # from 3
        a = ok & (st == 3)
        new[a & rising & above & (c > mxc)] = 2
        new[a & falling & below & (c < mnc)] = 4
        # from 4
        a = ok & (st == 4)
        new[a & (flat | rising)] = 1
        sb = ok & (((new == 2) & ~above) | ((new == 4) & ~below))
        changed = ok & (new != st)
        ln = np.where(changed, 1, np.where(ok, ln + 1, ln)).astype(np.int16)
        hh, ll = high[i], low[i]
        mxc = np.where(changed, c, np.where(ok, np.fmax(mxc, c), mxc))
        mnc = np.where(changed, c, np.where(ok, np.fmin(mnc, c), mnc))
        mxh = np.where(changed, hh, np.where(ok, np.fmax(mxh, hh), mxh))
        mnl = np.where(changed, ll, np.where(ok, np.fmin(mnl, ll), mnl))
        st = np.where(ok, new, st).astype(np.int8)
        stage[i] = np.where(ok, st, 0)
        sub[i] = sb
        ep_len[i] = np.where(ok, ln, 0)
        ep_maxc[i], ep_minc[i], ep_maxh[i], ep_minl[i] = mxc, mnc, mxh, mnl
    return {"stage": stage, "sub": sub, "ep_len": ep_len, "ep_maxc": ep_maxc, "ep_minc": ep_minc,
            "ep_maxh": ep_maxh, "ep_minl": ep_minl}


def stage_frame(close_w: pd.DataFrame, high_w=None, low_w=None) -> dict[str, pd.DataFrame]:
    ma = close_w.rolling(WRULES["MA_W"], min_periods=WRULES["MA_MIN"]).mean()
    slope = ma / ma.shift(WRULES["SLOPE_LAG"]) - 1
    res = classify(close_w.to_numpy(float), ma.to_numpy(float), slope.to_numpy(float),
                   None if high_w is None else high_w.to_numpy(float), None if low_w is None else low_w.to_numpy(float))
    out = {k: pd.DataFrame(v, index=close_w.index, columns=close_w.columns) for k, v in res.items()}
    out["ma30w"] = ma
    out["slope4"] = slope
    out["ma10w"] = close_w.rolling(10, min_periods=9).mean()
    return out


def group_indices(p, U: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight daily-return index per FinMind industry_category (members = universe as of yesterday)."""
    r = p.c / p.c.shift(1) - 1
    rU = r.where(U.shift(1, fill_value=False))
    sec = p.sector.reindex(p.ids).fillna("UNKNOWN")
    out = {}
    for name, cols in sec.groupby(sec).groups.items():
        if name == "UNKNOWN":
            continue
        sub = rU[list(cols)]
        n = sub.notna().sum(axis=1)
        ret = sub.mean(axis=1).where(n >= 5)
        lvl = np.exp(np.log1p(ret.fillna(0)).cumsum()).where(n >= 5)
        out[name] = lvl
    return pd.DataFrame(out, index=p.dates)


# ---------------------------------------------------------------------------------------------
# full weekly context
# ---------------------------------------------------------------------------------------------
@dataclass
class WContext:
    wd: WeeklyData
    S: dict                      # stock stage frames (weekly)
    rs: pd.DataFrame
    mrs: pd.DataFrame
    rs_slope4: pd.DataFrame
    rs_slope13: pd.DataFrame
    vol_strong_w: pd.DataFrame   # bool, for the week ending at index
    mkt: dict                    # TAIEX stage frames (single column)
    grp_stage_w: pd.DataFrame    # per stock: its group's stage (weekly)
    grp_na: pd.DataFrame
    breadth12: pd.Series
    grp_stage2_pct: pd.DataFrame
    allmax_h: pd.DataFrame       # expanding max weekly high


def build_context(p, log=print) -> WContext:
    R = WRULES
    wd = weekly_data(p)
    log(f"[weinstein] weekly bars {wd.close.shape}")
    S = stage_frame(wd.close, wd.high, wd.low)
    rs = wd.close.div(wd.mkt, axis=0)
    mrs = rs / rs.rolling(R["MRS_BASE"], min_periods=40).mean() - 1
    rs_s4 = rs / rs.shift(4) - 1
    rs_s13 = rs / rs.shift(R["RS_SLOPE_LONG"]) - 1
    v = wd.vol
    prior4 = v.shift(1).rolling(R["VOL_W_PRIOR"], min_periods=3).mean()
    rec3 = v.rolling(R["VOL_BUILD_RECENT"], min_periods=3).mean()
    prior8 = v.shift(R["VOL_BUILD_RECENT"]).rolling(R["VOL_BUILD_PRIOR"], min_periods=6).mean()
    vs = (v >= R["VOL_W_MULT"] * prior4) | ((rec3 >= R["VOL_W_MULT"] * prior8) & (v > v.shift(1)))
    vs = vs.where(v.notna() & prior4.notna())
    # market
    m = wd.mkt.to_frame("TAIEX")
    mk = stage_frame(m)
    # groups
    U = p.universe
    gi = group_indices(p, U)
    key = week_end_index(p.dates).to_numpy()
    gw = gi.groupby(key).last()
    gw.index = wd.week_end
    gS = stage_frame(gw)
    sec = p.sector.reindex(p.ids).fillna("UNKNOWN")
    g_stage = pd.DataFrame(np.nan, index=wd.week_end, columns=p.ids)
    g_na = pd.DataFrame(True, index=wd.week_end, columns=p.ids)
    gst = gS["stage"].where(gS["stage"] > 0)
    for name in gst.columns:
        cols = list(sec.index[sec == name])
        g_stage[cols] = np.repeat(gst[[name]].to_numpy(), len(cols), axis=1)
        g_na[cols] = np.repeat(gst[[name]].isna().to_numpy(), len(cols), axis=1)
    # breadth (universe at week end)
    Uw = U.reindex(wd.week_end).fillna(False).astype(bool)
    st = S["stage"]
    in12 = st.isin([1, 2]) & (st > 0)
    breadth = (in12 & Uw).sum(axis=1) / ((st > 0) & Uw).sum(axis=1).replace(0, np.nan)
    st2 = (st == 2) & Uw
    g2 = pd.DataFrame(np.nan, index=wd.week_end, columns=p.ids)
    for name, cols in sec.groupby(sec).groups.items():
        cols = list(cols)
        n = Uw[cols].sum(axis=1)
        pct = (st2[cols].sum(axis=1) / n.replace(0, np.nan)).where(n >= 5)
        g2[cols] = np.repeat(pct.to_numpy()[:, None], len(cols), axis=1)
    log(f"[weinstein] stages done; TAIEX latest stage={int(mk['stage'].iloc[-1, 0])}")
    return WContext(wd, S, rs, mrs, rs_s4, rs_s13, vs, mk, g_stage, g_na, breadth, g2, wd.high.cummax())


def overhead_count(close_w: np.ndarray, R: np.ndarray, window: int, band: float) -> np.ndarray:
    """# weekly closes in the previous `window` weeks (excluding the current) in (R, R*(1+band)]."""
    W, N = close_w.shape
    out = np.full((W, N), np.nan)
    for i in range(W):
        lo = max(0, i - window)
        if i - lo < 26:
            continue
        blk = close_w[lo:i]
        r = R[i]
        out[i] = ((blk > r) & (blk <= r * (1 + band))).sum(axis=0)
    out[~np.isfinite(R)] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# setups
# ---------------------------------------------------------------------------------------------
def _daily_pos(p, wk_index: pd.DatetimeIndex) -> np.ndarray:
    return p.dates.get_indexer(wk_index)


def weekly_setups(p, F, G, ctx: WContext, log=print) -> pd.DataFrame:
    """W1, W2 and S1 setups (one row per stock-week satisfying the textbook rules; modernized filters are
    stored as columns so the MODERNIZED variant is a strict subset/variation of the same events)."""
    R = WRULES
    S = ctx.S
    wd = ctx.wd
    we = wd.week_end
    ti = _daily_pos(p, we)
    st = S["stage"].to_numpy()
    sub = S["sub"].to_numpy()
    eplen = S["ep_len"].to_numpy()
    ma = S["ma30w"].to_numpy()
    sl = S["slope4"].to_numpy()
    C = wd.close.to_numpy()
    H = wd.high.to_numpy()
    L = wd.low.to_numpy()
    hi52 = wd.high.rolling(R["BASE_CAP_W"], min_periods=1).max().to_numpy()
    lo52 = wd.low.rolling(R["BASE_CAP_W"], min_periods=1).min().to_numpy()
    Rb = np.fmin(S["ep_maxh"].to_numpy(), hi52)          # base resistance (<= 52 weeks)
    Sb = np.fmax(S["ep_minl"].to_numpy(), lo52)          # base / top support (<= 52 weeks)
    low4 = wd.low.rolling(R["LOW_LOOK_W"], min_periods=2).min().to_numpy()
    high4 = wd.high.rolling(R["LOW_LOOK_W"], min_periods=2).max().to_numpy()
    k = R["W2_WIN"]
    R6 = wd.high.rolling(k, min_periods=k).max().to_numpy()
    max2 = wd.high.rolling(2, min_periods=2).max().to_numpy()
    Lo6 = wd.low.rolling(k, min_periods=k).min().to_numpy()
    mrs = ctx.mrs.to_numpy()
    rs4 = ctx.rs_slope4.to_numpy()
    rs13 = ctx.rs_slope13.to_numpy()
    rs_long = (rs13 >= 0) & ((mrs >= 0) | (rs4 > 0))
    rs_short = (mrs < 0) | (rs13 < 0)
    mkt_st = ctx.mkt["stage"].to_numpy()[:, 0]
    g_st = ctx.grp_stage_w.to_numpy()
    g_na = ctx.grp_na.to_numpy()
    g_long = g_na | np.isin(g_st, [1, 2])
    g_short = (~g_na) & np.isin(g_st, [3, 4])
    allmax = ctx.allmax_h.to_numpy()
    # daily matrices sampled at week ends
    U = p.universe.to_numpy()[ti]
    nodisp = (~(p.disp | p.disp_next)).to_numpy()[ti]
    atr = F["atr"].to_numpy()[ti]
    rs120 = None
    if "ret_120" in F:
        rs120 = F["ret_120"]
    ex120 = (p.c / p.c.shift(120) - 1).sub(p.market["adj_close"] / p.market["adj_close"].shift(120) - 1, axis=0)
    ex120_pct = ex120.where(p.universe).rank(axis=1, pct=True).to_numpy()[ti]
    supply15 = G["supply15"].to_numpy()[ti] if "supply15" in G else np.full_like(C, np.nan)
    res_dist = G["res_dist"].to_numpy()[ti] if "res_dist" in G else np.full_like(C, np.nan)
    swl10 = p.l.rolling(10, min_periods=8).min().to_numpy()[ti]
    swh10 = p.h.rolling(10, min_periods=8).max().to_numpy()[ti]
    mkt_c_d = p.market["adj_close"]
    mkt_ma150 = (mkt_c_d / mkt_c_d.rolling(150, min_periods=120).mean() - 1).to_numpy()[ti]
    breadth = ctx.breadth12.to_numpy()
    g2pct = ctx.grp_stage2_pct.to_numpy()
    n_days_next = np.r_[np.diff(ti), 5]       # trading days in the following week (live week: assume 5)

    rows = []
    # ---------------- W1 ----------------
    trigW1 = np.fmax(Rb, ma) * (1 + R["TRIG_PAD"])
    oh_w1 = overhead_count(C, trigW1, R["OH_WINDOW"], R["OH_BAND"])
    base_ok = (st == 1) & (eplen >= R["BASE_MIN_W"]) & (sl >= R["W1_SLOPE_MIN"]) & (C <= Rb)
    w1 = base_ok & rs_long & (oh_w1 < R["OH_FAIL"]) & np.isin(mkt_st, [1, 2])[:, None] & g_long & U & nodisp
    # ---------------- W2 ----------------
    trigW2 = R6 * (1 + R["TRIG_PAD"])
    oh_w2 = overhead_count(C, trigW2, R["OH_WINDOW"], R["OH_BAND"])
    cons = ((st == 2) & (eplen >= R["W2_AGE_MIN"]) & (sl >= R["W2_SLOPE_MIN"]) & (max2 < R6)
            & (R6 / Lo6 - 1 <= R["W2_RANGE_MAX"]) & (Lo6 >= ma * R["W2_LO_MIN"]) & (Lo6 <= ma * R["W2_LO_MAX"])
            & (C <= R6))
    w2 = cons & rs_long & (oh_w2 < R["OH_FAIL"]) & np.isin(mkt_st, [1, 2])[:, None] & g_long & U & nodisp
    # ---------------- S1 ----------------
    trigS1 = np.fmin(Sb, ma) * (1 - R["TRIG_PAD"])
    top_ok = (st == 3) & (eplen >= R["TOP_MIN_W"]) & (sl <= R["S1_SLOPE_MAX"]) & (C >= Sb)
    s1_core = top_ok & rs_short & g_short & U & nodisp
    s1 = s1_core & np.isin(mkt_st, [3, 4])[:, None]
    s1_mod = s1_core & (np.isin(mkt_st, [3, 4])[:, None] | (mkt_ma150 < 0)[:, None])

    def emit(mask, engine, side, trig, stop, level, oh, extra_mask=None):
        wi, js = np.nonzero(mask if extra_mask is None else (mask | extra_mask))
        for i, j in zip(wi, js):
            if n_days_next[i] <= 0:
                continue
            t = ti[i]
            a = atr[i, j]
            rows.append({
                "engine": engine, "side": side, "j": j, "stock_id": p.ids[j], "t": t, "setup_date": p.dates[t],
                "week_i": i, "order_days": int(n_days_next[i]),
                "trigger": trig[i, j], "limit": trig[i, j] * (1 + side * R["LIMIT_PAD"]), "stop": stop[i, j],
                "level": level[i, j], "textbook": bool(mask[i, j]),
                "stage": int(st[i, j]), "stage_sub": bool(sub[i, j]), "ep_len": int(eplen[i, j]), "ma30w": ma[i, j],
                "slope4": sl[i, j], "mrs": mrs[i, j], "rs_slope4": rs4[i, j], "rs_slope13": rs13[i, j],
                "oh15": oh[i, j], "virgin": bool(trig[i, j] >= allmax[i, j] * 0.999) if side > 0 else np.nan,
                "base_depth": Rb[i, j] / Sb[i, j] - 1 if side > 0 else Rb[i, j] / Sb[i, j] - 1,
                "mkt_stage": int(mkt_st[i]), "grp_stage": g_st[i, j], "grp_na": bool(g_na[i, j]),
                "breadth12": breadth[i], "grp_stage2_pct": g2pct[i, j], "atr": a,
                "ex120_pct": ex120_pct[i, j], "supply15": supply15[i, j], "res_dist": res_dist[i, j],
                "swl10": swl10[i, j], "swh10": swh10[i, j], "mkt_ma150_gap": mkt_ma150[i],
                "close_w": C[i, j],
            })

    cap = R["STOP_CAP"]
    stopW1 = np.fmax(low4 * (1 - R["STOP_PAD"]), trigW1 * (1 - cap))
    stopW2 = np.fmax(Lo6 * (1 - R["STOP_PAD"]), trigW2 * (1 - cap))
    stopS1 = np.fmin(high4 * (1 + R["STOP_PAD"]), trigS1 * (1 + cap))
    # modernized long candidates: same structural setup, alternative filters (market OR breadth)
    mkt_or_breadth = (np.isin(mkt_st, [1, 2]) | (breadth >= R["MOD_BREADTH"]))[:, None]
    w1_mod = base_ok & mkt_or_breadth & g_long & U & nodisp
    w2_mod = cons & mkt_or_breadth & g_long & U & nodisp
    emit(w1, "W1", 1, trigW1, stopW1, Rb, oh_w1, extra_mask=w1_mod)
    emit(w2, "W2", 1, trigW2, stopW2, R6, oh_w2, extra_mask=w2_mod)
    emit(s1, "S1", -1, trigS1, stopS1, Sb, np.full_like(C, np.nan), extra_mask=s1_mod)
    df = pd.DataFrame(rows)
    if len(df):
        df["modern_base"] = True   # structural setup holds (filters for MODERN applied later)
        df.loc[df["engine"] == "S1", "modern_mkt_ok"] = True
        log(f"[weinstein] weekly setups: " + ", ".join(f"{e}={n}" for e, n in df["engine"].value_counts().items()))
    return df


def breakout_events(p, setups: pd.DataFrame, engine: str) -> pd.DataFrame:
    """For W1 (or S1) setups: did price cross the trigger during the order week? -> first crossing day b."""
    s = setups[(setups["engine"] == engine)]
    H, L = p.h.to_numpy(), p.l.to_numpy()
    out = []
    for r in s.itertuples():
        t, j, n = r.t, r.j, r.order_days
        seg = slice(t + 1, t + 1 + n)
        x = H[seg, j] if r.side > 0 else L[seg, j]
        hit = (x >= r.trigger) if r.side > 0 else (x <= r.trigger)
        if hit.any():
            b = t + 1 + int(np.argmax(hit))
            out.append({"setup_row": r.Index, "j": j, "stock_id": r.stock_id, "b": b, "level": r.level,
                        "trigger": r.trigger, "textbook": r.textbook, "side": r.side})
    ev = pd.DataFrame(out)
    if len(ev):     # one breakout per stock per base: drop repeated crossings within 20 trading days
        ev = ev.sort_values(["j", "b"])
        keep = []
        last = {}
        for r in ev.itertuples():
            if r.j in last and r.b - last[r.j] <= 20:
                continue
            last[r.j] = r.b
            keep.append(r.Index)
        ev = ev.loc[keep].reset_index(drop=True)
    return ev


def followup_setups(p, F, ctx: WContext, ev: pd.DataFrame, kind: str, log=print) -> pd.DataFrame:
    """W3 (kind='W3', from W1 breakouts) or S2 (kind='S2', from S1 breakdowns). One signal per event:
    the first day d in [b+PB_START, b+PB_END] satisfying the rules; entry next open."""
    R = WRULES
    if ev is None or len(ev) == 0:
        return pd.DataFrame()
    side = 1 if kind == "W3" else -1
    Hd, Ld, Cd = p.h.to_numpy(), p.l.to_numpy(), p.c.to_numpy()
    vol = p.vol.where(~p.disp).to_numpy()
    v5 = pd.DataFrame(vol).rolling(5, min_periods=4).mean().to_numpy()
    we = week_end_index(p.dates)
    is_we = (we.to_numpy() == p.dates.to_numpy())
    wpos = pd.Series(np.arange(len(ctx.wd.week_end)), index=ctx.wd.week_end)
    wk_of_day = wpos.reindex(pd.DatetimeIndex(we.to_numpy())).to_numpy()
    vs_w = ctx.vol_strong_w.to_numpy()
    sl_w = ctx.S["slope4"].to_numpy()
    ma_w = ctx.S["ma30w"].to_numpy()
    ma_d = ctx.S["ma30w"].reindex(ctx.S["ma30w"].index.union(p.dates)).ffill().reindex(p.dates).to_numpy()
    # slope known at the close of each day = last completed week's slope (or this week's if d is week end)
    def wk_known(d):
        k = wk_of_day[d]
        return int(k) if is_we[d] else int(k) - 1
    U = p.universe.to_numpy()
    nodisp = (~(p.disp | p.disp_next)).to_numpy()
    atr = F["atr"].to_numpy()
    T = len(p.dates)
    rows = []
    for r in ev.itertuples():
        j, b, lvl = r.j, r.b, r.level
        kb = int(wk_of_day[b])
        if kind == "W3" and not (vs_w[kb, j] == 1):        # breakout-week volume must be strong (book p.105)
            continue
        peak = np.nan
        for d in range(b, min(b + R["PB_END"], T - 1) + 1):
            peak = np.fmax(peak, v5[d, j])
            if d < b + R["PB_START"]:
                continue
            k = wk_known(d)
            if k < 0:
                continue
            # invalidation by weekly close on the wrong side of the level
            if is_we[d]:
                wc = Cd[d, j]
                if (side > 0 and wc < lvl) or (side < 0 and wc > lvl):
                    break
            c = Cd[d, j]
            if not np.isfinite(c):
                continue
            if side > 0:
                cond = (Ld[d, j] <= lvl * R["PB_ZONE"]) and (c >= lvl) and (sl_w[k, j] >= 0)
                vthr = R["PULLBACK_VOL_TEXTBOOK"]
            else:
                cond = (Hd[d, j] >= lvl * R["RB_ZONE"]) and (c <= lvl) and (c < ma_d[d, j]) and (sl_w[k, j] <= 0)
                vthr = R["REBOUND_VOL_SHORT"]
            if not cond or not (U[d, j] and nodisp[d, j]):
                continue
            vr = v5[d, j] / peak if peak > 0 else np.nan
            text_ok = np.isfinite(vr) and vr <= vthr
            mod_ok = np.isfinite(vr) and vr <= (R["PULLBACK_VOL_MODERN"] if side > 0 else R["REBOUND_VOL_SHORT"])
            if not (text_ok or mod_ok):
                continue
            lo10 = np.nanmin(Ld[max(0, d - 9):d + 1, j])
            hi10 = np.nanmax(Hd[max(0, d - 9):d + 1, j])
            if side > 0:
                stop = max(lo10 * (1 - R["STOP_PAD"]), c * (1 - R["STOP_CAP"]))
                stop_mod = max(lo10 * 0.995, c - R["MOD_W3_STOP_ATR"] * atr[d, j])
            else:
                stop = min(max(hi10, lvl) * (1 + R["STOP_PAD"]), c * (1 + R["STOP_CAP"]))
                stop_mod = min(stop, c + R["MOD_STOP_ATR"] * atr[d, j])
            rows.append({"engine": kind, "side": side, "j": j, "stock_id": p.ids[j], "t": d, "setup_date": p.dates[d],
                         "order_days": 1, "trigger": np.nan, "limit": np.nan, "stop": stop, "stop_mod": stop_mod,
                         "level": lvl, "textbook": bool(text_ok and r.textbook), "parent_b": b,
                         "parent_textbook": bool(r.textbook), "vol_ratio_pb": vr, "atr": atr[d, j],
                         "days_since_breakout": d - b, "ref_close": c})
            break
    df = pd.DataFrame(rows)
    log(f"[weinstein] {kind} follow-up setups: {len(df)} (textbook {int(df['textbook'].sum()) if len(df) else 0})")
    return df
