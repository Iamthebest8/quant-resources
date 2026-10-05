"""Leader capture, false positives, matched-control placebo, leader optionality (Parts 31-35)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from engine.features import cs_pct
from research.labels import period_label
from strategy.config import StrategyConfig
from strategy.engine import run_campaign_alone
from strategy.metrics import campaign_metrics
from strategy.signals import Arrays


# ---------------------------------------------------------------------------
# Leader episodes & capture (Part 32)
# ---------------------------------------------------------------------------
def leader_episodes(p, L: dict, thr: float, start: str = config.RESEARCH_START,
                    end: str = config.BACKTEST_END, gap: int = 10) -> pd.DataFrame:
    flag = (L["fwd_maxc_40"] >= thr) & p.universe
    flag = flag.loc[(flag.index >= pd.Timestamp(start)) & (flag.index <= pd.Timestamp(end))]
    pos = {d: i for i, d in enumerate(p.dates)}
    c = p.c
    rows = []
    for sid in flag.columns:
        days = flag.index[flag[sid].to_numpy()]
        if len(days) == 0:
            continue
        idx = np.array([pos[d] for d in days])
        starts = [idx[0]] + [b for a, b in zip(idx[:-1], idx[1:]) if b - a > gap]
        for t0 in starts:
            if t0 + 1 >= len(p.dates):
                continue
            entry = p.o.iloc[t0 + 1][sid]
            seg = c[sid].iloc[t0 + 1:t0 + 41]
            if seg.dropna().empty or not np.isfinite(entry):
                continue
            pk = seg.idxmax()
            rows.append({"stock_id": sid, "threshold": thr, "start_date": p.dates[t0], "start_px": entry,
                         "peak_date": pk, "peak_gain": seg.max() / entry - 1,
                         "days_to_peak": pos[pk] - t0, "sector": p.sector.get(sid, "UNKNOWN"),
                         "atr_pct_at_start": np.nan})
    ep = pd.DataFrame(rows)
    if len(ep):
        ep["period"] = period_label(ep["start_date"])
        ep["year"] = ep["start_date"].dt.year
    return ep


def capture(ep: pd.DataFrame, camps: pd.DataFrame, dates: pd.DatetimeIndex, lead: int = 15,
            label: str = "trade_level") -> pd.DataFrame:
    """Mark each episode captured if a probe was entered early (before half of the move)."""
    if ep.empty:
        return ep
    pos = {d: i for i, d in enumerate(dates)}
    by_stock = {s: g for s, g in camps.groupby("stock_id")} if len(camps) else {}
    out = []
    for e in ep.itertuples():
        g = by_stock.get(e.stock_id)
        s0 = pos[e.start_date]
        lo = dates[max(s0 - lead, 0)]
        rec = {"early_probe": False, "any_probe": False, "added_before_peak": False, "profitable": False,
               "probe_date": pd.NaT, "probe_px": np.nan, "campaign_ret": np.nan}
        if g is not None:
            w = g[(g["probe_date"] >= lo) & (g["probe_date"] < e.peak_date)]
            if len(w):
                rec["any_probe"] = True
                early = w[w["probe_price_adj"] <= e.start_px * (1 + e.threshold / 2)]
                if len(early):
                    f = early.iloc[0]
                    rec.update(early_probe=True, probe_date=f["probe_date"], probe_px=f["probe_price"],
                               campaign_ret=f["ret_on_invested"],
                               added_before_peak=bool(pd.notna(f["add_date"]) and f["add_date"] <= e.peak_date),
                               profitable=bool(f["pnl"] > 0))
        out.append(rec)
    res = pd.concat([ep.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    res["source"] = label
    return res


def random_capture_baseline(ep: pd.DataFrame, camps: pd.DataFrame, universe: pd.DataFrame, lead: int = 15) -> float:
    """Expected early-ish capture rate if the same number of probes were spread at random."""
    if ep.empty or camps.empty:
        return np.nan
    starts = camps.groupby("probe_date").size()
    usize = universe.sum(axis=1)
    dates = universe.index
    vals = []
    for e in ep.itertuples():
        s0 = dates.searchsorted(e.start_date)
        lo = dates[max(s0 - lead, 0)]
        n = starts[(starts.index >= lo) & (starts.index < e.peak_date)].sum()
        u = usize.loc[lo:e.peak_date].mean()
        vals.append(1 - (1 - 1 / max(u, 1)) ** n)
    return float(np.mean(vals))


def capture_summary(cap: pd.DataFrame, baselines: dict | None = None) -> pd.DataFrame:
    rows = []
    if cap.empty:
        return pd.DataFrame()
    for (src, thr), g0 in cap.groupby(["source", "threshold"]):
        for pname, g in [("ALL", g0)] + [(f"Y{y}", gg) for y, gg in g0.groupby("year")] + \
                        [(p_, gg) for p_, gg in g0.groupby("period")]:
            rows.append({"source": src, "threshold": f">={int(thr * 100)}%", "period": pname, "episodes": len(g),
                         "early_capture_rate": g["early_probe"].mean(), "any_probe_rate": g["any_probe"].mean(),
                         "added_before_peak_rate": g["added_before_peak"].mean(),
                         "profitable_capture_rate": g["profitable"].mean(),
                         "random_baseline": (baselines or {}).get((src, thr, pname), np.nan)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# False positives (Part 33)
# ---------------------------------------------------------------------------
FP_FEATURES = ["disc_pct", "rank_score", "gap_signal", "ret_1", "ext_ma20", "ret_20", "ret_60", "atr_pct", "beta",
               "hv20", "vol_ratio", "val_accel", "secmkt_10", "srs_10", "cex_10", "outp_10", "is_cnt10", "dr60",
               "up60", "rs_accel", "dist_h250", "tight10", "val20", "m_ret"]


def signal_features(p, F: dict, conds: dict, camps: pd.DataFrame) -> pd.DataFrame:
    if camps.empty:
        return camps
    d_idx = {d: i for i, d in enumerate(p.dates)}
    j_idx = {s: i for i, s in enumerate(p.ids)}
    ti = camps["signal_date"].map(d_idx).to_numpy()
    ji = camps["stock_id"].map(j_idx).to_numpy()
    out = camps.copy()
    gap = (p.raw_o / p.raw_c.shift(1) - 1).to_numpy()
    for f in FP_FEATURES:
        if f == "disc_pct":
            arr = conds["disc_pct"].to_numpy()
        elif f == "rank_score":
            arr = conds["rank_score"].to_numpy()
        elif f == "gap_signal":
            arr = gap
        elif f == "m_ret":
            out[f] = F["mkt"]["m_ret"].to_numpy()[ti]
            continue
        else:
            arr = F[f].to_numpy()
        out[f] = arr[ti, ji]
    out["regime"] = F["mkt"]["regime"].to_numpy()[ti]
    out["limit_up_signal_day"] = out["ret_1"] >= 0.095
    out["sector_driven"] = out["secmkt_10"] > out["srs_10"]
    out["recent_disposition"] = [bool(p.disp.iloc[max(t - 20, 0):t + 1, j].any()) for t, j in zip(ti, ji)]
    return out


def false_positive_analysis(sf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if sf.empty:
        return pd.DataFrame()
    sf = sf.copy()
    sf["outcome"] = np.where(sf["final_state"] == "FAILED_PROBE", "FALSE_PROBE",
                             np.where(sf["ret_on_invested"] >= 0.20, "BIG_WINNER", "OTHER"))
    sf["period"] = period_label(sf["probe_date"])
    for pname, g in [("ALL", sf)] + list(sf.groupby("period")):
        fp, tp = g[g["outcome"] == "FALSE_PROBE"], g[g["outcome"] == "BIG_WINNER"]
        for f in FP_FEATURES:
            a, b = fp[f].astype(float).dropna(), tp[f].astype(float).dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            sd = np.sqrt((a.var() + b.var()) / 2)
            rows.append({"section": "feature_SMD_false_vs_bigwinner", "period": pname, "item": f,
                         "false_probe_median": a.median(), "big_winner_median": b.median(),
                         "all_median": g[f].astype(float).median(),
                         "smd": (a.mean() - b.mean()) / sd if sd > 0 else np.nan, "n_false": len(a), "n_win": len(b)})
        for cat in ("regime", "limit_up_signal_day", "sector_driven", "recent_disposition"):
            for v, gg in g.groupby(cat):
                rows.append({"section": f"false_rate_by_{cat}", "period": pname, "item": str(v), "n": len(gg),
                             "false_probe_rate": (gg["outcome"] == "FALSE_PROBE").mean(),
                             "big_winner_rate": (gg["outcome"] == "BIG_WINNER").mean(),
                             "avg_ret": gg["ret_on_invested"].mean()})
        for f, bins in (("ext_ma20", [-1, 0.05, 0.10, 0.15, 0.25, 9]), ("gap_signal", [-1, 0, 0.02, 0.05, 0.095, 9]),
                        ("ret_60", [-9, 0, 0.2, 0.4, 0.8, 99]), ("atr_pct", [0, 0.025, 0.035, 0.05, 0.07, 9])):
            for v, gg in g.groupby(pd.cut(g[f].astype(float), bins), observed=True):
                rows.append({"section": f"false_rate_by_{f}", "period": pname, "item": str(v), "n": len(gg),
                             "false_probe_rate": (gg["outcome"] == "FALSE_PROBE").mean(),
                             "big_winner_rate": (gg["outcome"] == "BIG_WINNER").mean(),
                             "avg_ret": gg["ret_on_invested"].mean()})
    return pd.DataFrame(rows)


def strong_but_failed(R: pd.DataFrame) -> pd.DataFrame:
    """Label-based: top-5% Discovery-score stock-days that then fell -5% / -10% within 20D."""
    if "disc_pct" not in R.columns:
        return pd.DataFrame()
    top = R[R["disc_pct"] >= 0.95].copy()
    top["fp5"] = top["fwd_ret_20"] <= -0.05
    top["fp10"] = top["fwd_ret_20"] <= -0.10
    top["leader"] = top["leader20"] > 0
    rows = []
    feats = ["ext_ma20", "ret_60", "atr_pct", "beta", "vol_ratio", "secmkt_10", "srs_10", "cex_10", "is_cnt10",
             "dr60", "up60", "rs_accel", "gap"]
    for pname, g in [("ALL", top)] + list(top.groupby("period")):
        rows.append({"section": "strong_then_fall_rates", "period": pname, "item": "top5pct_disc", "n": len(g),
                     "fall5_rate": g["fp5"].mean(), "fall10_rate": g["fp10"].mean(),
                     "leader20_rate": g["leader"].mean()})
        for f in feats:
            if f not in g.columns:
                continue
            a, b = g.loc[g["fp10"], f].dropna(), g.loc[g["leader"], f].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            sd = np.sqrt((a.var() + b.var()) / 2)
            rows.append({"section": "strong_then_fall10_vs_leader_SMD", "period": pname, "item": f,
                         "fall10_median": a.median(), "leader_median": b.median(),
                         "smd": (a.mean() - b.mean()) / sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Matched-control placebo (Part 35)
# ---------------------------------------------------------------------------
def matched_controls(p, F: dict, A: Arrays, conds: dict, camps: pd.DataFrame, cfg: StrategyConfig,
                     end_idx: int, slip: float, cost: float, k: int = 3, seed: int = 11) -> pd.DataFrame:
    """For each probe: up to k control stock-days on the SAME date (same market regime), same ATR,
    beta, liquidity terciles and RS20 quintile, preferring same sector, that did NOT fire a probe;
    then run the identical probe/fail/confirm/add/exit mechanics from the control."""
    rng = np.random.default_rng(seed)
    U = p.universe
    atr_t = np.ceil(cs_pct(F["atr_pct"], U).clip(1e-9, 1) * 3).to_numpy()
    beta_t = np.ceil(cs_pct(F["beta"], U).clip(1e-9, 1) * 3).to_numpy()
    liq_t = np.ceil(cs_pct(F["val20"], U).clip(1e-9, 1) * 3).to_numpy()
    rs_q = np.ceil(F["rs_pct_20"].clip(1e-9, 1) * 5).to_numpy()
    sig = conds["probe_signal"].to_numpy()
    ok_stop = conds["c_stop_ok"].to_numpy() & conds["c_no_disposition"].to_numpy() & U.to_numpy()
    sec = p.sector.reindex(p.ids).to_numpy()
    d_idx = {d: i for i, d in enumerate(p.dates)}
    j_idx = {s: i for i, s in enumerate(p.ids)}
    recs = []
    for c in camps.itertuples():
        t, j = d_idx[c.signal_date], j_idx[c.stock_id]
        pool = np.where(ok_stop[t] & ~sig[t] & (atr_t[t] == atr_t[t, j]) & (beta_t[t] == beta_t[t, j])
                        & (liq_t[t] == liq_t[t, j]) & (rs_q[t] == rs_q[t, j]))[0]
        pool = pool[pool != j]
        if len(pool) == 0:
            continue
        same = pool[sec[pool] == sec[j]]
        pick = same if len(same) >= 1 else pool
        for jj in rng.choice(pick, size=min(k, len(pick)), replace=False):
            cp = run_campaign_alone(A, int(jj), t, cfg, end_idx, slip, cost)
            if cp is not None and cp.legs:
                r = cp.record()
                r["matched_to"] = c.stock_id
                r["same_sector"] = bool(sec[jj] == sec[j])
                recs.append(r)
    return pd.DataFrame(recs)


def optionality_table(camps: pd.DataFrame, cap: pd.DataFrame | None = None) -> pd.DataFrame:
    """LEADER_OPTIONALITY (Part 31): small probe losses paid vs early leader exposure obtained."""
    rows = []
    if camps.empty:
        return pd.DataFrame()
    camps = camps.copy()
    camps["period"] = period_label(camps["probe_date"])
    camps["year"] = camps["probe_date"].dt.year
    groups = [("ALL", camps)] + [(f"Y{y}", g) for y, g in camps.groupby("year")] + list(camps.groupby("period"))
    for pname, g in groups:
        m = campaign_metrics(g)
        failed = g[g["final_state"] == "FAILED_PROBE"]
        win = g[g["ret_on_invested"] >= 0.20]
        lc = np.nan
        if cap is not None and len(cap):
            cc = cap[(cap["threshold"] == 0.3)]
            if pname.startswith("Y"):
                cc = cc[cc["year"] == int(pname[1:])]
            elif pname != "ALL":
                cc = cc[cc["period"] == pname]
            lc = cc["early_probe"].mean() if len(cc) else np.nan
        rows.append({"period": pname, "probe_count": len(g), "failure_rate": m["false_probe_rate"],
                     "confirmation_rate": m["confirmation_rate"], "add_rate": m["add_rate"],
                     "avg_failed_loss_ret": m["avg_failed_loss_ret"], "avg_failed_loss_slots": m["avg_failed_loss_slots"],
                     "total_failed_cost_slots": failed["pnl_slots"].sum(),
                     "avg_successful_winner_ret": win["ret_on_invested"].mean() if len(win) else np.nan,
                     "n_winners_ge20": len(win), "winners_pnl_slots": win["pnl_slots"].sum(),
                     "optionality_ratio": (win["pnl_slots"].sum() / -failed["pnl_slots"].sum())
                     if len(failed) and failed["pnl_slots"].sum() < 0 else np.nan,
                     "leader_capture_rate_ge30": lc, "pf": m["pf"], "payoff": m["payoff"],
                     "ev_slots_per_probe": m["ev_slots_per_probe"]})
    return pd.DataFrame(rows)
