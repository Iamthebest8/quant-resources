"""Feature research & event studies (Parts 8-21, 34-35).

Every study is computed per period (DISCOVERY / STRICT_OOS / years). Only rows of the
training window (purged by LABEL_HORIZON so that labels never read prices after the
window end) are used for any *selection*; other periods are reported, never used to choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from engine.features import DISCOVERY_FAMILIES, FEATURE_FAMILIES, NEGATIVE

LEADER = "leader20"


def purged(R: pd.DataFrame, start: str, end: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows in [start, end] whose 40D label window ends on/before `end` (embargo)."""
    end_ts = pd.Timestamp(end)
    pos = dates.searchsorted(end_ts, side="right") - 1
    cutoff = dates[max(pos - config.LABEL_HORIZON, 0)]
    return R[(R["date"] >= pd.Timestamp(start)) & (R["date"] <= cutoff)]


def window(R: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return R[(R["date"] >= pd.Timestamp(start)) & (R["date"] <= pd.Timestamp(end))]


def outcome_stats(g: pd.DataFrame) -> pd.Series:
    x = g["fwd_ret_20"].dropna()
    mx = g["fwd_maxc_40"].dropna()
    return pd.Series({
        "n": len(g),
        "fwd20_mean": x.mean(), "fwd20_median": x.median(),
        "fwd20_ex_mean": g["fwd_ex_20"].mean(),
        "p_leader20": g["leader20"].mean(), "p_leader30": g["leader30"].mean(), "p_leader40": g["leader40"].mean(),
        "p_mae10": g["bad10"].mean(),
        "maxc40_p90": mx.quantile(0.9) if len(mx) else np.nan,
        "fwd20_p95": x.quantile(0.95) if len(x) else np.nan,
        "fwd20_p05": x.quantile(0.05) if len(x) else np.nan,
    })


def stratified_lift(df: pd.DataFrame, flag: pd.Series, label: str = LEADER, strata=("atr_t3",)) -> tuple:
    """P(label | flag, stratum) / P(label | stratum), averaged with stratum weights of flagged rows.

    Controls for volatility (ATR tercile by default): high-ATR names reach +20% more often
    mechanically, so raw lift would reward volatility, not leadership (Part 18/35).
    """
    d = df[list(strata) + [label]].copy()
    d["flag"] = flag.reindex(d.index).fillna(False).astype(bool)
    d = d.dropna()
    if d.empty or d["flag"].sum() < 30:
        return np.nan, np.nan, int(d["flag"].sum()) if len(d) else 0
    base = d.groupby(list(strata))[label].mean()
    sub = d[d["flag"]].groupby(list(strata))[label].agg(["mean", "size"])
    sub = sub.join(base.rename("base"))
    sub = sub[sub["base"] > 0]
    if sub.empty:
        return np.nan, np.nan, int(d["flag"].sum())
    lift = np.average(sub["mean"] / sub["base"], weights=sub["size"])
    diff = np.average(sub["mean"] - sub["base"], weights=sub["size"])
    return float(lift), float(diff), int(sub["size"].sum())


# ---------------------------------------------------------------------------
# Response surfaces (Part 9: surfaces first, no threshold optimisation)
# ---------------------------------------------------------------------------
def feature_response_surface(R: pd.DataFrame, periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fam, feats in FEATURE_FAMILIES.items():
        for f in feats:
            col = f if f.startswith("rs_pct_") or f in ("val_pct", "turnover_pct") else f"pct_{f}"
            if col not in R.columns:
                continue
            for pname, D in periods.items():
                q = np.ceil(D[col].clip(1e-9, 1) * 5)
                for qi, g in D.groupby(q):
                    st = outcome_stats(g)
                    rows.append({"family": fam, "feature": f, "period": pname, "quintile": int(qi), **st.to_dict()})
    return pd.DataFrame(rows)


def feature_lift_table(D: pd.DataFrame, top_q: float = 0.8) -> pd.DataFrame:
    """Vol-controlled lift of top quintile for each candidate feature (selection input)."""
    rows = []
    for fam, feats in FEATURE_FAMILIES.items():
        for f in feats:
            col = f if f.startswith("rs_pct_") or f in ("val_pct", "turnover_pct") else f"pct_{f}"
            if col not in D.columns:
                continue
            flag = D[col] >= top_q
            lift, diff, n = stratified_lift(D, flag)
            lift30, _, _ = stratified_lift(D, flag, "leader30")
            bad_lift, _, _ = stratified_lift(D, flag, "bad10")
            top = D[flag]
            rows.append({"family": fam, "feature": f, "orientation": "lower_better" if f in NEGATIVE else "higher_better",
                         "n_top": n, "lift_leader20_volctl": lift, "diff_leader20_volctl": diff,
                         "lift_leader30_volctl": lift30, "lift_mae10_volctl": bad_lift,
                         "top_fwd20_mean": top["fwd_ret_20"].mean(), "all_fwd20_mean": D["fwd_ret_20"].mean(),
                         "top_maxc40_p90": top["fwd_maxc_40"].quantile(0.9) if len(top) else np.nan})
    return pd.DataFrame(rows)


def select_discovery_features(D: pd.DataFrame, min_lift: float = 1.10) -> dict:
    """Pre-declared selection rule: best vol-controlled-lift feature per DISCOVERY family,
    family kept only if lift >= min_lift and the feature does not raise the -10% MAE rate
    by more than its leader lift (i.e. not just volatility)."""
    t = feature_lift_table(D)
    chosen = {}
    for fam in DISCOVERY_FAMILIES:
        sub = t[(t["family"] == fam)].dropna(subset=["lift_leader20_volctl"])
        sub = sub[sub["lift_leader20_volctl"] >= min_lift]
        sub = sub[sub["lift_leader20_volctl"] >= sub["lift_mae10_volctl"].fillna(1.0)]
        if sub.empty:
            continue
        best = sub.sort_values("lift_leader20_volctl", ascending=False).iloc[0]
        chosen[fam] = {"feature": best["feature"], "lift": round(float(best["lift_leader20_volctl"]), 4)}
    return {"families": chosen, "table": t}


# ---------------------------------------------------------------------------
# Independent strength (Part 9)
# ---------------------------------------------------------------------------
MKT_BINS = [-1, -0.01, -0.003, 0.003, 0.01, 1]
MKT_LABELS = ["mkt<-1%", "-1%~-0.3%", "flat±0.3%", "+0.3%~+1%", "mkt>+1%"]
EX_BINS = [-1, -0.02, 0, 0.02, 0.04, 0.06, 1]
EX_LABELS = ["ex<-2%", "-2~0%", "0~2%", "2~4%", "4~6%", "ex>6%"]


def independent_strength_surface(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pname, D in periods.items():
        d = D.assign(mbin=pd.cut(D["m_ret"], MKT_BINS, labels=MKT_LABELS),
                     xbin=pd.cut(D["ex_1"], EX_BINS, labels=EX_LABELS),
                     strong_close=(D["clv"] >= 0.7))
        for (mb, xb, sc), g in d.groupby(["mbin", "xbin", "strong_close"], observed=True):
            if len(g) < 30:
                continue
            rows.append({"period": pname, "market_day": mb, "stock_excess": xb, "close_loc>=0.7": bool(sc),
                         **outcome_stats(g).to_dict()})
        # multi-day: market regime x 10D cumulative excess
        d2 = D.assign(cbin=pd.qcut(D["cex_10"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]))
        for (rg, cb), g in d2.groupby(["regime", "cbin"], observed=True):
            rows.append({"period": pname, "market_day": f"REGIME:{rg}", "stock_excess": f"cex10_{cb}",
                         "close_loc>=0.7": None, **outcome_stats(g).to_dict()})
    return pd.DataFrame(rows)


def independent_strength_increment(D: pd.DataFrame) -> dict:
    """Does strength while the market is NOT up carry more information than the same
    excess return on an up day?  Stratified (ATR x beta x excess bucket) difference."""
    d = D[D["ex_1"] >= 0.02].copy()
    d["xbin"] = pd.cut(d["ex_1"], [0.02, 0.04, 0.06, 1])
    d["indep"] = d["m_ret"] <= 0.003
    res = {}
    for label in ("leader20", "leader30", "bad10"):
        rows = []
        for key, g in d.groupby(["atr_t3", "beta_t3", "xbin"], observed=True):
            a, b = g[g["indep"]][label].dropna(), g[~g["indep"]][label].dropna()
            if len(a) >= 10 and len(b) >= 10:
                w = 2 / (1 / len(a) + 1 / len(b))
                rows.append((a.mean() - b.mean(), w, a.mean(), b.mean()))
        if rows:
            arr = np.array(rows)
            res[f"{label}_diff_indep_minus_upday"] = float(np.average(arr[:, 0], weights=arr[:, 1]))
            res[f"{label}_indep_rate"] = float(np.average(arr[:, 2], weights=arr[:, 1]))
            res[f"{label}_upday_rate"] = float(np.average(arr[:, 3], weights=arr[:, 1]))
    # IS event vs matched non-event (same ATR/beta tercile, same cex_10 quintile)
    flag = D["is_event"] > 0
    lift, diff, n = stratified_lift(D.assign(cq=np.ceil(D["pct_cex_10"].clip(1e-9, 1) * 5)), flag,
                                    strata=("atr_t3", "beta_t3", "cq"))
    res.update(is_event_lift_vs_same_cex=lift, is_event_diff=diff, is_event_n=n)
    flag2 = D["is_cnt10"] >= 2
    lift2, diff2, n2 = stratified_lift(D.assign(cq=np.ceil(D["pct_cex_10"].clip(1e-9, 1) * 5)), flag2,
                                       strata=("atr_t3", "beta_t3", "cq"))
    res.update(is_cnt10ge2_lift_vs_same_cex=lift2, is_cnt10ge2_diff=diff2, is_cnt10ge2_n=n2)
    return res


# ---------------------------------------------------------------------------
# Downside resilience / upside participation / asymmetry (Parts 10-12)
# ---------------------------------------------------------------------------
def resilience_study(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pname, D in periods.items():
        for f in ("dr60", "dcap60", "up60", "ucap60", "asym60"):
            col = f"pct_{f}"
            q = np.ceil(D[col].clip(1e-9, 1) * 5)
            for qi, g in D.groupby(q):
                rows.append({"period": pname, "study": f, "bucket": f"Q{int(qi)}", **outcome_stats(g).to_dict()})
            lift, diff, n = stratified_lift(D, D[col] >= 0.8, strata=("atr_t3", "beta_t3"))
            rows.append({"period": pname, "study": f, "bucket": "TOP20%_volbeta_ctl_lift", "n": n,
                         "p_leader20": lift, "fwd20_ex_mean": diff})
        # 3x3 interaction (no arbitrary weighting)
        d = D.assign(drt=np.ceil(D["pct_dr60"].clip(1e-9, 1) * 3), upt=np.ceil(D["pct_up60"].clip(1e-9, 1) * 3))
        for (a, b), g in d.groupby(["drt", "upt"]):
            rows.append({"period": pname, "study": "DRxUP_grid", "bucket": f"DR_T{int(a)}|UP_T{int(b)}",
                         **outcome_stats(g).to_dict()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sector decomposition (Part 13)
# ---------------------------------------------------------------------------
def sector_decomposition(periods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pname, D in periods.items():
        d = D[D["sector"] != "UNKNOWN"].assign(
            stock_vs_sector=np.ceil(D["pct_srs_10"].clip(1e-9, 1) * 3),
            sector_vs_mkt=np.ceil(D["secmkt_10"].rank(pct=True).clip(1e-9, 1) * 3))
        for (a, b), g in d.groupby(["stock_vs_sector", "sector_vs_mkt"]):
            rows.append({"period": pname, "stock_vs_sector_T": int(a), "sector_vs_market_T": int(b),
                         **outcome_stats(g).to_dict()})
        for nm, flag in (("stock_specific_top20", d["pct_srs_10"] >= 0.8),
                         ("sector_beta_top20", d["secmkt_10"].rank(pct=True) >= 0.8),
                         ("market_rel_top20", d["pct_cex_10"] >= 0.8)):
            lift, diff, n = stratified_lift(d, flag, strata=("atr_t3", "beta_t3"))
            rows.append({"period": pname, "stock_vs_sector_T": nm, "sector_vs_market_T": "volbeta_ctl_lift",
                         "n": n, "p_leader20": lift, "fwd20_ex_mean": diff})
    return pd.DataFrame(rows)


def regime_profile(periods: dict[str, pd.DataFrame], score_col: str = "disc_pct") -> pd.DataFrame:
    """MARKET_SIDEWAYS + STOCK_STRONG vs other regimes for the same top-decile score."""
    rows = []
    for pname, D in periods.items():
        if score_col not in D.columns:
            continue
        for rg, g in D.groupby("regime"):
            top = g[g[score_col] >= 0.9]
            if len(top) < 20:
                continue
            lift, diff, n = stratified_lift(g, g[score_col] >= 0.9, strata=("atr_t3", "beta_t3"))
            rows.append({"period": pname, "regime": rg, "n_top": len(top), "volbeta_ctl_lift": lift,
                         **{f"top_{k}": v for k, v in outcome_stats(top).to_dict().items()},
                         "all_p_leader20": g["leader20"].mean()})
    return pd.DataFrame(rows)
