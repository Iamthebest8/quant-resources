"""Campaign- and portfolio-level metrics (Parts 25, 29, 31, 46-48)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NaN = float("nan")


def _pf(x: pd.Series) -> float:
    w, l = x[x > 0].sum(), -x[x < 0].sum()
    return float(w / l) if l > 0 else (float("inf") if w > 0 else NaN)


def campaign_metrics(df: pd.DataFrame, years: float | None = None) -> dict:
    if df is None or df.empty:
        return {"n_probes": 0}
    df = df[df["pnl"].notna()]
    n = len(df)
    pnl = df["pnl_slots"]
    ret = df["ret_on_invested"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    failed = df[df["final_state"] == "FAILED_PROBE"]
    conf = df[df["confirmed"]]
    adds = df[df["n_adds"] > 0]
    srt = pnl.sort_values(ascending=False)
    k1, k5 = max(1, int(round(n * 0.01))), max(1, int(round(n * 0.05)))
    tot = pnl.sum()
    if years is None:
        span = (df["probe_date"].max() - df["probe_date"].min()).days / 365.25 if n > 1 else 1
        years = max(span, 0.25)
    m = {
        "n_probes": n,
        "probes_per_year": n / years,
        "win_rate": float((pnl > 0).mean()),
        "pf": _pf(pnl),
        "payoff": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else NaN,
        "ev_slots_per_probe": float(pnl.mean()),
        "total_pnl_slots": float(tot),
        "slot_days": float(df["slot_days"].sum()),
        "pnl_per_100_slot_days": float(100 * tot / df["slot_days"].sum()) if df["slot_days"].sum() > 0 else NaN,
        "avg_winner_ret": float(ret[ret > 0].mean()) if (ret > 0).any() else NaN,
        "avg_loser_ret": float(ret[ret < 0].mean()) if (ret < 0).any() else NaN,
        "avg_winner_slots": float(wins.mean()) if len(wins) else NaN,
        "avg_loser_slots": float(losses.mean()) if len(losses) else NaN,
        "median_holding": float(df["holding_days"].median()),
        "ge10": int((ret >= 0.10).sum()), "ge20": int((ret >= 0.20).sum()),
        "ge30": int((ret >= 0.30).sum()), "ge40": int((ret >= 0.40).sum()),
        "ge20_rate": float((ret >= 0.20).mean()),
        "top1pct_share": float(srt.iloc[:k1].sum() / tot) if tot > 0 else NaN,
        "top5pct_share": float(srt.iloc[:k5].sum() / tot) if tot > 0 else NaN,
        "top5_trades_share": float(srt.iloc[:5].sum() / tot) if tot > 0 else NaN,
        "pnl_ex_top5_slots": float(srt.iloc[5:].sum()),
        "ret_p95": float(ret.quantile(0.95)), "ret_p99": float(ret.quantile(0.99)), "ret_max": float(ret.max()),
        "skew": float(ret.skew()) if n > 2 else NaN,
        # --- probe metrics (Part 48) ---
        "false_probe_rate": float(len(failed) / n),
        "avg_failed_loss_ret": float(failed["ret_on_invested"].mean()) if len(failed) else NaN,
        "median_failed_loss_ret": float(failed["ret_on_invested"].median()) if len(failed) else NaN,
        "avg_failed_loss_slots": float(failed["pnl_slots"].mean()) if len(failed) else NaN,
        "failed_mae_mean": float(failed["probe_mae"].mean()) if len(failed) else NaN,
        "failed_worst5pct_ret": float(failed["ret_on_invested"].quantile(0.05)) if len(failed) else NaN,
        "failed_holding_median": float(failed["holding_days"].median()) if len(failed) else NaN,
        "failed_slot_days": float(failed["slot_days"].sum()),
        "confirmation_rate": float(len(conf) / n),
        "add_rate": float(len(adds) / n),
        "full_rate": float(df["full"].mean()),
        "days_probe_to_confirm_mean": float(conf["days_probe_to_confirm"].mean()) if len(conf) else NaN,
        "days_probe_to_confirm_median": float(conf["days_probe_to_confirm"].median()) if len(conf) else NaN,
        # --- add quality (Part 29) ---
        "add_premium_mean": float(adds["add_premium"].mean()) if len(adds) else NaN,
        "add_premium_median": float(adds["add_premium"].median()) if len(adds) else NaN,
        "add_mfe_mean": float(adds["add_mfe"].mean()) if len(adds) else NaN,
        "add_mae_mean": float(adds["add_mae"].mean()) if len(adds) else NaN,
        "add_leg_pf": _pf(adds["add_leg_ret"]) if len(adds) else NaN,
        "add_payoff": (float(adds.loc[adds["add_leg_ret"] > 0, "add_leg_ret"].mean()
                             / -adds.loc[adds["add_leg_ret"] < 0, "add_leg_ret"].mean())
                       if len(adds) and (adds["add_leg_ret"] < 0).any() and (adds["add_leg_ret"] > 0).any() else NaN),
        "add_reward_risk_at_add": (float((adds["add_mfe"] / adds["add_mae"].abs().clip(lower=0.005)).median())
                                   if len(adds) else NaN),
    }
    return m


def objective(m: dict, min_n: int = 30, min_pf: float = 1.2) -> float:
    """Pre-declared selection objective: PnL per 100 slot-days (capital efficiency),
    only for candidates with >= min_n probes and PF >= min_pf; others get -inf."""
    if m.get("n_probes", 0) < min_n or not np.isfinite(m.get("pnl_per_100_slot_days", NaN)):
        return -np.inf
    if not (m.get("pf", 0) >= min_pf):
        return -1e6 + m["pnl_per_100_slot_days"]       # ranked below every eligible candidate
    return m["pnl_per_100_slot_days"]


def equity_metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 3:
        return {}
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else NaN
    dd = eq / eq.cummax() - 1
    return {"total_return": float(eq.iloc[-1] / eq.iloc[0] - 1), "cagr": float(cagr), "mdd": float(dd.min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else NaN,
            "calmar": float(cagr / -dd.min()) if dd.min() < 0 else NaN, "vol": float(r.std() * np.sqrt(252))}
