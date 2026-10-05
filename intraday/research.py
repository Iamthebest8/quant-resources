"""Intraday entry research: dataset assembly, learned trigger, policy comparison (Parts 22-27).

Chronological split ONLY (never random): TRAIN = candidate days 2023-01-01..2024-12-31,
TEST = 2025-01-01..2026-09-03 (~40% of covered days, == Strict OOS of the daily research).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from intraday.engine import FEATURES, day_dataset, utility

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-09-03"


def build(cands: pd.DataFrame, step: int = 15, log=print) -> pd.DataFrame:
    """cands columns: cand_id, strategy, stock_id, date (entry day), side, trigger, stop, atr, prev_close,
    adv_lots, next_close (all UNADJUSTED prices)."""
    out = []
    for i, c in enumerate(cands.itertuples()):
        try:
            df = day_dataset(c.stock_id, c.date, int(c.side), float(c.trigger), float(c.stop), float(c.atr),
                             float(c.prev_close), float(c.adv_lots), float(c.next_close) if np.isfinite(c.next_close) else None,
                             step=step, ctx={"cand_id": c.cand_id, "strategy": c.strategy})
        except Exception as e:      # malformed tick file etc. -> skip, keep going
            log(f"[intraday] {c.stock_id} {c.date}: {type(e).__name__} {str(e)[:80]}")
            continue
        if df is not None:
            out.append(df)
        if (i + 1) % 500 == 0:
            log(f"[intraday] {i + 1}/{len(cands)} candidate days")
    if not out:
        return pd.DataFrame()
    ds = pd.concat(out, ignore_index=True)
    for c in ds.columns:
        if ds[c].dtype == np.float64:
            ds[c] = ds[c].astype(np.float32)
    ds["date"] = pd.to_datetime(ds["date"])
    ds["split"] = np.where(ds["date"] <= pd.Timestamp(TRAIN_END), "TRAIN", "TEST")
    return ds


# ---------------------------------------------------------------------------
# rule-based policies (each returns the FIRST decision row satisfying the rule, per candidate day)
# ---------------------------------------------------------------------------
def _first(ds: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    sub = ds[mask]
    # first ROW per candidate (groupby.first() would fill NaN columns from later rows -> look-ahead)
    return sub.sort_values(["cand_id", "sec"]).drop_duplicates("cand_id", keep="first")


def policy_rows(ds: pd.DataFrame, model=None, thr: float | None = None) -> dict[str, pd.DataFrame]:
    P = {}
    P["OPEN"] = ds.sort_values(["cand_id", "sec"]).drop_duplicates("cand_id", keep="first")
    crossed = ds["crossed"] > 0
    P["BREAKOUT_IMMEDIATE"] = _first(ds, crossed)
    P["BREAKOUT_HOLD_60S"] = _first(ds, crossed & (ds["secs_since_cross"] >= 60) & (ds["dist_trigger"] > 0))
    P["BREAKOUT_RETEST_HOLD"] = _first(ds, ds["retest_hold"] > 0)
    P["BREAKOUT_MICRO_HL"] = _first(ds, crossed & (ds["micro_hl"] > 0) & (ds["dist_trigger"] > 0))
    P["BREAKOUT_VWAP_RECLAIM"] = _first(ds, crossed & (ds["vwap_reclaim"] > 0))
    P["BREAKOUT_REL_ACCEL"] = _first(ds, crossed & (ds["rel_r_5m"] > 0.005))
    P["VWAP_RECLAIM_ANY"] = _first(ds, ds["vwap_reclaim"] > 0)
    if model is not None and thr is not None:
        X = ds[FEATURES].astype(float)
        score = model.predict(X)
        P["LEARNED"] = _first(ds, pd.Series(score >= thr, index=ds.index))
    return P


def fit_model(train: pd.DataFrame, lam: float = 1.0, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = utility(train, lam).astype(float)
    m = np.isfinite(y)
    model = HistGradientBoostingRegressor(max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=seed)
    model.fit(train.loc[m, FEATURES].astype(float), y[m])
    return model


def choose_threshold(model, train: pd.DataFrame, qs=(0.6, 0.7, 0.8, 0.9), min_fill: float = 0.5) -> tuple:
    """Pre-declared rule: among score quantiles, pick the one maximising the mean utility of FIRST-trigger
    entries on TRAIN, subject to filling at least `min_fill` of candidate days."""
    s = model.predict(train[FEATURES].astype(float))
    best, best_u, table = None, -np.inf, []
    n_days = train["cand_id"].nunique()
    for q in qs:
        thr = float(np.quantile(s, q))
        rows = _first(train, pd.Series(s >= thr, index=train.index))
        fill = len(rows) / max(n_days, 1)
        u = float(utility(rows).mean()) if len(rows) else -np.inf
        table.append({"quantile": q, "threshold": thr, "fill_rate": fill, "train_utility": u})
        if fill >= min_fill and u > best_u:
            best, best_u = thr, u
    if best is None:
        best = table[0]["threshold"]
    return best, pd.DataFrame(table)


def summarize(P: dict[str, pd.DataFrame], n_days: int, label: str) -> pd.DataFrame:
    rows = []
    for k, r in P.items():
        if r is None or len(r) == 0:
            rows.append({"set": label, "policy": k, "fill_rate": 0.0})
            continue
        rows.append({"set": label, "policy": k, "n": len(r), "fill_rate": len(r) / max(n_days, 1),
                     "median_entry_time": r["time"].sort_values().iloc[len(r) // 2],
                     "entry_vs_open": float((r["entry_price"] / r["open_price"] - 1).mean() * r["side"].mean()),
                     "fwd_15m": float(r["fwd_15m"].mean()), "fwd_60m": float(r["fwd_60m"].mean()),
                     "fwd_eod": float(r["fwd_eod"].mean()), "fwd_next": float(r["fwd_next"].mean()),
                     "mfe_eod": float(r["mfe_eod"].mean()), "mae_eod": float(r["mae_eod"].mean()),
                     "mfe_mae": float(r["mfe_eod"].mean() / -r["mae_eod"].mean()) if r["mae_eod"].mean() < 0 else np.nan,
                     "utility": float(utility(r).mean()),
                     "daily_stop_hit": float(r["daily_stop_hit"].mean()),
                     "stop_dist_daily_atr": float(r["stop_dist_daily_atr"].median()),
                     "stop_dist_micro_atr": float(r["stop_dist_micro_atr"].median()),
                     "micro_stop_hit": float(r["micro_stop_hit"].mean()),
                     "stop_dist_lod_atr": float(r["stop_dist_lod_atr"].median()) if "stop_dist_lod_atr" in r else np.nan,
                     "lod_stop_hit": float(r["lod_stop_hit"].mean()) if "lod_stop_hit" in r else np.nan})
    return pd.DataFrame(rows)
