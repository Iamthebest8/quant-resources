"""Signal layer: Discovery score, probe conditions, and numpy arrays for the engine.

Only PIT features (engine.features) are used here. No forward labels are imported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.features import cs_pct, ranking_score
from engine.panel import Panel
from strategy.config import StrategyConfig


@dataclass
class Arrays:
    dates: pd.DatetimeIndex
    ids: list
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    G: np.ndarray          # open / previous valid close (adj)  -> limit-up / gap checks
    LOCK_UP: np.ndarray
    LOCK_DN: np.ndarray
    MA10: np.ndarray
    MA20: np.ndarray
    LOW3: np.ndarray
    HI20: np.ndarray
    RSL: np.ndarray
    RSLMAX40: np.ndarray
    RS20: np.ndarray
    RS40: np.ndarray
    TREND: np.ndarray
    ATR: np.ndarray
    SEC: np.ndarray        # cumulative sector level (leave-one-out EW)
    MKT_C: np.ndarray
    MKT_O: np.ndarray
    VAL: np.ndarray        # traded value TWD
    RAW_C: np.ndarray
    last_valid: np.ndarray

    @property
    def T(self):
        return self.C.shape[0]


def build_arrays(p: Panel, F: dict) -> Arrays:
    c = p.c
    prev = c.ffill().shift(1)
    G = (p.o / prev).to_numpy()
    r = p.r
    lock = (p.h - p.l).abs() <= 1e-9
    last_valid = np.array([c[s].last_valid_index() for s in p.ids], dtype=object)
    lv = np.array([p.dates.get_loc(x) if x is not None else -1 for x in last_valid])
    return Arrays(
        dates=p.dates, ids=list(p.ids),
        O=p.o.to_numpy(), H=p.h.to_numpy(), L=p.l.to_numpy(), C=c.to_numpy(), G=G,
        LOCK_UP=(lock & (r >= 0.095)).to_numpy(), LOCK_DN=(lock & (r <= -0.095)).to_numpy(),
        MA10=F["ma10"].to_numpy(), MA20=F["ma20"].to_numpy(), LOW3=F["low3"].to_numpy(),
        HI20=p.h.rolling(20, min_periods=15).max().to_numpy(),
        RSL=F["rsl"].to_numpy(), RSLMAX40=F["rsl"].rolling(40, min_periods=30).max().to_numpy(),
        RS20=F["rs_20"].to_numpy(), RS40=F["rs_40"].to_numpy(), TREND=F["trend_ok"].to_numpy(),
        ATR=F["atr"].to_numpy(), SEC=F["sec_level"].to_numpy(),
        MKT_C=p.market["adj_close"].to_numpy(), MKT_O=p.market["adj_open"].to_numpy(),
        VAL=p.val.to_numpy(), RAW_C=p.raw_c.to_numpy(), last_valid=lv)


def discovery_score(F: dict, U: pd.DataFrame, features: tuple) -> pd.DataFrame:
    """Equal-weight mean of cross-sectional percentiles, re-ranked within the universe."""
    if not features:
        raise ValueError("discovery score needs at least one feature")
    parts = [F[f"pct_{f}"] if f"pct_{f}" in F else cs_pct(F[f], U) for f in features]
    raw = sum(p.fillna(0.5) for p in parts) / len(parts)
    return cs_pct(raw, U)


def condition_frames(p: Panel, F: dict, cfg: StrategyConfig, disc_pct: pd.DataFrame | None = None) -> dict:
    """Boolean condition matrices (used by signals, Trigger Panel and the 3653 timeline)."""
    U = p.universe
    if disc_pct is None:
        disc_pct = discovery_score(F, U, cfg.disc_features)
    c = p.c
    regime = F["mkt"]["regime"]
    reg_ok = pd.Series(True, index=p.dates) if cfg.regime_filter == "all" else (regime != "MARKET_UP")
    reg_ok = pd.DataFrame(np.repeat(reg_ok.to_numpy()[:, None], len(p.ids), axis=1), index=p.dates, columns=p.ids)
    px_high10 = (c >= c.rolling(10, min_periods=8).max() - 1e-9)
    trig = {
        "rs_high": (F["rsl_high20"] > 0) & px_high10,
        "price_high": F["relhigh"] > 0,
        "indep": F["is_event"] > 0,
    }
    stop_dist = 1 - F["low3"] / c
    conds = {
        "c_universe": U,
        "c_no_disposition": ~(p.disp | p.disp_next),
        "c_watch": disc_pct >= cfg.watch_q,
        "c_discovery": disc_pct >= cfg.disc_q,
        "c_probe_score": disc_pct >= cfg.probe_q,
        "c_trigger": trig[cfg.trigger],
        "c_trig_rs_high": trig["rs_high"],
        "c_trig_price_high": trig["price_high"],
        "c_trig_indep": trig["indep"],
        "c_regime": reg_ok,
        "c_stop_ok": (stop_dist <= cfg.max_stop_dist) & (stop_dist > 0.003),
    }
    conds["probe_signal"] = (conds["c_universe"] & conds["c_no_disposition"] & conds["c_probe_score"]
                             & conds["c_trigger"] & conds["c_regime"] & conds["c_stop_ok"])
    conds["disc_pct"] = disc_pct
    conds["rank_score"] = ranking_score(F, U)
    conds["stop_dist"] = stop_dist
    return conds


PROBE_CONDITION_LABELS = {
    "c_universe": "流動性/價格/上市天數",
    "c_no_disposition": "非處置股",
    "c_probe_score": "Discovery 分數達門檻",
    "c_trigger": "觸發條件 (相對新高/獨立強勢)",
    "c_regime": "大盤 regime 條件",
    "c_stop_ok": "停損距離合理",
}
