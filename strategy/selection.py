"""Staged, pre-declared parameter selection on a training window (Parts 27-30, 36, 39, 44-45).

Entry and exit are never optimised jointly. Each stage varies ONE block while all other
blocks stay at the pre-declared baseline / previously selected value:

  stage 1 Probe definition   : score pct {0.90,0.95} x trigger {rs_high, price_high, indep} x regime {all, not_up}
  stage 2 Failed-probe exit  : F1 | F1+F2 | F1+F3 | F1+F4(3/5/10) | F1+F2+F3 | F1+F2+F3+F4(5)
  stage 3 Confirmation       : price | rs | persist | price+rs | price+persist
  stage 4 Add architecture   : A (0->.25->1) | B (0->.5->1) | C (0->.25->.5->1)
  stage 5 Exit               : HS | TF | MS | MP
  stage 6 Portfolio          : max probes {2,3,4} x capital {A, B reserve 1.0, B reserve 1.5}

Objective stages 1-5: PnL per 100 slot-days (trade level, base cost+slippage) subject to n>=30, PF>=1.2.
Objective stage 6: Calmar (CAGR/|MDD|) of the 10-slot portfolio, tie-break Sharpe.
Campaigns are force-closed at the window end so the training window never reads later prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from strategy.config import StrategyConfig
from strategy.engine import trade_level
from strategy.metrics import campaign_metrics, objective
from strategy.portfolio import simulate_portfolio
from strategy.signals import Arrays, condition_frames

BASELINE = dict(probe_q=0.90, trigger="rs_high", regime_filter="all", fail="F1+F2+F3", time_stop=5,
                confirm="price+rs", add_arch="A", exit="TF", max_probes=3, capital_arch="A", probe_reserve=1.0)

STAGES = {
    1: [dict(probe_q=q, trigger=tr, regime_filter=rg) for q in (0.90, 0.95)
        for tr in ("rs_high", "price_high", "indep") for rg in ("all", "not_up")],
    2: [dict(fail="F1"), dict(fail="F1+F2"), dict(fail="F1+F3"), dict(fail="F1+F4", time_stop=3),
        dict(fail="F1+F4", time_stop=5), dict(fail="F1+F4", time_stop=10), dict(fail="F1+F2+F3"),
        dict(fail="F1+F2+F3+F4", time_stop=5)],
    3: [dict(confirm=c) for c in ("price", "rs", "persist", "price+rs", "price+persist")],
    4: [dict(add_arch=a) for a in ("A", "B", "C")],
    5: [dict(exit=e) for e in ("HS", "TF", "MS", "MP")],
    6: [dict(max_probes=k, capital_arch=ca, probe_reserve=rv) for k in (2, 3, 4)
        for ca, rv in (("A", 1.0), ("B", 1.0), ("B", 1.5))],
}
STAGE_NAMES = {1: "PROBE_DEFINITION", 2: "FAILED_PROBE_EXIT", 3: "CONFIRMATION", 4: "ADD_ARCHITECTURE",
               5: "EXIT", 6: "PORTFOLIO_CAPACITY"}


def window_idx(dates: pd.DatetimeIndex, start: str, end: str) -> tuple[int, int]:
    s = int(dates.searchsorted(pd.Timestamp(start)))
    e = int(dates.searchsorted(pd.Timestamp(end), side="right") - 1)
    return s, e


class SignalCache:
    """Caches probe-signal arrays per (disc features, probe_q, trigger, regime); keeps memory small."""

    def __init__(self, p, F):
        self.p, self.F = p, F
        self._c: dict = {}
        self._disc: dict = {}
        self._rank = None

    def get(self, cfg: StrategyConfig):
        key = (cfg.disc_features, cfg.probe_q, cfg.trigger, cfg.regime_filter, cfg.max_stop_dist)
        if key not in self._c:
            from strategy.signals import discovery_score
            if cfg.disc_features not in self._disc:
                self._disc[cfg.disc_features] = discovery_score(self.F, self.p.universe, cfg.disc_features)
            conds = condition_frames(self.p, self.F, cfg, disc_pct=self._disc[cfg.disc_features])
            if self._rank is None:
                self._rank = conds["rank_score"].to_numpy(dtype=np.float32)
            self._c[key] = (conds["probe_signal"].to_numpy(dtype=bool), self._rank, None)
        return self._c[key]


def staged_selection(A: Arrays, sc: SignalCache, disc_features: tuple, train: tuple[str, str],
                     version: str = "V1", log=print, cost=config.BASE_COST,
                     slip_bps=config.BASE_SLIPPAGE_BPS) -> tuple[StrategyConfig, pd.DataFrame]:
    s, e = window_idx(A.dates, *train)
    cur = StrategyConfig(version=version, disc_features=tuple(disc_features), **BASELINE)
    rows = []
    for st in range(1, 6):
        best, best_obj = None, -np.inf
        for cand in STAGES[st]:
            cfg = cur.with_(**cand)
            sig, rank, _ = sc.get(cfg)
            df = trade_level(A, sig, cfg, s, e, slip_bps / 1e4, cost, rank)
            m = campaign_metrics(df, years=(e - s + 1) / 252)
            obj = objective(m)
            rows.append({"window": f"{train[0]}~{train[1]}", "stage": st, "stage_name": STAGE_NAMES[st],
                         "candidate": cfg.short(), "params": str(cand), "objective": obj, **m})
            if obj > best_obj:
                best, best_obj = cfg, obj
        cur = best
        for r in rows:
            if r["stage"] == st:
                r["selected"] = r["candidate"] == cur.short()
        log(f"[select {train[0][:4]}-{train[1][:4]}] stage {st} {STAGE_NAMES[st]} -> {cur.short()} (obj={best_obj:.4f})")
    # stage 6: portfolio capacity / capital architecture
    best, best_key = None, (-np.inf, -np.inf)
    for cand in STAGES[6]:
        cfg = cur.with_(**cand)
        sig, rank, _ = sc.get(cfg)
        res = simulate_portfolio(A, sig, rank, cfg, s, e, cost=cost, slip_bps=slip_bps)
        m = res["metrics"]
        key = (m.get("calmar", -np.inf) if np.isfinite(m.get("calmar", np.nan)) else -np.inf,
               m.get("sharpe", -np.inf))
        rows.append({"window": f"{train[0]}~{train[1]}", "stage": 6, "stage_name": STAGE_NAMES[6],
                     "candidate": cfg.short(), "params": str(cand), "objective": key[0],
                     **{k: v for k, v in m.items()}})
        if key > best_key:
            best, best_key = cfg, key
    cur = best
    for r in rows:
        if r["stage"] == 6:
            r["selected"] = r["candidate"] == cur.short()
    log(f"[select {train[0][:4]}-{train[1][:4]}] stage 6 PORTFOLIO -> {cur.short()} (calmar={best_key[0]:.3f})")
    return cur, pd.DataFrame(rows)
