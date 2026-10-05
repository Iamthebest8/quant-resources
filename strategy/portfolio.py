"""10-SLOT PATH-DEPENDENT PORTFOLIO (Parts 43-46).

* 1 slot = 10% of current equity (normal position). Probes consume real capital.
* Capital architecture A: all 10 slots shared by probes and confirmed positions.
  Capital architecture B: `probe_reserve` slot-equivalents reserved for probes;
  confirmed/full positions may use at most 10 - reserve slots.
* At most `max_probes` simultaneous probes. Candidates ranked by the Part-23 ranking score.
* Fills at the open with slippage + half round-trip cost per side; cash never negative;
  per-fill participation <= MAX_PARTICIPATION of that day's traded value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from strategy.config import StrategyConfig
from strategy.engine import Campaign
from strategy.metrics import campaign_metrics, equity_metrics
from strategy.signals import Arrays


def simulate_portfolio(A: Arrays, signal: np.ndarray, rank: np.ndarray, cfg: StrategyConfig, start_idx: int,
                       end_idx: int, cost: float = config.BASE_COST,
                       slip_bps: float = config.BASE_SLIPPAGE_BPS, capital: float = config.INITIAL_CAPITAL,
                       record_positions: bool = False, live: bool = False) -> dict:
    """live=True: positions open on the last data date are kept open (marked to market), not force-closed."""
    slip = slip_bps / 1e4
    half = cost / 2
    N = config.N_SLOTS
    cash = capital
    equity_prev = capital
    active: dict[int, Campaign] = {}
    cooldown: dict[int, int] = {}
    pending_new: list[tuple[int, int, float]] = []
    closed: list[Campaign] = []
    daily = []
    positions = []
    traded = 0.0
    n_missed_capacity = 0

    def used():
        tot = sum(c.size() for c in active.values())
        pr = sum(c.size() for c in active.values() if c.state == "PROBE")
        npr = sum(1 for c in active.values() if c.state == "PROBE")
        return tot, pr, npr

    for d in range(start_idx, end_idx + 1):
        slot_value = equity_prev / N
        state = {"cash": cash}

        def alloc(kind: str, size: float, _cp=None) -> float:
            tot, pr, npr = used()
            if kind == "PROBE":
                if npr >= cfg.max_probes:
                    return 0.0
                room = (N - tot) if cfg.capital_arch == "A" else min(cfg.probe_reserve - pr, N - tot)
            else:
                room = (N - tot) if cfg.capital_arch == "A" else ((N - cfg.probe_reserve) - (tot - pr))
            size = max(0.0, min(size, room))
            # cash + liquidity constraints
            j = _cp.j if _cp is not None else None
            max_cash = state["cash"] / (slot_value * (1 + half)) if slot_value > 0 else 0
            size = min(size, max_cash)
            if j is not None and np.isfinite(A.VAL[d, j]) and A.VAL[d, j] > 0:
                size = min(size, config.MAX_PARTICIPATION * A.VAL[d, j] / slot_value)
            return float(np.floor(size * 100) / 100)

        def step(cp: Campaign, fn):
            nonlocal cash, traded
            n_before = len(cp.legs)
            was_open = cp.is_open()
            fn()
            for leg in cp.legs[n_before:]:
                cash -= leg.notional * (1 + half)
                traded += leg.notional
                state["cash"] = cash
            if was_open and not cp.is_open() and cp.exit_day == d:
                val = sum(l.notional * cp.exit_px / l.px for l in cp.legs)
                cash += val * (1 - half)
                traded += val
                state["cash"] = cash

        # ---- 1. open: exits first, then adds, then new probes ------------------------------
        for cp in sorted(active.values(), key=lambda c: 0 if any(k == "EXIT" for k, _, _ in c.pending) else 1):
            cp.slot_value = slot_value
            step(cp, lambda cp=cp: cp.on_open(d, lambda k, s, cp=cp: alloc(k, s, cp)))
        for j, t, rk in pending_new:
            if j in active:
                continue
            cp = Campaign(A=A, j=j, t=t, cfg=cfg, slip=slip, cost=cost, slot_value=slot_value,
                          end_idx=(A.T + 5) if live else end_idx, rank=rk)
            if not cp.init_signal():
                continue
            step(cp, lambda cp=cp: cp.on_open(d, lambda k, s, cp=cp: alloc(k, s, cp)))
            if cp.legs:
                active[j] = cp
            elif cp.skipped == "no_capacity":
                n_missed_capacity += 1
        pending_new = []
        # ---- 2. intraday stops ------------------------------------------------------------
        for cp in list(active.values()):
            step(cp, lambda cp=cp: cp.on_intraday(d))
        # ---- 3. close ---------------------------------------------------------------------
        for cp in list(active.values()):
            step(cp, lambda cp=cp: cp.on_close(d))
        for j in [j for j, cp in active.items() if not cp.is_open()]:
            cp = active.pop(j)
            closed.append(cp)
            cooldown[j] = d + cfg.cooldown
        mv = sum(cp.mtm_value(d) for cp in active.values())
        equity = cash + mv
        tot, pr, npr = used()
        probe_mv = sum(cp.mtm_value(d) for cp in active.values() if cp.state == "PROBE")
        daily.append({"date": A.dates[d], "equity": equity, "cash": cash, "invested_mv": mv,
                      "n_probe": npr, "n_confirmed": sum(1 for c in active.values() if c.state in ("CONFIRMED", "ADD")),
                      "n_full": sum(1 for c in active.values() if c.state == "FULL"),
                      "slots_used": tot, "probe_slots": pr, "probe_capital": probe_mv,
                      "occupancy": tot / N})
        if record_positions:
            for cp in active.values():
                positions.append({"date": A.dates[d], "stock_id": A.ids[cp.j], "state": cp.state,
                                  "size_slots": cp.size(), "avg_cost": cp.raw_px(d, cp.avg_cost()),
                                  "close": cp.raw_px(d, A.C[d, cp.j]), "unrealized": A.C[d, cp.j] / cp.avg_cost() - 1,
                                  "mtm": cp.mtm_value(d), "stop": cp.raw_px(d, cp.stop),
                                  "probe_date": A.dates[cp.entry_day] if cp.entry_day >= 0 else pd.NaT})
        equity_prev = equity
        # ---- 4. new probe candidates for tomorrow ----------------------------------------------
        if d < end_idx:
            slots_free = cfg.max_probes - npr
            if slots_free > 0:
                cand = np.where(signal[d])[0]
                cand = [j for j in cand if j not in active and cooldown.get(j, -1) < d]
                cand.sort(key=lambda j: -(rank[d, j] if np.isfinite(rank[d, j]) else -1))
                pending_new = [(j, d, rank[d, j]) for j in cand[:slots_free]]

    # force close leftovers at the window end (already done by Campaign at end_idx)
    for cp in active.values():
        if cp.is_open() and not live:
            cp._close_out(end_idx, A.C[end_idx, cp.j] * (1 - slip), "END_OF_WINDOW")
        closed.append(cp)
    eq = pd.DataFrame(daily).set_index("date")
    trades = pd.DataFrame([c.record() for c in closed if c.legs])
    yrs = len(eq) / 252
    m = equity_metrics(eq["equity"])
    cm = campaign_metrics(trades, years=yrs) if len(trades) else {"n_probes": 0}
    avg_eq = eq["equity"].mean()
    failed = trades[trades["final_state"] == "FAILED_PROBE"] if len(trades) else trades
    ok = trades[trades["final_state"] != "FAILED_PROBE"] if len(trades) else trades
    pnl_per_slot_day_ok = (ok["pnl"].sum() / ok["slot_days"].sum()) if len(ok) and ok["slot_days"].sum() else 0
    m.update({
        "avg_occupancy": float(eq["occupancy"].mean()),
        "avg_probe_capital_pct": float((eq["probe_capital"] / eq["equity"]).mean()),
        "probe_capital_utilization": float(eq["probe_slots"].mean() /
                                           (cfg.probe_reserve if cfg.capital_arch == "B" else
                                            cfg.max_probes * cfg.probe_size)),
        "avg_cash_pct": float((eq["cash"] / eq["equity"]).mean()),
        "turnover_annual": float(traded / 2 / avg_eq / yrs) if yrs > 0 else np.nan,
        "failed_probe_slot_days": float(failed["slot_days"].sum()) if len(failed) else 0.0,
        "opportunity_cost_pct": float(failed["slot_days"].sum() * pnl_per_slot_day_ok / avg_eq) if len(failed) else 0.0,
        "missed_for_capacity": n_missed_capacity,
        "pnl_per_100_slot_days_pct": float(100 * trades["pnl"].sum() /
                                           (eq["slots_used"].sum() * avg_eq / config.N_SLOTS))
        if len(trades) and eq["slots_used"].sum() > 0 else np.nan,
        "trades_per_year": float(len(trades) / yrs) if yrs > 0 else np.nan,
    })
    m.update({f"c_{k}": v for k, v in cm.items()})
    out = {"equity": eq, "trades": trades, "metrics": m}
    if record_positions:
        out["positions"] = pd.DataFrame(positions)
    return out
