"""Alpha overlap, hybrids A-D and the multi-alpha 10-slot portfolio (phase 2).

PRE-REGISTERED definitions (written before any hybrid / portfolio result was computed):
  HYBRID_A  Weinstein-filtered momentum: frozen MOMENTUM probes kept only if the stock's last completed weekly
            stage is 2 and the 30-week MA is not declining (book p.14, 129: never buy below / with declining MA).
  HYBRID_B  Momentum-confirmed Weinstein breakout: W1/W2 TEXTBOOK trades whose stock had a MOMENTUM V1
            discovery percentile >= 0.90 on any of the 10 trading days before the fill.
  HYBRID_C  Leader pullback: W3 TEXTBOOK trades with 120-day excess-return percentile >= 0.80 at the signal.
  HYBRID_D  Momentum entry + Weinstein MODERN exit: MOMENTUM probe entries (same day / price / stop), managed
            with the MODERN exit (stage-4, weekly close < MA10w, break-even after 2R) instead of the V1 exits.
  Components for every hybrid: MOMENTUM_LONG_BASELINE and WEINSTEIN_LONG_TEXTBOOK (W1+W2+W3 pooled).

  MULTI_ALPHA_PORTFOLIO: 10 slots (1 slot = 10% of equity at entry; NT$10M start). Every position, long or short,
  consumes its slot fraction of capital. Engines: MOMENTUM (frozen V1 campaigns, size = campaign max size),
  W1/W2/W3 TEXTBOOK (1 slot), S1/S2 TEXTBOOK EXECUTABLE (1 slot, at most 3 short slots). Same-day priority:
  W3 > W1 > W2 > MOMENTUM > S2 > S1, then by entry order. Positions marked to market daily on adjusted closes;
  the trade's realised return (incl. costs) is booked at exit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline.phase2 import COST, END, OUT, P2, SLIP, WINDOWS, log, save_csv, stats_table  # noqa: E402

PRIORITY = {"W3": 0, "W1": 1, "W2": 2, "MOMENTUM": 3, "S2": 4, "S1": 5, "HYBRID_A": 3, "HYBRID_B": 1,
            "HYBRID_C": 0, "HYBRID_D": 3, "HIGH_RR_STAGE2": 2}


def momentum_trades(p) -> pd.DataFrame:
    pe = pd.read_csv(OUT / "PROBE_EVENTS.csv", dtype={"stock_id": str}, parse_dates=["probe_date", "exit_date"])
    ids = {s: j for j, s in enumerate(p.ids)}
    pe = pe[pe["stock_id"].map(ids).notna()].copy()
    out = pd.DataFrame({
        "engine": "MOMENTUM", "variant": "FROZEN_V1", "exit": "V1", "stock_id": pe["stock_id"], "side": "LONG",
        "entry_date": pe["probe_date"], "exit_date": pe["exit_date"], "entry_price_adj": pe["probe_price_adj"],
        "size": pe["max_size"].clip(lower=0.25), "ret": pe["pnl_slots"] / pe["max_size"].clip(lower=0.25),
        "pnl_slots": pe["pnl_slots"], "slot_days": pe["slot_days"], "stop_dist_pct": pe["probe_risk"],
        "mfe": pe["mfe"], "mae": pe["mae"], "holding_days": pe["holding_days"], "exit_reason": pe["exit_reason"],
        "j": pe["stock_id"].map(ids).astype(int), "probe_stop_adj": pe["probe_stop_adj"],
        "disc_pct": pe.get("disc_pct")})
    out["t"] = p.dates.get_indexer(out["entry_date"]) - 1
    out["r_multiple"] = out["ret"] / out["stop_dist_pct"]
    out["stop_dist_atr"] = np.nan
    out["first_exit_reason"] = out["exit_reason"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------------------------
def overlap(p, trades: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Share of each engine's entries where another engine entered the same stock within +-window days, and the
    PnL of overlapping vs non-overlapping entries; plus correlation of the engines' daily equal-weight PnL."""
    tr = trades.copy()
    tr["e"] = p.dates.get_indexer(tr["entry_date"])
    engines = sorted(tr["engine"].unique())
    by = {e: tr[tr["engine"] == e] for e in engines}
    rows = []
    for a in engines:
        A = by[a]
        for b in engines:
            if a == b:
                continue
            B = by[b]
            key = B.groupby("stock_id")["e"].apply(np.array).to_dict()
            hit = np.array([bool(len(key.get(s, [])) and np.any(np.abs(key[s] - e) <= window))
                            for s, e in zip(A["stock_id"], A["e"])])
            for w, (s0, s1) in WINDOWS.items():
                m = (A["entry_date"] >= s0) & (A["entry_date"] <= s1)
                if m.sum() == 0:
                    continue
                rows.append({"engine": a, "other": b, "window": w, "n": int(m.sum()),
                             "overlap_share": float(hit[m].mean()),
                             "ev_overlap": float(A["ret"][m & hit].mean()) if (m & hit).any() else np.nan,
                             "ev_no_overlap": float(A["ret"][m & ~hit].mean()) if (m & ~hit).any() else np.nan,
                             "n_overlap": int((m & hit).sum())})
    ov = pd.DataFrame(rows)
    # daily pnl correlation (trade returns spread evenly over holding days)
    T = len(p.dates)
    series = {}
    for e in engines:
        x = np.zeros(T)
        for r in by[e].itertuples():
            s, f = p.dates.get_loc(r.entry_date), p.dates.get_loc(r.exit_date)
            n = max(f - s + 1, 1)
            x[s:f + 1] += r.ret * getattr(r, "size", 1.0) / n
        series[e] = pd.Series(x, index=p.dates)
    S = pd.DataFrame(series)
    cr = []
    for w, (s0, s1) in WINDOWS.items():
        c = S.loc[s0:s1].corr()
        for a in engines:
            for b in engines:
                if a < b:
                    cr.append({"engine": a, "other": b, "window": w, "daily_pnl_corr": c.at[a, b]})
    return pd.concat([ov.assign(table="ENTRY_OVERLAP"), pd.DataFrame(cr).assign(table="PNL_CORRELATION")],
                     ignore_index=True)


# ---------------------------------------------------------------------------------------------
# hybrids
# ---------------------------------------------------------------------------------------------
def hybrids(p, F, G, ctx, wtr: pd.DataFrame, mom: pd.DataFrame, M, E) -> pd.DataFrame:
    from alpha.trades import simulate
    from pipeline.run_research import load_frozen
    from strategy.signals import discovery_score
    st = ctx.S["stage"]
    sl = ctx.S["slope4"]
    def wk_val(frame, t, j):
        k = ctx.wd.week_end.searchsorted(p.dates[t], side="right") - 1
        return frame.iat[k, j] if k >= 0 else np.nan
    out = []
    # A
    a = mom.copy()
    keep = [(wk_val(st, r.t, r.j) == 2) and (wk_val(sl, r.t, r.j) >= -0.01) for r in a.itertuples()]
    out.append(a[keep].assign(engine="HYBRID_A", variant="PRE_REGISTERED"))
    # B
    cfg, _ = load_frozen("V1")
    disc = discovery_score(F, p.universe, cfg.disc_features).to_numpy()
    tb = wtr[(wtr["variant"] == "TEXTBOOK")]
    b = tb[tb["engine"].isin(["W1", "W2"])].copy()
    e_idx = p.dates.get_indexer(b["entry_date"])
    keep = [np.nanmax(disc[max(0, e - 10):e, j]) >= 0.90 if e > 0 else False for e, j in zip(e_idx, b["j"])]
    out.append(b[keep].assign(engine="HYBRID_B", variant="PRE_REGISTERED"))
    # C
    c = tb[tb["engine"] == "W3"].copy()
    mk = p.market["adj_close"]
    ex = (p.c / p.c.shift(120) - 1).sub(mk / mk.shift(120) - 1, axis=0)
    pct = ex.where(p.universe).rank(axis=1, pct=True).to_numpy()
    keep = [pct[t, j] >= 0.80 for t, j in zip(c["t"], c["j"])]
    out.append(c[keep].assign(engine="HYBRID_C", variant="PRE_REGISTERED"))
    # D
    end_idx = int(p.dates.searchsorted(pd.Timestamp(END), side="right") - 1)
    rows = []
    for r in mom.itertuples():
        e = p.dates.get_loc(r.entry_date)
        res = simulate(M, int(r.j), e - 1, 1, "open", np.nan, r.probe_stop_adj, E["MODERN_NOVOL"], end_idx, COST,
                       SLIP, fixed_entry=(e, r.entry_price_adj))
        if res:
            res.update({"engine": "HYBRID_D", "variant": "PRE_REGISTERED", "exit": "MODERN_NOVOL", "j": r.j,
                        "t": e - 1})
            rows.append(res)
    d = pd.DataFrame(rows)
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# 10-slot portfolio
# ---------------------------------------------------------------------------------------------
def portfolio(p, trades: pd.DataFrame, start: str, end: str, n_slots: int = 10, max_short_slots: int = 3,
              extra_cost: float = 0.0, extra_slip: float = 0.0, capital: float = config.INITIAL_CAPITAL) -> dict:
    """Path-dependent slot portfolio over a trade list (entry/exit dates, entry_price_adj, ret incl. costs)."""
    C = p.c.to_numpy()
    dates = p.dates
    s_i = int(dates.searchsorted(pd.Timestamp(start)))
    e_i = int(dates.searchsorted(pd.Timestamp(end), side="right") - 1)
    tr = trades[(trades["entry_date"] >= start) & (trades["entry_date"] <= end)].copy()
    tr["e"] = dates.get_indexer(tr["entry_date"])
    tr["x"] = dates.get_indexer(tr["exit_date"]).clip(max=e_i)
    tr["prio"] = tr["engine"].map(PRIORITY).fillna(9)
    tr = tr.sort_values(["e", "prio"]).reset_index(drop=True)
    by_day = {k: g for k, g in tr.groupby("e")}
    cash = capital
    open_pos = []      # dict(j, side, notional, px, size, x, ret, engine)
    eq = []
    taken = []
    for d in range(s_i, e_i + 1):
        # exits at today's prices (booked with the trade's realised return)
        still = []
        for q in open_pos:
            if d >= q["x"]:
                ret = q["ret"] - extra_cost - 2 * extra_slip
                cash += q["notional"] * (1 + ret)
            else:
                still.append(q)
        open_pos = still
        # mark to market
        def mtm(q):
            c = C[d, q["j"]]
            if not np.isfinite(c):
                return q["last"]
            q["last"] = q["notional"] * (1 + q["side"] * (c / q["px"] - 1))
            return q["last"]
        equity = cash + sum(mtm(q) for q in open_pos)
        # entries
        if d in by_day:
            used = sum(q["size"] for q in open_pos)
            used_s = sum(q["size"] for q in open_pos if q["side"] < 0)
            for r in by_day[d].itertuples():
                size = float(getattr(r, "size", 1.0) or 1.0)
                side = 1 if r.side == "LONG" else -1
                if used + size > n_slots + 1e-9:
                    continue
                if side < 0 and used_s + size > max_short_slots + 1e-9:
                    continue
                if any(q["j"] == r.j for q in open_pos):
                    continue
                notional = equity * 0.10 * size
                if notional > cash + 1e-6:
                    continue
                cash -= notional
                q = {"j": int(r.j), "side": side, "notional": notional, "px": r.entry_price_adj, "size": size,
                     "x": int(r.x), "ret": r.ret, "engine": r.engine, "last": notional}
                open_pos.append(q)
                used += size
                used_s += size if side < 0 else 0
                taken.append(r.Index)
        eq.append({"date": dates[d], "equity": equity, "n_pos": len(open_pos),
                   "slots_used": sum(q["size"] for q in open_pos),
                   "short_slots": sum(q["size"] for q in open_pos if q["side"] < 0)})
    E = pd.DataFrame(eq).set_index("date")
    return {"equity": E, "taken": tr.loc[taken]}


def perf(E: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    eq = E["equity"]
    r = eq.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    out = {"cagr": cagr, "total_ret": eq.iloc[-1] / eq.iloc[0] - 1,
           "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, "mdd": dd.min(),
           "calmar": cagr / -dd.min() if dd.min() < 0 else np.nan, "avg_slots_used": E["slots_used"].mean(),
           "exposure": (E["slots_used"] > 0).mean()}
    for y in (2023, 2024, 2025, 2026):
        ey = eq[eq.index.year == y]
        if len(ey) > 5:
            prev = eq[eq.index.year < y]
            base = prev.iloc[-1] if len(prev) else ey.iloc[0]
            out[f"ret_{y}"] = ey.iloc[-1] / base - 1
    if bench is not None:
        b = bench.reindex(eq.index).ffill()
        out["bench_ret"] = b.iloc[-1] / b.iloc[0] - 1
    return out
