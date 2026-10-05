"""User-spec items (PART 37/38): HIGH_RR_STAGE2_SETUP (7 conditions), HYBRID_C, HYBRID_D (probe -> Weinstein
confirmation -> add). Definitions registered in STRATEGY_DECISION_LOG.md §A2 before any of these was computed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.phase2 import (COST, END, P2, SLIP, WINDOWS, bootstrap_p, exit_specs, log, no_overlap,  # noqa: E402
                             save_csv, stats_table, tag_windows)

COND = ["C2_RS_LEADS", "C3_LOW_OVERHEAD", "C4_COMPRESSION", "C5_BASE_VOL_CONTRACTION", "C6_BREAKOUT_VOLUME",
        "C7_TIGHT_STOP"]


def stage1_breakouts(p, F, G, ctx) -> pd.DataFrame:
    from alpha.weinstein import WRULES as R, overhead_count
    S, wd = ctx.S, ctx.wd
    ti = p.dates.get_indexer(wd.week_end)
    st, eplen, sl = S["stage"].to_numpy(), S["ep_len"].to_numpy(), S["slope4"].to_numpy()
    ma = S["ma30w"].to_numpy()
    C = wd.close.to_numpy()
    Rb = np.fmin(S["ep_maxh"].to_numpy(), wd.high.rolling(52, min_periods=1).max().to_numpy())
    trig = np.fmax(Rb, ma) * (1 + R["TRIG_PAD"])
    oh = overhead_count(C, trig, R["OH_WINDOW"], R["OH_BAND"])
    U = p.universe.to_numpy()[ti] & (~(p.disp | p.disp_next)).to_numpy()[ti]
    base_ok = (st == 1) & (eplen >= R["BASE_MIN_W"]) & (sl >= R["W1_SLOPE_MIN"]) & U
    low4 = wd.low.rolling(4, min_periods=2).min().to_numpy()
    H, Cd, Ld = p.h.to_numpy(), p.c.to_numpy(), p.l.to_numpy()
    vol = p.vol.where(~p.disp).to_numpy()
    v5 = pd.DataFrame(vol).shift(1).rolling(5, min_periods=4).mean().to_numpy()
    lo10 = p.l.rolling(10, min_periods=8).min().to_numpy()
    atr = F["atr"].to_numpy()
    g = {k: G[k].to_numpy() for k in ("rs_leads60_10d", "rs_leads120_10d", "atr5_20", "bbw_pct120", "vol10_60",
                                       "supply15")}
    n_next = np.r_[np.diff(ti), 0]
    rows, last = [], {}
    for i, j in zip(*np.nonzero(base_ok)):
        if n_next[i] <= 0 or i + 1 >= len(ti):
            continue
        t = ti[i]
        seg = H[t + 1:t + 1 + n_next[i], j]
        hit = seg >= trig[i, j]
        if not hit.any():
            continue
        b = t + 1 + int(np.argmax(hit))
        if j in last and b - last[j] <= 20:
            continue
        last[j] = b
        if b + 1 >= len(p.dates):
            continue
        stop = lo10[b, j] * 0.995
        rows.append({
            "stock_id": p.ids[j], "j": j, "t": t, "b": b, "setup_date": p.dates[t], "breakout_date": p.dates[b],
            "trigger": trig[i, j], "close_b": Cd[b, j], "stop": stop, "stop_textbook": max(low4[i, j] * 0.99,
                                                                                         Cd[b, j] * 0.85),
            "atr_b": atr[b, j], "base_weeks": eplen[i, j], "oh15": oh[i, j],
            "C2_RS_LEADS": bool((g["rs_leads60_10d"][t, j] > 0) or (g["rs_leads120_10d"][t, j] > 0)),
            "C3_LOW_OVERHEAD": bool(oh[i, j] < R["OH_FAIL"]),
            "C4_COMPRESSION": bool((g["atr5_20"][t, j] <= 0.85) or (g["bbw_pct120"][t, j] <= 0.25)),
            "C5_BASE_VOL_CONTRACTION": bool(g["vol10_60"][t, j] <= 0.85),
            "C6_BREAKOUT_VOLUME": bool(vol[b, j] >= 2.0 * v5[b, j]) if np.isfinite(v5[b, j]) else False,
            "C7_TIGHT_STOP": bool((Cd[b, j] - stop) / atr[b, j] <= 2.0) if atr[b, j] > 0 else False,
            "stop_dist_atr_b": (Cd[b, j] - stop) / atr[b, j] if atr[b, j] > 0 else np.nan,
            "supply15": g["supply15"][t, j]})
    ev = pd.DataFrame(rows)
    ev["n_conditions"] = ev[COND].sum(axis=1)
    log(f"[userspec] stage-1 breakouts: {len(ev)}; all-7 = {int((ev['n_conditions'] == 6).sum())}")
    return ev


def simulate_events(M, ev: pd.DataFrame, exit_name: str, stop_col: str, label: str, end_idx: int) -> pd.DataFrame:
    from alpha.trades import simulate
    E = exit_specs()
    out = []
    for r in ev.itertuples():
        res = simulate(M, int(r.j), int(r.b), 1, "open", np.nan, getattr(r, stop_col), E[exit_name], end_idx, COST,
                       SLIP, order_days=1, level=r.trigger)
        if res:
            res.update({"engine": label, "variant": "USER_SPEC", "exit": exit_name, "ev_idx": r.Index, "j": int(r.j),
                        "stock_id": M.ids[int(r.j)],
                        "n_conditions": r.n_conditions, **{c: getattr(r, c) for c in COND}})
            out.append(res)
    df = pd.DataFrame(out)
    if len(df):
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df = tag_windows(df)
    return df


def hrr_stage2_userspec(p, F, G, ctx, M, end_idx):
    ev = stage1_breakouts(p, F, G, ctx)
    ev.to_pickle(P2 / "userspec_stage1_breakouts.pkl")
    rows = []
    trades = {}
    for exn in ("TEXTBOOK_NOVOL", "MODERN_NOVOL"):
        tr = simulate_events(M, ev, exn, "stop", "STAGE1_BREAKOUT_ALL", end_idx)
        tr = no_overlap(tr)
        trades[exn] = tr
        tr.to_pickle(P2 / f"userspec_breakout_trades_{exn}.pkl")
        groups = {"ALL_STAGE1_BREAKOUTS": tr["n_conditions"] >= 0,
                  "HIGH_RR_STAGE2_SETUP(C1+all 6 conditions)": tr["n_conditions"] == 6}
        for k in (0, 1, 2, 3, 4, 5, 6):
            groups[f"n_conditions={k}"] = tr["n_conditions"] == k
        for c in COND:
            groups[f"{c}=1"] = tr[c]
            groups[f"{c}=0"] = ~tr[c]
        for gname, m in groups.items():
            sub = tr[m].assign(engine=gname)
            if len(sub) == 0:
                continue
            t_ = stats_table(sub, ["engine"])
            t_.insert(1, "exit", exn)
            rows.append(t_)
    res = pd.concat(rows, ignore_index=True)
    # MAE of failures, MFE/MAE
    extra = []
    for exn, tr in trades.items():
        for w, (a, b) in {k: WINDOWS[k] for k in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS")}.items():
            x = tr[(tr["entry_date"] >= a) & (tr["entry_date"] <= b)]
            for gname, m in (("HIGH_RR_STAGE2_SETUP(C1+all 6 conditions)", x["n_conditions"] == 6),
                             ("n_conditions>=4", x["n_conditions"] >= 4), ("ALL_STAGE1_BREAKOUTS", x["n_conditions"] >= 0)):
                s_ = x[m]
                f_ = s_[s_["ret"] < 0]
                extra.append({"engine": gname, "exit": exn, "window": w, "table": "FAILURE_MAE",
                              "n": len(s_), "mae_failed_mean": f_["mae"].mean() if len(f_) else np.nan,
                              "mfe_mae_ratio": s_["mfe"].mean() / -s_["mae"].mean() if len(s_) and s_["mae"].mean() < 0 else np.nan,
                              "placebo_p_vs_rest": bootstrap_p(s_["ret"].to_numpy(), x[~m]["ret"].to_numpy())
                              if gname != "ALL_STAGE1_BREAKOUTS" else np.nan})
    # same-day random control: 2 random universe stocks per breakout day, same stop rule (10-day low * 0.995 at b),
    # entry next open, same exit -> does the Stage-1 breakout SELECTION add anything over "trend exit in this market"?
    rng = np.random.default_rng(17)
    U = p.universe.to_numpy()
    lo10 = p.l.rolling(10, min_periods=8).min().to_numpy()
    busy = set(zip(ev["b"], ev["j"]))
    crow = []
    for r in ev.itertuples():
        cand = [c for c in np.nonzero(U[r.b])[0] if (r.b, c) not in busy]
        for c in rng.choice(cand, size=min(2, len(cand)), replace=False):
            crow.append({"j": int(c), "b": r.b, "stop": lo10[r.b, c] * 0.995, "trigger": np.nan, "n_conditions": -1,
                         **{k: False for k in COND}})
    cev = pd.DataFrame(crow)
    ctl_rows = []
    for exn in ("TEXTBOOK_NOVOL", "MODERN_NOVOL"):
        ct = simulate_events(M, cev, exn, "stop", "RANDOM_SAME_DAY_CONTROL", end_idx)
        ct = no_overlap(ct, keys=("engine", "variant", "exit", "j"))
        t_ = stats_table(ct, ["engine"])
        t_.insert(1, "exit", exn)
        ctl_rows.append(t_.assign(table="TRADE_STATS"))
        tr = trades[exn]
        for w, (a, b) in {k: WINDOWS[k] for k in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS")}.items():
            x = tr[(tr["entry_date"] >= a) & (tr["entry_date"] <= b)]["ret"].to_numpy()
            y = ct[(ct["entry_date"] >= a) & (ct["entry_date"] <= b)]["ret"].to_numpy()
            ctl_rows.append(pd.DataFrame([{"engine": "ALL_STAGE1_BREAKOUTS_vs_RANDOM_SAME_DAY", "exit": exn, "window": w,
                                           "table": "PLACEBO", "n": len(x), "ev": np.mean(x) if len(x) else np.nan,
                                           "ev_control": np.mean(y) if len(y) else np.nan,
                                           "placebo_p": bootstrap_p(x, y)}]))
    res = pd.concat([res.assign(table="TRADE_STATS"), pd.DataFrame(extra)] + ctl_rows, ignore_index=True)
    return ev, trades, res


def hybrid_c_userspec(M, ev: pd.DataFrame, end_idx: int) -> pd.DataFrame:
    m = ev["C2_RS_LEADS"] & ev["C3_LOW_OVERHEAD"] & ev["C4_COMPRESSION"] & ev["C6_BREAKOUT_VOLUME"]
    tr = simulate_events(M, ev[m], "TEXTBOOK_NOVOL", "stop_textbook", "HYBRID_C", end_idx)
    return no_overlap(tr)


def hybrid_d_userspec(p, ctx, M, mom: pd.DataFrame, end_idx: int, probe_size: float = 0.25,
                      confirm_days: int = 20) -> pd.DataFrame:
    """Momentum probe (0.25) -> Weinstein confirmation (weekly stage 2 & weekly close > prior 6-week high, at a
    week end within `confirm_days`) -> add 0.75 at next open -> MODERN exit rules on the full position."""
    O, H, L, C = M.O, M.H, M.L, M.C
    wd = ctx.wd
    W_close = wd.close.to_numpy()
    W_high = wd.high.to_numpy()
    prior6 = pd.DataFrame(W_high).shift(1).rolling(6, min_periods=6).max().to_numpy()
    low4w = wd.low.rolling(4, min_periods=2).min().to_numpy()
    wpos = {d: k for k, d in enumerate(wd.week_end)}
    stage = ctx.S["stage"].to_numpy()
    out = []
    for r in mom.itertuples():
        j = int(r.j)
        e = p.dates.get_loc(r.entry_date)
        px0 = r.entry_price_adj
        stop = r.probe_stop_adj
        if not (np.isfinite(px0) and np.isfinite(stop) and stop < px0):
            continue
        size, cost_acc, realized, slot_days = probe_size, COST * probe_size, 0.0, 0.0
        legs = [(probe_size, px0)]
        added, add_day, exit_day, reason = False, None, None, ""
        peak = px0
        d = e
        last = min(end_idx, len(p.dates) - 1)
        pending = None
        conf_k = -1
        while d <= last:
            o, hi, lo, c = O[d, j], H[d, j], L[d, j], C[d, j]
            if not np.isfinite(c):
                d += 1
                continue
            if pending is not None and d > e and np.isfinite(o):
                kind = pending
                pending = None
                if kind == "ADD":
                    pxa = o * (1 + SLIP)
                    legs.append((1 - probe_size, pxa))
                    size = 1.0
                    cost_acc += COST * (1 - probe_size)
                    added, add_day = True, d
                    lw = low4w[conf_k, j]
                    stop = max(stop, lw * 0.99) if np.isfinite(lw) else stop
                else:
                    xp = o * (1 - SLIP)
                    exit_day, reason = d, kind
                    break
            if lo <= stop:
                xp = min(o, stop) * (1 - SLIP) if d > e else stop * (1 - SLIP)
                exit_day, reason = d, "STOP" if not added else "STOP_AFTER_ADD"
                break
            slot_days += size
            peak = max(peak, hi)
            if d >= end_idx:
                xp = c
                exit_day, reason = d, "END_OF_WINDOW"
                break
            avg_px = sum(s * px_ for s, px_ in legs) / sum(s for s, _ in legs)
            if M.WE[d]:
                k = wpos.get(p.dates[d])
                if k is not None:
                    if not added and d - e < confirm_days and stage[k, j] == 2 and W_close[k, j] > prior6[k, j]:
                        pending = "ADD"
                        conf_k = k
                    elif added:
                        if stage[k, j] == 4:
                            pending = "STAGE4_EXIT"
                        elif np.isfinite(M.MA10W_WE[d, j]) and c < M.MA10W_WE[d, j]:
                            pending = "MA10W_EXIT"
            if not added and d - e + 1 >= confirm_days and pending is None:
                pending = "NO_CONFIRMATION_TIME_EXIT"
            if added:
                risk = (avg_px - stop) / avg_px if avg_px > stop else np.nan
                if np.isfinite(risk) and (peak / avg_px - 1) >= 2 * risk and avg_px > stop and avg_px < c:
                    stop = avg_px
            d += 1
        if exit_day is None:
            exit_day, xp, reason = last, C[last, j], "END_OF_DATA"
        gross = sum(s * (xp / px_ - 1) for s, px_ in legs)
        pnl_slots = gross - cost_acc
        out.append({"engine": "HYBRID_D", "variant": "USER_SPEC", "exit": "PROBE_CONFIRM_ADD", "stock_id": r.stock_id,
                    "j": j, "side": "LONG", "entry_date": p.dates[e], "exit_date": p.dates[exit_day],
                    "entry_price_adj": px0, "added": added, "add_date": p.dates[add_day] if add_day else pd.NaT,
                    "exit_reason": reason, "first_exit_reason": reason, "size": size,
                    "pnl_slots": pnl_slots, "ret": pnl_slots / size, "slot_days": slot_days,
                    "holding_days": exit_day - e + 1, "stop_dist_pct": r.stop_dist_pct,
                    "r_multiple": (pnl_slots / size) / r.stop_dist_pct if r.stop_dist_pct else np.nan,
                    "mfe": np.nan, "mae": np.nan, "stop_dist_atr": np.nan})
    df = pd.DataFrame(out)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return tag_windows(df)
