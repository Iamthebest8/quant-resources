"""Phase-2 research pipeline: Weinstein (long / short), High R/R research, exits, overlap, portfolio.

    python -m pipeline.phase2 state        # build + cache panel / features / geometry / weekly context
    python -m pipeline.phase2 momentum     # MOMENTUM_LONG_BASELINE reproduction + trade list
    python -m pipeline.phase2 weinstein    # W1/W2/W3/S1/S2 textbook + modernized + exit research
    python -m pipeline.phase2 highrr       # High R/R feature research (Discovery bins -> frozen score -> OOS)
    python -m pipeline.phase2 portfolio    # alpha overlap, hybrids, multi-alpha 10-slot portfolio

All rules come from WEINSTEIN_QUANT_RULES.md (frozen before results). Windows: PRE 2020-2022 (extra history),
DISCOVERY 2023-2024, STRICT_OOS 2025-01-01 .. 2026-09-03 (never used to choose anything).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

OUT = config.OUT_DIR
P2 = OUT / "phase2"
P2.mkdir(parents=True, exist_ok=True)
ROOT = config.ROOT
STATE = config.PIT_DIR / "phase2_state.pkl"
COST = config.BASE_COST
SLIP = config.BASE_SLIPPAGE_BPS / 10000
END = config.BACKTEST_END
WINDOWS = {
    "PRE_2020_2022": ("2020-01-01", "2022-12-31"),
    "DISCOVERY": config.DISCOVERY,
    "STRICT_OOS": config.STRICT_OOS,
    "EXTENDED": config.EXTENDED_VALIDATION,
    "FULL_2023_2026": ("2023-01-01", END),
}
YEARS = [str(y) for y in range(2020, 2027)]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_csv(df: pd.DataFrame, name: str, root: bool = True) -> Path:
    base = (ROOT if not config.IS_SYNTHETIC else OUT) if root else P2      # synthetic runs never touch the repo root
    path = base / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"wrote {path} ({len(df)} rows)")
    return path


# ---------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------
def build_state():
    from alpha.weinstein import build_context
    from engine.features import compute_features
    from engine.geometry import compute_geometry
    from engine.panel import build_panel
    t0 = time.time()
    p = build_panel(log=log)
    F = compute_features(p, log=log)
    G = compute_geometry(p, F, log=log)
    ctx = build_context(p, log=log)
    with open(STATE, "wb") as fh:
        pickle.dump({"p": p, "F": F, "G": G, "ctx": ctx, "built_utc": now_utc()}, fh, protocol=5)
    log(f"state built in {time.time() - t0:.0f}s -> {STATE}")
    return p, F, G, ctx


def load_state():
    if not STATE.exists():
        return build_state()
    with open(STATE, "rb") as fh:
        s = pickle.load(fh)
    log(f"loaded state built {s['built_utc']}")
    return s["p"], s["F"], s["G"], s["ctx"]


def window_of(d: pd.Timestamp) -> str:
    for k in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS"):
        a, b = WINDOWS[k]
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return k
    return "OUTSIDE"


def tag_windows(df: pd.DataFrame, col: str = "entry_date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df["window"] = df[col].map(window_of)
    df["year"] = df[col].dt.year.astype(str)
    return df


def no_overlap(df: pd.DataFrame, keys=("engine", "variant", "exit", "stock_id")) -> pd.DataFrame:
    """One open position per (engine, variant, exit, stock): drop trades entering before the previous exits."""
    if df.empty:
        return df
    df = df.sort_values(list(keys) + ["entry_date"]).reset_index(drop=True)
    keep = np.ones(len(df), bool)
    last_key, last_exit = None, None
    for i, r in enumerate(df[list(keys) + ["entry_date", "exit_date"]].itertuples(index=False)):
        k = tuple(r[:len(keys)])
        if k == last_key and r[-2] <= last_exit:
            keep[i] = False
            continue
        last_key, last_exit = k, r[-1]
    return df[keep].reset_index(drop=True)


def stats_table(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    from alpha.trades import trade_stats
    rows = []
    for keys, g in trades.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        for w, (a, b) in WINDOWS.items():
            sub = g[(g["entry_date"] >= a) & (g["entry_date"] <= b)]
            yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
            rows.append({**base, "window": w, **trade_stats(sub, years=yrs)})
        for y in YEARS:
            sub = g[g["entry_date"].dt.year == int(y)]
            if len(sub):
                rows.append({**base, "window": f"Y{y}", **trade_stats(sub, years=1.0 if y != "2026" else 0.67)})
    return pd.DataFrame(rows)


def bootstrap_p(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 7) -> float:
    """One-sided p-value that mean(a) > mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    obs = a.mean() - b.mean()
    cnt = 0
    for _ in range(n):
        x = rng.choice(pooled, len(a))
        y = rng.choice(pooled, len(b))
        cnt += (x.mean() - y.mean()) >= obs
    return (cnt + 1) / (n + 1)


# ---------------------------------------------------------------------------------------------
# momentum baseline
# ---------------------------------------------------------------------------------------------
def momentum(p, F):
    """Reproduce the frozen MOMENTUM_LONG_BASELINE (V1) on the phase-2 panel and write MOMENTUM_LONG_RESULTS.csv."""
    from pipeline.phase2_portfolio import momentum_trades
    from pipeline.run_research import load_frozen, run_tl
    from strategy.metrics import campaign_metrics
    from strategy.selection import SignalCache
    from strategy.signals import build_arrays
    cfg, frozen = load_frozen("V1")
    A = build_arrays(p, F)
    sc = SignalCache(p, F)
    ref = pd.read_csv(OUT / "frozen" / "MOMENTUM_LONG_BASELINE_reference_metrics.csv").set_index("window")
    W = {"DISCOVERY": config.DISCOVERY, "EXTENDED": config.EXTENDED_VALIDATION, "STRICT_OOS": config.STRICT_OOS,
         "FULL": (config.RESEARCH_START, END)}
    keys = ("n_probes", "pf", "payoff", "ev_slots_per_probe", "pnl_per_100_slot_days", "win_rate", "ge20",
            "top5pct_share")
    rows = []
    for k, w in W.items():
        tl = run_tl(A, sc, cfg, *w)
        m = campaign_metrics(tl, years=(pd.Timestamp(w[1]) - pd.Timestamp(w[0])).days / 365.25)
        r = {"table": "REPRODUCTION", "window": k, "source": "phase2_panel(DATA_START=2019-01-01)"}
        r.update({x: m.get(x) for x in keys})
        rows.append(r)
        rr = {"table": "REPRODUCTION", "window": k, "source": "phase1_reference(frozen)"}
        rr.update({x: ref.at[k, x] if k in ref.index and x in ref.columns else np.nan for x in keys})
        rows.append(rr)
        same = all(np.isclose(float(r[x] or np.nan), float(rr[x]), rtol=1e-6, equal_nan=True) for x in keys)
        rows.append({"table": "REPRODUCTION", "window": k, "source": "MATCH" if same else "DIFFERS"})
    rep = pd.DataFrame(rows)
    log("momentum reproduction:\n" + rep.to_string())
    mt = tag_windows(momentum_trades(p))
    st_ = stats_table(mt, ["engine"])
    st_.insert(0, "table", "TRADE_STATS_PER_CAMPAIGN(ret on max size)")
    out = pd.concat([rep, st_], ignore_index=True)
    save_csv(out, "MOMENTUM_LONG_RESULTS.csv")
    return out


# ---------------------------------------------------------------------------------------------
# Weinstein
# ---------------------------------------------------------------------------------------------
def exit_specs():
    from alpha.trades import ExitSpec
    E = {
        "TEXTBOOK_INVESTOR": ExitSpec("TEXTBOOK_INVESTOR", stage_exit="investor", weak_vol="weekly"),
        "TEXTBOOK_TRADER": ExitSpec("TEXTBOOK_TRADER", stage_exit="trader", weak_vol="weekly"),
        "TEXTBOOK_NOVOL": ExitSpec("TEXTBOOK_NOVOL", stage_exit="investor"),           # W3 / shorts
        "TEXTBOOK_TRADER_NOVOL": ExitSpec("TEXTBOOK_TRADER_NOVOL", stage_exit="trader"),
        "MODERN": ExitSpec("MODERN", stage_exit="modern", weak_vol="daily", ma10w_exit=True, breakeven_r=2.0),
        "MODERN_NOVOL": ExitSpec("MODERN_NOVOL", stage_exit="modern", ma10w_exit=True, breakeven_r=2.0),
        "STOP_STAGE4_ONLY": ExitSpec("STOP_STAGE4_ONLY", stage_exit="modern"),
    }
    for b in ("B1", "B2"):
        for n, k in ((20, 2.0), (20, 2.5), (30, 2.0)):
            for part in (1.0, 0.5):
                nm = f"BB_{b}_{n}_{k:g}_{'FULL' if part == 1.0 else 'HALF'}"
                E[nm] = ExitSpec(nm, bb=b, bb_n=n, bb_k=k, bb_partial=part,
                                 stage_exit="modern" if part == 1.0 else "investor")
    return E


TEXTBOOK_EXIT = {"W1": "TEXTBOOK_INVESTOR", "W2": "TEXTBOOK_TRADER", "W3": "TEXTBOOK_NOVOL",
                 "S1": "TEXTBOOK_NOVOL", "S2": "TEXTBOOK_NOVOL"}
MODERN_EXIT = {"W1": "MODERN", "W2": "MODERN", "W3": "MODERN_NOVOL", "S1": "MODERN_NOVOL", "S2": "MODERN_NOVOL"}


def modern_rows(setups: pd.DataFrame) -> pd.DataFrame:
    """MODERNIZED variant: same structural events, research filters + ATR stops (QUANT_RULES §J)."""
    from alpha.weinstein import WRULES as R
    s = setups.copy()
    a = s["atr"]
    if "trigger" in s and s["engine"].isin(["W1", "W2"]).any():
        lo = s["swl10"] * 0.995
        struct = (s["trigger"] - lo) / a
        stop_mod = np.maximum(lo, s["trigger"] - R["MOD_STOP_ATR"] * a)
        longm = s["engine"].isin(["W1", "W2"])
        ok = (longm & (s["ex120_pct"] >= R["MOD_RS_PCT"]) & (s["supply15"] <= R["MOD_SUPPLY15"])
              & (struct <= R["MOD_STOP_ATR_MAX"]) & (struct > 0))
        s.loc[longm, "stop"] = stop_mod[longm]
        s.loc[longm, "modern"] = ok[longm]
    if (s["engine"] == "S1").any():
        m = s["engine"] == "S1"
        s.loc[m, "stop"] = np.minimum(s.loc[m, "stop"], s.loc[m, "trigger"] + R["MOD_STOP_ATR"] * a[m])
        s.loc[m, "modern"] = True
    for k in ("W3", "S2"):
        m = s["engine"] == k
        if m.any():
            s.loc[m, "stop"] = s.loc[m, "stop_mod"]
            s.loc[m, "modern"] = True
    s["modern"] = s["modern"].fillna(False).astype(bool)
    return s[s["modern"]]


def simulate_setups(M, setups: pd.DataFrame, exit_name: str, E: dict, variant: str, end_idx: int,
                    cost: float = COST, slip: float = SLIP) -> pd.DataFrame:
    from alpha.trades import simulate
    ex = E[exit_name]
    out = []
    for r in setups.itertuples():
        side = int(r.side)
        if r.engine in ("W3", "S2"):
            res = simulate(M, r.j, r.t, side, "open", np.nan, r.stop, ex, end_idx, cost, slip, order_days=1,
                           level=r.level)
        else:
            res = simulate(M, r.j, r.t, side, "stoplimit", r.trigger, r.stop, ex, end_idx, cost, slip,
                           order_days=r.order_days, limit=r.limit, level=r.level)
        if res is None:
            continue
        res.update({"engine": r.engine, "variant": variant, "exit": exit_name, "setup_idx": r.Index, "t": r.t,
                    "j": r.j, "level": r.level})
        out.append(res)
    df = pd.DataFrame(out)
    if len(df):
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


def matched_controls(M, p, trades: pd.DataFrame, E: dict, end_idx: int, setups_all: pd.DataFrame,
                     n_ctrl: int = 2, seed: int = 11) -> pd.DataFrame:
    """Placebo: same entry day, random universe stock with NO setup of the same engine that week, entered at the
    open with the SAME stop distance (%) and the SAME exit rules. Isolates the selection/timing alpha."""
    from alpha.trades import simulate
    rng = np.random.default_rng(seed)
    U = p.universe.to_numpy()
    O = M.O
    busy = set(zip(setups_all["t"], setups_all["j"]))
    out = []
    for r in trades.itertuples():
        e = p.dates.get_loc(r.entry_date)
        t = e - 1
        if t < 0:
            continue
        cand = np.nonzero(U[t])[0]
        cand = [c for c in cand if (t, c) not in busy and c != r.j]
        if not cand:
            continue
        for c in rng.choice(cand, size=min(n_ctrl, len(cand)), replace=False):
            o = O[e, c]
            if not np.isfinite(o):
                continue
            side = 1 if r.side == "LONG" else -1
            stop = o * (1 - side * r.stop_dist_pct)
            res = simulate(M, int(c), t, side, "open", np.nan, stop, E[r.exit], end_idx, COST, SLIP, order_days=1,
                           level=o)
            if res is None:
                continue
            res.update({"engine": r.engine, "variant": r.variant, "exit": r.exit, "control_for": r.Index})
            out.append(res)
    df = pd.DataFrame(out)
    if len(df):
        df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def funnels(p, ctx) -> pd.DataFrame:
    """Condition funnel (stock-weeks passing each TEXTBOOK condition, cumulative) for W1 / W2 / S1 per window."""
    from alpha.weinstein import WRULES as R, overhead_count
    S, wd = ctx.S, ctx.wd
    ti = p.dates.get_indexer(wd.week_end)
    st, eplen, sl = S["stage"].to_numpy(), S["ep_len"].to_numpy(), S["slope4"].to_numpy()
    mrs, rs4, rs13 = ctx.mrs.to_numpy(), ctx.rs_slope4.to_numpy(), ctx.rs_slope13.to_numpy()
    mkt = ctx.mkt["stage"].to_numpy()[:, 0][:, None]
    g, gna = ctx.grp_stage_w.to_numpy(), ctx.grp_na.to_numpy()
    U = p.universe.to_numpy()[ti] & (~(p.disp | p.disp_next)).to_numpy()[ti]
    C, H, L, ma = wd.close.to_numpy(), wd.high.to_numpy(), wd.low.to_numpy(), S["ma30w"].to_numpy()
    Rb = np.fmin(S["ep_maxh"].to_numpy(), wd.high.rolling(52, min_periods=1).max().to_numpy())
    R6 = wd.high.rolling(6, min_periods=6).max().to_numpy()
    Lo6 = wd.low.rolling(6, min_periods=6).min().to_numpy()
    max2 = wd.high.rolling(2, min_periods=2).max().to_numpy()
    oh1 = overhead_count(C, np.fmax(Rb, ma) * 1.003, R["OH_WINDOW"], R["OH_BAND"])
    oh2 = overhead_count(C, R6 * 1.003, R["OH_WINDOW"], R["OH_BAND"])
    rs_long = (rs13 >= 0) & ((mrs >= 0) | (rs4 > 0))
    rs_short = (mrs < 0) | (rs13 < 0)
    chains = {
        "W1": [("stage==1", st == 1), ("base>=8w", eplen >= R["BASE_MIN_W"]), ("MA not declining", sl >= -0.01),
               ("RS_OK_LONG", rs_long), ("OH15<4", oh1 < R["OH_FAIL"]), ("TAIEX stage 1/2", np.isin(mkt, [1, 2])),
               ("group stage 1/2", gna | np.isin(g, [1, 2]))],
        "W2": [("stage==2", st == 2), ("stage2 age>=8w", eplen >= R["W2_AGE_MIN"]), ("MA slope>=2%", sl >= 0.02),
               ("6w high >=2w old", max2 < R6), ("range<=25%", R6 / Lo6 - 1 <= 0.25),
               ("low near MA (0.97-1.10)", (Lo6 >= ma * 0.97) & (Lo6 <= ma * 1.10)), ("RS_OK_LONG", rs_long),
               ("OH15<4", oh2 < R["OH_FAIL"]), ("TAIEX stage 1/2", np.isin(mkt, [1, 2])),
               ("group stage 1/2", gna | np.isin(g, [1, 2]))],
        "S1": [("stage==3", st == 3), ("top>=6w", eplen >= R["TOP_MIN_W"]), ("MA not rising", sl <= 0.01),
               ("RS_OK_SHORT", rs_short), ("group stage 3/4", (~gna) & np.isin(g, [3, 4])),
               ("TAIEX stage 3/4", np.isin(mkt, [3, 4]))],
    }
    yr = wd.week_end
    rows = []
    for w, (a, b) in {k: WINDOWS[k] for k in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS")}.items():
        wm = np.asarray((yr >= pd.Timestamp(a)) & (yr <= pd.Timestamp(b)))[:, None]
        for eng, chain in chains.items():
            m = U & wm
            rows.append({"engine": eng, "window": w, "step": 0, "condition": "universe stock-weeks", "n": int(m.sum())})
            for k, (nm, cond) in enumerate(chain, 1):
                m = m & cond
                rows.append({"engine": eng, "window": w, "step": k, "condition": nm, "n": int(m.sum())})
    return pd.DataFrame(rows)


def weinstein(p, F, G, ctx):
    from alpha.trades import attach_weekly, build_mats, trade_stats
    from alpha.weinstein import breakout_events, followup_setups, weekly_setups
    t0 = time.time()
    setups = weekly_setups(p, F, G, ctx, log=log)
    ev_w1 = breakout_events(p, setups, "W1")
    ev_s1 = breakout_events(p, setups, "S1")
    w3 = followup_setups(p, F, ctx, ev_w1, "W3", log=log)
    s2 = followup_setups(p, F, ctx, ev_s1, "S2", log=log)
    allset = pd.concat([setups, w3, s2], ignore_index=True)
    fn = funnels(p, ctx)
    fn.to_csv(P2 / "WEINSTEIN_FUNNEL.csv", index=False)
    allset.to_pickle(P2 / "weinstein_setups.pkl")
    M = attach_weekly(build_mats(p, F, G), p, ctx)
    end_idx = int(p.dates.searchsorted(pd.Timestamp(END), side="right") - 1)
    E = exit_specs()
    text = allset[allset["textbook"]]
    mod = modern_rows(allset)
    log(f"setups textbook={len(text)} modern={len(mod)}")
    frames = []
    for eng in ("W1", "W2", "W3", "S1", "S2"):
        st = text[text["engine"] == eng]
        sm = mod[mod["engine"] == eng]
        tb = no_overlap(simulate_setups(M, st, TEXTBOOK_EXIT[eng], E, "TEXTBOOK", end_idx))
        frames.append(tb)
        frames.append(no_overlap(simulate_setups(M, sm, MODERN_EXIT[eng], E, "MODERNIZED", end_idx)))
        # exit research on EXACTLY the frozen TEXTBOOK entries (same setups, same fills; no re-filtering per exit)
        st_fixed = st.loc[tb["setup_idx"].unique()] if len(tb) else st.iloc[:0]
        others = ["STOP_STAGE4_ONLY"] + [k for k in E if k.startswith("BB_")]
        others += [MODERN_EXIT[eng]] if MODERN_EXIT[eng] != TEXTBOOK_EXIT[eng] else []
        for xn in others:
            frames.append(simulate_setups(M, st_fixed, xn, E, "TEXTBOOK_ENTRY_EXITRESEARCH", end_idx))
        log(f"  {eng} simulated ({time.time() - t0:.0f}s)")
    tr = pd.concat([f for f in frames if len(f)], ignore_index=True)
    tr = tag_windows(tr)
    tr.to_pickle(P2 / "weinstein_trades.pkl")
    log(f"weinstein trades: {len(tr)} in {time.time() - t0:.0f}s")
    return allset, tr, M, E, end_idx


def short_executability(p, tr: pd.DataFrame) -> pd.DataFrame:
    """Flag each short trade: PLAIN_TICK_OK (fill >= previous close: allowed for any marginable stock),
    EXEMPT_PROXY (top-150 TWSE + top-50 TPEx market value that month ~ Taiwan 50 / Mid-Cap 100 / TPEx 50 constituents,
    which may short below the previous close), MARGIN_ELIGIBLE_PROXY (listed >= 120 trading days, not disposition)."""
    tr = tr.copy()
    if tr.empty:
        return tr
    e = p.dates.get_indexer(tr["entry_date"])
    j = tr["j"].astype(int).to_numpy()
    prev_raw = p.raw_c.to_numpy()[e - 1, j]
    tr["prev_close_raw"] = prev_raw
    tr["plain_tick_ok"] = tr["entry_price"].to_numpy() >= prev_raw * 0.9999
    mv = p.mktval
    exempt = np.zeros(len(tr), bool)
    if mv is not None:
        mvv = mv.reindex(index=p.dates, columns=p.ids).ffill().to_numpy()
        typ = p.stock_type.reindex(p.ids).to_numpy()
        for k, (ee, jj) in enumerate(zip(e, j)):
            row = mvv[ee - 1]
            if not np.isfinite(row[jj]):
                continue
            tw = (typ == "twse") & np.isfinite(row)
            tp = (typ == "tpex") & np.isfinite(row)
            if typ[jj] == "twse":
                exempt[k] = (row[tw] > row[jj]).sum() < 150
            elif typ[jj] == "tpex":
                exempt[k] = (row[tp] > row[jj]).sum() < 50
    tr["exempt_proxy"] = exempt
    ld = p.listed_days.to_numpy()[e - 1, j]
    dsp = (p.disp | p.disp_next).to_numpy()[e - 1, j]
    tr["margin_eligible_proxy"] = (ld >= 120) & ~dsp
    tr["executable"] = tr["margin_eligible_proxy"] & (tr["plain_tick_ok"] | tr["exempt_proxy"])
    return tr


def weinstein_reports(p, F, G, ctx, allset, tr, M, E, end_idx):
    from alpha.trades import trade_stats
    from alpha.verdicts2 import bb_verdict, entry_verdict, short_verdict
    out = {}
    main = tr[tr["variant"].isin(["TEXTBOOK", "MODERNIZED"])]
    # ---------- placebo (matched controls) ----------
    ctrl = matched_controls(M, p, main, E, end_idx, allset)
    ctrl = tag_windows(ctrl) if len(ctrl) else ctrl
    ctrl.to_pickle(P2 / "weinstein_controls.pkl")
    plac = []
    for (eng, var), g in main.groupby(["engine", "variant"]):
        c = ctrl[(ctrl["engine"] == eng) & (ctrl["variant"] == var)] if len(ctrl) else ctrl
        for w in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS"):
            a = g[g["window"] == w]["ret"].to_numpy()
            b = c[c["window"] == w]["ret"].to_numpy() if len(c) else np.array([])
            plac.append({"engine": eng, "variant": var, "window": w, "n": len(a), "ev_signal": np.mean(a) if len(a) else np.nan,
                         "n_control": len(b), "ev_control": np.mean(b) if len(b) else np.nan,
                         "placebo_p": bootstrap_p(a, b)})
    plac = pd.DataFrame(plac)
    out["placebo"] = plac
    # ---------- long results ----------
    longs = main[main["side"] == "LONG"]
    lt = stats_table(longs, ["engine", "variant", "exit"])
    pooled = stats_table(longs.assign(engine="W_LONG_POOLED"), ["engine", "variant"])
    pooled["exit"] = "ENGINE_DEFAULT"
    lt = pd.concat([lt, pooled], ignore_index=True)
    lt = lt.merge(plac, on=["engine", "variant", "window"], how="left", suffixes=("", "_plc"))
    # cost / slippage grid (Strict OOS) by re-simulation of the textbook / modern entries
    grid = []
    for (eng, var), g in longs.groupby(["engine", "variant"]):
        st_rows = allset.loc[g["setup_idx"].unique()]
        if var == "MODERNIZED":
            st_rows = modern_rows(allset).loc[lambda x: x.index.isin(g["setup_idx"].unique())]
        exn = g["exit"].iloc[0]
        for cst in config.COST_GRID:
            for sb in config.SLIPPAGE_GRID_BPS:
                if cst == COST and sb == config.BASE_SLIPPAGE_BPS:
                    sim = g
                else:
                    sim = simulate_setups(M, st_rows, exn, E, var, end_idx, cost=cst, slip=sb / 10000)
                    sim = no_overlap(sim)
                if len(sim) == 0:
                    continue
                sim = sim[(sim["entry_date"] >= WINDOWS["STRICT_OOS"][0]) & (sim["entry_date"] <= WINDOWS["STRICT_OOS"][1])]
                grid.append({"engine": eng, "variant": var, "exit": exn, "window": "STRICT_OOS_COST_GRID",
                             "round_trip_cost": cst, "slippage_bps": sb, **trade_stats(sim)})
    lt = pd.concat([lt, pd.DataFrame(grid)], ignore_index=True)
    save_csv(lt, "WEINSTEIN_LONG_RESULTS.csv")
    out["long"] = lt
    # entry verdicts (TEXTBOOK = the pre-registered primary; MODERNIZED reported)
    ver = {}
    for eng in ("W1", "W2", "W3"):
        for var in ("TEXTBOOK", "MODERNIZED"):
            g = longs[(longs["engine"] == eng) & (longs["variant"] == var)]
            o = g[g["window"] == "STRICT_OOS"]
            m = trade_stats(o)
            pr = plac[(plac["engine"] == eng) & (plac["variant"] == var) & (plac["window"] == "STRICT_OOS")]
            ver[(eng, var)] = entry_verdict({"n": m.get("n"), "pf": m.get("pf"), "payoff": m.get("payoff"),
                                             "ev": m.get("ev"), "placebo_p": pr["placebo_p"].iloc[0] if len(pr) else np.nan,
                                             "ev_2025": o[o["year"] == "2025"]["ret"].mean(),
                                             "ev_2026": o[o["year"] == "2026"]["ret"].mean()})
    out["entry_verdicts"] = ver
    # ---------- shorts ----------
    shorts = short_executability(p, main[main["side"] == "SHORT"])
    srows = []
    for lab, sub in (("THEORETICAL", shorts), ("EXECUTABLE", shorts[shorts["executable"]] if len(shorts) else shorts)):
        if len(sub):
            t = stats_table(sub, ["engine", "variant", "exit"])
            t["executability"] = lab
            srows.append(t)
            t2 = stats_table(sub.assign(engine="S_POOLED"), ["engine", "variant"])
            t2["executability"] = lab
            t2["exit"] = "ENGINE_DEFAULT"
            srows.append(t2)
    exe_tab = []
    if len(shorts):
        for (eng, var, w), g in shorts.groupby(["engine", "variant", "window"]):
            exe_tab.append({"engine": eng, "variant": var, "window": w, "executability": "AUDIT", "n": len(g),
                            "plain_tick_ok": g["plain_tick_ok"].mean(), "exempt_proxy": g["exempt_proxy"].mean(),
                            "margin_eligible_proxy": g["margin_eligible_proxy"].mean(),
                            "executable": g["executable"].mean()})
    st_ = pd.concat(srows + [pd.DataFrame(exe_tab)], ignore_index=True) if srows else pd.DataFrame(exe_tab)
    st_ = st_.merge(plac, on=["engine", "variant", "window"], how="left", suffixes=("", "_plc")) if len(st_) else st_
    save_csv(st_, "WEINSTEIN_SHORT_RESULTS.csv")
    shorts.to_pickle(P2 / "weinstein_shorts_exec.pkl")
    out["short"] = st_
    sv = {}
    for var in ("TEXTBOOK", "MODERNIZED"):
        for lab in ("THEORETICAL", "EXECUTABLE"):
            g = shorts[(shorts["variant"] == var)] if len(shorts) else shorts
            if lab == "EXECUTABLE" and len(g):
                g = g[g["executable"]]
            o = g[g["window"] == "STRICT_OOS"] if len(g) else g
            m = trade_stats(o)
            sv[(var, lab)] = short_verdict({"n": m.get("n", 0), "pf": m.get("pf"), "ev": m.get("ev")},
                                           executability_verified=False)
    out["short_verdicts"] = sv
    # ---------- exit research ----------
    xr = tr[tr["variant"].isin(["TEXTBOOK", "TEXTBOOK_ENTRY_EXITRESEARCH"])]
    xt = stats_table(xr, ["engine", "side", "exit"])
    xt_pool = stats_table(xr[xr["side"] == "LONG"].assign(engine="W_LONG_POOLED"), ["engine", "side", "exit"])
    xt = pd.concat([xt, xt_pool], ignore_index=True)
    # PART 34: portfolio-level CAGR / MDD per exit on the same frozen long entries (10-slot, path dependent)
    from pipeline import phase2_portfolio as PF
    pm = []
    xl_ = xr[xr["side"] == "LONG"]
    tb_def = tr[(tr["variant"] == "TEXTBOOK") & (tr["side"] == "LONG")].assign(exit="TEXTBOOK(engine default)")
    for exn, g in list(xl_.groupby("exit")) + [("TEXTBOOK(engine default)", tb_def)]:
        for w in ("DISCOVERY", "STRICT_OOS"):
            a_, b_ = WINDOWS[w]
            res_ = PF.portfolio(p, g, a_, b_)
            pf_ = PF.perf(res_["equity"])
            pm.append({"engine": "W_LONG_POOLED", "side": "LONG", "exit": exn, "window": w,
                       "portfolio_cagr": pf_["cagr"], "portfolio_mdd": pf_["mdd"], "portfolio_sharpe": pf_["sharpe"]})
    pm = pd.DataFrame(pm)
    xt = xt.merge(pm, on=["engine", "side", "exit", "window"], how="left")
    save_csv(xt, "WEINSTEIN_EXIT_RESEARCH.csv")
    # Bollinger: freeze the B1 variant with the best DISCOVERY PnL/100 slot-days on pooled W long textbook entries
    disc = xt_pool[(xt_pool["window"] == "DISCOVERY")]
    b1 = disc[disc["exit"].str.startswith("BB_B1_")]
    frz = OUT / "frozen" / "BOLLINGER_EXIT_V1.json"
    if frz.exists():
        bb_name = json.loads(frz.read_text(encoding="utf-8"))["exit"]
    else:
        bb_name = b1.sort_values("pnl_per_100_slot_days", ascending=False)["exit"].iloc[0] if len(b1) else "BB_B1_20_2_FULL"
        frz.write_text(json.dumps({"name": "BOLLINGER_EXIT_V1", "exit": bb_name, "frozen_utc": now_utc(),
                                   "rule": "best DISCOVERY PnL/100 slot-days among B1 variants on pooled W1+W2+W3 "
                                           "TEXTBOOK entries"}, ensure_ascii=False, indent=1), encoding="utf-8")
    o = xt_pool[xt_pool["window"] == "STRICT_OOS"].set_index("exit")
    # textbook exit on pooled entries = engine-specific textbook exits
    tb_pool = stats_table(tr[(tr["variant"] == "TEXTBOOK") & (tr["side"] == "LONG")].assign(engine="P", exit="TB"),
                          ["engine", "exit"])
    tbo = tb_pool[tb_pool["window"] == "STRICT_OOS"].iloc[0]
    bbo = o.loc[bb_name] if bb_name in o.index else pd.Series(dtype=float)
    bbv = bb_verdict({"eff": bbo.get("pnl_per_100_slot_days"), "eff_text": tbo.get("pnl_per_100_slot_days"),
                      "pf": bbo.get("pf"), "pf_text": tbo.get("pf"),
                      "n30": bbo.get("ge30", np.nan) * bbo.get("n", np.nan),
                      "n30_text": tbo.get("ge30", np.nan) * tbo.get("n", np.nan)})
    bb_tab = xt_pool[xt_pool["exit"].str.startswith("BB_") | (xt_pool["exit"].isin(["STOP_STAGE4_ONLY", "MODERN"]))].copy()
    bb_tab = pd.concat([bb_tab, tb_pool.assign(exit="TEXTBOOK(engine default)", engine="W_LONG_POOLED")], ignore_index=True)
    bb_tab = bb_tab.merge(pm, on=["engine", "exit", "window"], how="left", suffixes=("", "_pm"))
    bb_tab["frozen_choice"] = bb_tab["exit"] == bb_name
    save_csv(bb_tab, "BOLLINGER_EXIT_RESULTS.csv")
    out["bb"] = (bb_name, bbv, bb_tab)
    # right-tail analysis: how much of big moves each exit keeps (same entries)
    rt = []
    xl = xr[xr["side"] == "LONG"]
    for (ex_, w), g in xl.groupby(["exit", "window"]):
        for thr in (0.2, 0.3, 0.5):
            big = g[g["mfe"] >= thr]
            if len(big) == 0:
                continue
            rt.append({"exit": ex_, "window": w, "mfe_threshold": thr, "n_trades": len(g), "n_big_mfe": len(big),
                       "kept_ratio_mean": float((big["ret"] / big["mfe"]).mean()),
                       "kept_ratio_median": float((big["ret"] / big["mfe"]).median()),
                       "giveback_mean": float((big["mfe"] - big["ret"]).mean()),
                       "n_ret_ge30": int((g["ret"] >= 0.3).sum()), "n_ret_ge50": int((g["ret"] >= 0.5).sum()),
                       "median_hold_big": float(big["holding_days"].median())})
    save_csv(pd.DataFrame(rt), "RIGHT_TAIL_EXIT_ANALYSIS.csv")
    return out


def dash_files(p, ctx, allset: pd.DataFrame, tr: pd.DataFrame, extra_trades: pd.DataFrame | None = None):
    """Dashboard inputs (local files only)."""
    S = ctx.S
    wk = ctx.wd.week_end
    long = []
    for k, fr in (("close", ctx.wd.close), ("high", ctx.wd.high), ("low", ctx.wd.low), ("vol", ctx.wd.vol),
                  ("ma30w", S["ma30w"]), ("ma10w", S["ma10w"]), ("stage", S["stage"]), ("mrs", ctx.mrs)):
        x = fr.astype("float32").stack(future_stack=True)
        x.name = k
        long.append(x)
    W = pd.concat(long, axis=1).reset_index()
    W.columns = ["date", "stock_id"] + list(W.columns[2:])
    W = W[W["close"].notna()]
    W.to_parquet(P2 / "dash_weekly.parquet", index=False)
    mk = pd.DataFrame({"date": wk, "taiex": ctx.wd.mkt.to_numpy(), "mkt_stage": ctx.mkt["stage"].iloc[:, 0].to_numpy(),
                       "breadth12": ctx.breadth12.to_numpy()})
    mk.to_csv(P2 / "dash_market_stage.csv", index=False)
    # current setups (last week-end)
    t_last = int(p.dates.get_loc(wk[-1]))
    cur = allset[(allset["t"] == t_last) & allset["engine"].isin(["W1", "W2", "S1"])].copy()
    if len(cur):
        mod = modern_rows(allset)
        cur["variant_ok"] = np.where(cur["textbook"], "TEXTBOOK", "") + np.where(cur.index.isin(mod.index), "+MODERN", "")
        cur = cur[cur["variant_ok"] != ""]
        f = p.raw_c.iloc[t_last] / p.c.iloc[t_last]
        fac = cur["stock_id"].map(f)
        for c in ("trigger", "limit", "stop"):
            cur[f"{c}_raw"] = cur[c] * fac
        cur["stop_pct"] = (cur["trigger"] - cur["stop"]).abs() / cur["trigger"]
        cur["name"] = cur["stock_id"].map(p.names)
        cur["sector"] = cur["stock_id"].map(p.sector)
    cur.to_csv(P2 / "dash_weinstein_current.csv", index=False)
    # examples: >= 10 successes and 10 failures per strategy (TEXTBOOK + MODERNIZED + extras)
    ex = []
    base = tr[tr["variant"].isin(["TEXTBOOK", "MODERNIZED"])]
    if extra_trades is not None and len(extra_trades):
        base = pd.concat([base, extra_trades], ignore_index=True)
    for (eng, var), g in base.groupby(["engine", "variant"]):
        g = g[g["entry_date"] >= "2023-01-01"]
        if len(g) == 0:
            continue
        ex.append(g.nlargest(min(10, len(g)), "ret").assign(example="SUCCESS"))
        ex.append(g.nsmallest(min(10, len(g)), "ret").assign(example="FAILURE"))
    E_ = pd.concat(ex, ignore_index=True) if ex else pd.DataFrame()
    if len(E_):
        E_["name"] = E_["stock_id"].map(p.names)
        E_["sector"] = E_["stock_id"].map(p.sector)
        cols = ["example", "engine", "variant", "exit", "stock_id", "name", "sector", "entry_date", "entry_price",
                "exit_date", "exit_reason", "holding_days", "ret", "r_multiple", "mfe", "mae", "stop_dist_pct",
                "stop_dist_atr", "window"]
        E_ = E_[[c for c in cols if c in E_]]
        E_.to_csv(P2 / "dash_examples.csv", index=False)
        save_csv(E_, "TRADE_EXAMPLES_PHASE2.csv")
    return E_


WIN3 = {k: WINDOWS[k] for k in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS")}
SURFACES = {
    "PRE_TRADE_RR": [("stop_dist_atr", "res_dist_capped"), ("stop_dist_pct", "room_h250")],
    "OVERHEAD": [("supply15", "rs120_pct"), ("res_n15", "stop_dist_atr")],
    "RS_LEADS": [("rs_leads120_10d", "stage_w")],
    "RESILIENCE": [("dcap60", "ucap60")],
    "SECTOR": [("rs120_pct", "secmkt_20")],
    "COMPRESSION": [("atr5_20", "brk_vol"), ("bbw_pct120", "vol5_20")],
    "VOLUME": [("vol10_60", "brk_vol")],
    "BASE": [("base_weeks", "base_depth")],
    "EXTENSION": [("stage2_age", "ext_ma60_atr")],
}
FAMILY_FILE = {
    "OVERHEAD": "OVERHEAD_RESISTANCE_ANALYSIS.csv", "RS_LEADS": "RS_LEADS_PRICE_ANALYSIS.csv",
    "RESILIENCE": "DOWNSIDE_RESILIENCE_ANALYSIS.csv", "SECTOR": "SECTOR_LEADERSHIP_ANALYSIS.csv",
    "BASE": "BASE_QUALITY_ANALYSIS.csv", "COMPRESSION": "COMPRESSION_EXPANSION_ANALYSIS.csv",
    "VOLUME": "VOLUME_PATTERN_ANALYSIS.csv", "EXTENSION": "TREND_EXTENSION_ANALYSIS.csv",
}


def surface(df: pd.DataFrame, fx: str, fy: str, family: str) -> pd.DataFrame:
    """3x3 tercile response surface (DISCOVERY breakpoints) of the stop-aware outcome."""
    disc = (df["date"] >= WINDOWS["DISCOVERY"][0]) & (df["date"] <= WINDOWS["DISCOVERY"][1])
    def cut(f):
        x = df[f].astype(float)
        if x[disc].nunique() <= 3:
            return x.rank(method="dense").clip(upper=3) - 1
        e = np.nanquantile(x[disc], [1 / 3, 2 / 3])
        return pd.Series(np.digitize(x, e), index=df.index).where(np.isfinite(x))
    bx, by = cut(fx), cut(fy)
    rows = []
    for w, (a, b) in WIN3.items():
        m = (df["date"] >= a) & (df["date"] <= b) & bx.notna() & by.notna()
        for (i, k), g in df[m].groupby([bx[m], by[m]]):
            rows.append({"analysis": "SURFACE", "family": family, "x": fx, "y": fy, "window": w, "x_tercile": int(i),
                         "y_tercile": int(k), "n": len(g), "ret_s40": g["ret_s40"].mean(), "r_s40": g["r_s40"].mean(),
                         "ge20": g["ge20"].mean(), "stop_hit": g["stop_hit"].mean(),
                         "up20_before_stop": g["up20_before_stop"].mean()})
    return pd.DataFrame(rows)


def flag_table(df: pd.DataFrame, flag: pd.Series, name: str, family: str) -> pd.DataFrame:
    rows = []
    for w, (a, b) in WIN3.items():
        m = (df["date"] >= a) & (df["date"] <= b)
        for v, g in df[m].groupby(flag[m]):
            pos, neg = g["ret_s40"][g["ret_s40"] > 0], g["ret_s40"][g["ret_s40"] < 0]
            rows.append({"analysis": "GROUP", "family": family, "group_var": name, "group": v, "window": w, "n": len(g),
                         "ret_s40": g["ret_s40"].mean(), "r_s40": g["r_s40"].mean(), "ge20": g["ge20"].mean(),
                         "stop_hit": g["stop_hit"].mean(), "mfe40": g["mfe40"].mean(), "mae40": g["mae40"].mean(),
                         "up20_before_stop": g["up20_before_stop"].mean(),
                         "payoff": pos.mean() / -neg.mean() if len(pos) and len(neg) else np.nan})
    return pd.DataFrame(rows)


def highrr(p, F, G, ctx, wtr: pd.DataFrame | None = None, mom: pd.DataFrame | None = None):
    from alpha import highrr as HR
    from alpha.trades import trade_stats
    from alpha.verdicts2 import highrr_verdict
    ev = HR.event_panel(p, F, G, ctx, log=log)
    ev.to_pickle(P2 / "highrr_panel.pkl")
    tabs, summ = [], []
    for fam, feats in HR.FAMILIES.items():
        for f in feats:
            if f not in ev:
                continue
            t = HR.decile_table(ev, f, WIN3)
            if t.empty:
                continue
            t.insert(0, "family", fam)
            tabs.append(t)
            r = {"family": fam, "feature": f}
            for w in WIN3:
                r[f"mono_{w}"] = HR.monotonicity(t, w)
                tw = t[t["window"] == w].sort_values("bin")
                r[f"spread_{w}"] = float(tw["ret_s40"].iloc[-1] - tw["ret_s40"].iloc[0]) if len(tw) >= 2 else np.nan
                r[f"ge20_spread_{w}"] = float(tw["ge20"].iloc[-1] - tw["ge20"].iloc[0]) if len(tw) >= 2 else np.nan
            disc = (ev["date"] >= WINDOWS["DISCOVERY"][0]) & (ev["date"] <= WINDOWS["DISCOVERY"][1])
            r["ic_DISCOVERY"] = HR.spearman_ic(ev[disc], f)
            summ.append(r)
            log(f"  decile {fam}/{f}: mono D={r['mono_DISCOVERY']:.2f} PRE={r['mono_PRE_2020_2022']:.2f}")
    deciles = pd.concat(tabs, ignore_index=True)
    summary = pd.DataFrame(summ)
    deciles.to_csv(P2 / "highrr_deciles_all.csv", index=False)
    summary.to_csv(P2 / "highrr_feature_summary.csv", index=False)
    # ---- family CSVs: deciles + response surfaces + group tables ----
    ev["leader_type"] = np.select(
        [(ev["rs120_pct"] >= 0.8) & (ev["secmkt_20"] > 0), (ev["rs120_pct"] >= 0.8) & (ev["secmkt_20"] <= 0)],
        ["SECTOR_CONFIRMED", "INDEPENDENT"], "NOT_LEADER")
    ev["rs_leads_flag"] = np.where(ev["rs_leads120_10d"] > 0, "RS_LEADS_PRICE",
                                   np.where(ev["rs_newhigh120"] > 0, "RS_AND_PRICE_HIGH", "NO_RS_HIGH"))
    ev["comp_exp"] = np.select([(ev["atr5_20"] < 0.8) & (ev["brk_vol"] >= 2), (ev["atr5_20"] < 0.8)],
                               ["COMPRESSION_THEN_VOL_EXPANSION", "COMPRESSION_ONLY"], "NO_COMPRESSION")
    ev["vol_pattern"] = np.select([(ev["vol10_60"] < 0.8) & (ev["brk_vol"] >= 2), ev["brk_vol"] >= 2,
                                   ev["vol10_60"] < 0.8], ["DRYUP_THEN_EXPANSION", "EXPANSION_ONLY", "DRYUP_ONLY"],
                                  "NEITHER")
    ev["resilience_type"] = np.select([(ev["dcap60"] < 0.8) & (ev["ucap60"] > 1.2), ev["dcap60"] < 0.8,
                                       ev["ucap60"] > 1.2], ["ASYMMETRIC_STRENGTH", "DOWNSIDE_RESILIENT_ONLY",
                                                             "UPSIDE_PARTICIPATION_ONLY"], "NEITHER")
    groups = {"SECTOR": ("leader_type",), "RS_LEADS": ("rs_leads_flag",), "COMPRESSION": ("comp_exp",),
              "VOLUME": ("vol_pattern",), "RESILIENCE": ("resilience_type",),
              "EXTENSION": ("stage_w",), "BASE": ("stage_w",), "OVERHEAD": ("blue_sky",)}
    for fam, fn in FAMILY_FILE.items():
        parts = [deciles[deciles["family"].isin([fam] + (["UPSIDE"] if fam == "OVERHEAD" else [])
                                                 + (["RS"] if fam == "RS_LEADS" else []))].assign(analysis="DECILE")]
        for fx, fy in SURFACES.get(fam, []):
            parts.append(surface(ev, fx, fy, fam))
        for gv in groups.get(fam, ()):
            parts.append(flag_table(ev, ev[gv].astype(str), gv, fam))
        save_csv(pd.concat(parts, ignore_index=True), fn)
    rr_parts = [deciles[deciles["family"].isin(["PRE_TRADE_RR", "RISK", "UPSIDE"])].assign(analysis="DECILE")]
    for fx, fy in SURFACES["PRE_TRADE_RR"]:
        rr_parts.append(surface(ev, fx, fy, "PRE_TRADE_RR"))
    pre_rr = pd.concat(rr_parts, ignore_index=True)
    pre_rr.to_csv(P2 / "PRE_TRADE_RR_ANALYSIS.csv", index=False)
    # ---- frozen HIGH_RR score ----
    frz = OUT / "frozen" / "HIGH_RR_SCORE_V1.json"
    if frz.exists():
        spec = json.loads(frz.read_text(encoding="utf-8"))["features"]
        log(f"HIGH_RR_SCORE_V1 loaded (frozen) {list(spec)}")
    else:
        cand = HR.select_score(summary)
        ranked = sorted(cand.items(), key=lambda kv: -abs(summary.set_index("feature").at[kv[1]["feature"],
                                                                                        "spread_DISCOVERY"]))
        spec = dict(ranked[:5])
        frz.write_text(json.dumps({"name": "HIGH_RR_SCORE_V1", "frozen_utc": now_utc(), "features": spec,
                                   "rule": "<=1 feature per family; |Spearman of DISCOVERY decile means of ret_s40| >= 0.6;"
                                           " sign agrees with PRE 2020-22; top-5 families by |D10-D1 spread| (DISCOVERY);"
                                           " score = mean of signed weekly cross-sectional percentiles",
                                   "selection_windows": ["DISCOVERY", "PRE_2020_2022"]}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        log(f"HIGH_RR_SCORE_V1 FROZEN: {spec}")
    ev["hrr_score"] = HR.score_frame(ev, spec)
    disc = (ev["date"] >= WINDOWS["DISCOVERY"][0]) & (ev["date"] <= WINDOWS["DISCOVERY"][1])
    t_hi = float(np.nanquantile(ev.loc[disc, "hrr_score"], 2 / 3))
    ev["hrr_top"] = ev["hrr_score"] >= t_hi
    sc_tab = HR.decile_table(ev, "hrr_score", WIN3)
    sc_tab.insert(0, "family", "HIGH_RR_SCORE")
    # ---- HIGH_RR_STAGE2_SETUP hypothesis (event panel) ----
    h2 = (ev["stage_w"] == 2) & (ev["stage2_age"] <= 130) & ev["hrr_top"] & (ev["stop_dist_atr"] <= 2.5)
    st2 = ev["stage_w"] == 2
    h2_tab = pd.concat([flag_table(ev[st2], h2[st2].map({True: "HIGH_RR_STAGE2_SETUP", False: "OTHER_STAGE2"}),
                                   "HIGH_RR_STAGE2_SETUP", "STAGE2"),
                        flag_table(ev, ev["hrr_top"].map({True: "HRR_TOP_TERCILE", False: "REST"}), "hrr_top", "ALL")],
                       ignore_index=True)
    # ---- strategy-conditional filter tests ----
    strat_rows, scored = [], []
    if wtr is not None and len(wtr):
        base = wtr[(wtr["variant"] == "TEXTBOOK") & (wtr["side"] == "LONG")].copy()
        if mom is not None and len(mom):
            base = pd.concat([base, mom], ignore_index=True)
        base["hrr_score"] = HR.score_at(p, F, G, ctx, spec, base[["t", "j"]], ev)
        for eng, g in base.groupby("engine"):
            gd = g[(g["entry_date"] >= WINDOWS["DISCOVERY"][0]) & (g["entry_date"] <= WINDOWS["DISCOVERY"][1])]
            thr = float(np.nanquantile(gd["hrr_score"], 2 / 3)) if gd["hrr_score"].notna().sum() >= 15 else t_hi
            g = g.assign(hrr_top=g["hrr_score"] >= thr, hrr_thr=thr)
            scored.append(g)
            for w, (a, b) in WINDOWS.items():
                sub = g[(g["entry_date"] >= a) & (g["entry_date"] <= b)]
                for lab, ss in (("ALL", sub), ("HRR_TOP_TERCILE", sub[sub["hrr_top"]]),
                                ("HRR_BOTTOM_2_TERCILES", sub[~sub["hrr_top"]])):
                    strat_rows.append({"engine": eng, "window": w, "subset": lab, "threshold": thr, **trade_stats(ss)})
    strat = pd.DataFrame(strat_rows)
    save_csv(pd.concat([strat.assign(table="STRATEGY_FILTER"), h2_tab.assign(table="EVENT_PANEL"),
                        sc_tab.assign(table="SCORE_DECILES")], ignore_index=True), "HIGH_RR_WEINSTEIN_RESULTS.csv")
    # verdict (pooled W1+W2+W3 textbook, STRICT_OOS)
    verdict = ("REJECT", "no strategy trades")
    if len(strat):
        sc_all = pd.concat(scored, ignore_index=True)
        wl = sc_all[sc_all["engine"].isin(["W1", "W2", "W3"])]
        o = wl[(wl["entry_date"] >= WINDOWS["STRICT_OOS"][0]) & (wl["entry_date"] <= WINDOWS["STRICT_OOS"][1])]
        a_, f_ = trade_stats(o), trade_stats(o[o["hrr_top"]])
        verdict = highrr_verdict({"ev_f": f_.get("ev"), "ev_all": a_.get("ev"), "pay_f": f_.get("payoff"),
                                  "pay_all": a_.get("payoff"), "ge20_f": f_.get("ge20"), "ge20_all": a_.get("ge20"),
                                  "pf_f": f_.get("pf")})
        sc_all.to_pickle(P2 / "strategy_trades_scored.pkl")
    # ---- HIGH_RR_STAGE2 as a tradeable engine (structural stop, MODERN exit) ----
    h2_res = highrr_stage2_trades(p, F, G, ctx, ev[h2])
    # matched control: same weeks, same count, random OTHER stage-2 stock-weeks, identical stop / exits
    rng = np.random.default_rng(5)
    oth = ev[st2 & ~h2]
    cnt = ev[h2].groupby("date").size()
    parts = []
    for d, k in cnt.items():
        pool = oth[oth["date"] == d]
        if len(pool):
            parts.append(pool.iloc[rng.choice(len(pool), size=min(k, len(pool)), replace=False)])
    ctrl = pd.concat(parts) if parts else oth.iloc[:0]
    c_res = highrr_stage2_trades(p, F, G, ctx, ctrl, label="STAGE2_RANDOM_CONTROL")
    h2_all = pd.concat([h2_res, c_res], ignore_index=True)
    # bootstrap p (HIGH_RR_STAGE2 EV > control EV) per window / exit
    pv = []
    for exn in ("MODERN_NOVOL", "TEXTBOOK_NOVOL"):
        a_ = pd.read_pickle(P2 / f"highrr_stage2_trades_{exn}.pkl")
        b_ = pd.read_pickle(P2 / f"highrr_stage2_trades_STAGE2_RANDOM_CONTROL_{exn}.pkl")
        for w, (s0, s1) in WIN3.items():
            x = a_[(a_["entry_date"] >= s0) & (a_["entry_date"] <= s1)]["ret"]
            y = b_[(b_["entry_date"] >= s0) & (b_["entry_date"] <= s1)]["ret"]
            pv.append({"engine": "HIGH_RR_STAGE2_SCORE_V1_vs_CONTROL", "exit": exn, "window": w, "n": len(x),
                       "ev": x.mean(), "ev_control": y.mean(), "placebo_p": bootstrap_p(x.to_numpy(), y.to_numpy())})
    h2_all = pd.concat([h2_all, pd.DataFrame(pv)], ignore_index=True)
    h2_all.insert(0, "definition", "SCORE_V1 (registered §L; NOT the user-spec 7-condition setup)")
    h2_all.to_csv(P2 / "highrr_stage2_score_v1.csv", index=False)
    save_csv(h2_all, "HIGH_RR_STAGE2_RESULTS.csv")
    # ---- failed high R/R setups with controls ----
    top = ev[ev["hrr_top"]]
    failed = top[(top["stop_hit"] > 0) & (top["ret_s40"] < 0)]
    winners = top[top["up20_before_stop"] > 0]
    feats = [v["feature"] for v in spec.values()] + ["stop_dist_atr", "rr_res", "res_dist_capped", "supply15",
                                                      "rs120_pct", "secmkt_20", "atr5_20", "brk_vol", "ext_ma60_atr",
                                                      "stage2_age", "dcap60"]
    feats = list(dict.fromkeys(f for f in feats if f in ev))
    comp = []
    for w, (a, b) in WIN3.items():
        fm = failed[(failed["date"] >= a) & (failed["date"] <= b)]
        wm = winners[(winners["date"] >= a) & (winners["date"] <= b)]
        for f in feats:
            comp.append({"row_type": "FEATURE_CONTRAST", "window": w, "feature": f, "failed_mean": fm[f].mean(),
                         "winner_control_mean": wm[f].mean(), "failed_n": len(fm), "winner_n": len(wm)})
    mkt = p.market["adj_close"]
    mret40 = (mkt.shift(-40) / mkt.shift(-1) - 1).reindex(failed["date"]).to_numpy()
    fl = failed.assign(row_type="FAILED_SETUP", mkt_fwd40=mret40, name=failed["stock_id"].map(p.names),
                       sector=failed["stock_id"].map(p.sector))
    fl["failure_type"] = np.select([fl["mkt_fwd40"] < -0.05, fl["secmkt_20"] < 0, fl["ext_ma60_atr"] > 4,
                                    fl["stop_dist_atr"] < 1.0],
                                   ["MARKET_DOWN", "SECTOR_WEAK", "TOO_EXTENDED", "STOP_TOO_TIGHT"], "STOCK_SPECIFIC")
    keep = ["row_type", "date", "stock_id", "name", "sector", "hrr_score", "failure_type", "ret_s40", "mae40",
            "mfe40", "mkt_fwd40"] + feats
    save_csv(pd.concat([pd.DataFrame(comp), fl[keep].sort_values("date").tail(3000)], ignore_index=True),
             "FAILED_HIGH_RR_SETUPS.csv")
    # ---- current radar candidates ----
    radar = radar_candidates(p, F, G, ctx, ev, spec)
    save_csv(radar, "HIGH_RR_CANDIDATES.csv")
    ev.to_pickle(P2 / "highrr_panel.pkl")
    return {"summary": summary, "spec": spec, "verdict": verdict, "strat": strat, "ev": ev, "h2": h2_res,
            "pre_rr": pre_rr}


def highrr_stage2_trades(p, F, G, ctx, sig: pd.DataFrame, label: str = "HIGH_RR_STAGE2_SCORE_V1") -> pd.DataFrame:
    from alpha.trades import attach_weekly, build_mats, simulate, trade_stats
    M = attach_weekly(build_mats(p, F, G), p, ctx)
    E = exit_specs()
    end_idx = int(p.dates.searchsorted(pd.Timestamp(END), side="right") - 1)
    swl = p.l.rolling(20, min_periods=15).min().to_numpy()
    rows = []
    for exn in ("MODERN_NOVOL", "TEXTBOOK_NOVOL"):
        out = []
        for r in sig.itertuples():
            stop = swl[r.t, r.j] * 0.99
            res = simulate(M, int(r.j), int(r.t), 1, "open", np.nan, stop, E[exn], end_idx, COST, SLIP, order_days=1)
            if res:
                res.update({"engine": label, "variant": "V1", "exit": exn, "j": int(r.j), "t": int(r.t)})
                out.append(res)
        df = pd.DataFrame(out)
        if df.empty:
            continue
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df = no_overlap(df)
        df = tag_windows(df)
        df.to_pickle(P2 / (f"highrr_stage2_trades_{exn}.pkl" if label == "HIGH_RR_STAGE2_SCORE_V1"
                           else f"highrr_stage2_trades_{label}_{exn}.pkl"))
        rows.append(stats_table(df, ["engine", "exit"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def radar_candidates(p, F, G, ctx, ev, spec) -> pd.DataFrame:
    """Latest week-end High R/R Radar with the PART 47 fields (one row per stage-1/2 universe stock)."""
    from alpha import highrr as HR
    last = ev["date"].max()
    i = len(ctx.wd.week_end) - 1
    t = int(p.dates.get_loc(ctx.wd.week_end[-1]))
    st = ctx.S["stage"].to_numpy()[i]
    U = p.universe.to_numpy()[t]
    js = np.nonzero(U & np.isin(st, [1, 2]))[0]
    rows = pd.DataFrame({"t": t, "j": js})
    cnt = ev.groupby("date").size()
    ref_date = cnt[cnt >= 300].index.max()          # latest week with a full cross-section (40-day outcomes known)
    rows["hrr_score"] = HR.score_at(p, F, G, ctx, spec, rows, ev[ev["date"] == ref_date])
    c, raw = p.c.iloc[t], p.raw_c.iloc[t]
    swl = p.l.rolling(20, min_periods=15).min().iloc[t]
    atr = F["atr"].iloc[t]
    mk = p.market["adj_close"]
    def exn(n):
        return (p.c.iloc[t] / p.c.iloc[t - n] - 1) - (mk.iloc[t] / mk.iloc[t - n] - 1)
    rs = {n: exn(n) for n in (20, 40, 60, 120)}
    rs120_pct = rs[120][p.universe.iloc[t]].rank(pct=True)
    mkt_stage = int(ctx.mkt["stage"].iloc[i, 0])
    setups = pd.read_pickle(P2 / "weinstein_setups.pkl") if (P2 / "weinstein_setups.pkl").exists() else pd.DataFrame()
    cur = setups[(setups["t"] == t)] if len(setups) else setups
    mod_idx = set(modern_rows(setups).index) if len(setups) else set()
    strat = {}
    for r in cur.itertuples():
        tag = r.engine + ("(TEXTBOOK)" if r.textbook else "") + ("(MODERN)" if r.Index in mod_idx else "")
        if r.textbook or r.Index in mod_idx:
            strat.setdefault(r.stock_id, []).append(tag)
    try:
        sig = pd.read_parquet(OUT / "signals" / "states_daily.parquet")
        sig = sig[sig["date"] == sig["date"].max()]
        for r in sig.itertuples():
            if str(getattr(r, "state", "")) in ("PROBE_READY", "PROBED", "WAIT_CONFIRM", "CONFIRMED", "ADD_READY", "FULL"):
                strat.setdefault(str(r.stock_id), []).append(f"MOMENTUM({r.state})")
    except Exception:   # noqa: BLE001
        pass
    gst = ctx.grp_stage_w.to_numpy()[i]
    hi52 = ctx.wd.high.rolling(52, min_periods=1).max().to_numpy()[i]
    out = []
    for r in rows.itertuples():
        j = r.j
        sid = p.ids[j]
        f = raw[sid] / c[sid] if c[sid] > 0 else np.nan
        stop = swl[sid] * 0.99
        sd = 1 - stop / c[sid]
        res = G["res_dist"].iat[t, j]
        room = res if np.isfinite(res) else np.nan
        out.append({
            "date": p.dates[t].date(), "symbol": sid, "name": p.names.get(sid, ""),
            "strategy": ", ".join(strat.get(sid, [])) or "STAGE_WATCH",
            "stage": int(st[j]), "stage_age_weeks": int(ctx.S["ep_len"].to_numpy()[i, j]),
            "market_regime": f"TAIEX Stage {mkt_stage}",
            "sector_regime": f"{p.sector.get(sid, '')} Stage {int(gst[j])}" if np.isfinite(gst[j]) else f"{p.sector.get(sid, '')} (n/a)",
            "rs20": rs[20][sid], "rs40": rs[40][sid], "rs60": rs[60][sid], "rs120_pct": rs120_pct.get(sid, np.nan),
            "rs_acceleration": F["rs_accel"].iat[t, j],
            "rs_leads_price": bool((G["rs_leads60_10d"].iat[t, j] > 0) or (G["rs_leads120_10d"].iat[t, j] > 0)),
            "downside_resilience_dcap60": F["dcap60"].iat[t, j], "upside_participation_ucap60": F["ucap60"].iat[t, j],
            "base_duration_weeks": int(ctx.S["ep_len"].to_numpy()[i, j]) if st[j] == 1 else np.nan,
            "base_tightness_10w": ctx.wd.high.iloc[i - 9:i + 1, j].max() / ctx.wd.low.iloc[i - 9:i + 1, j].min() - 1,
            "atr_compression_5_20": G["atr5_20"].iat[t, j], "bb_width_pct120": G["bbw_pct120"].iat[t, j],
            "breakout_volume_ratio": G["brk_vol"].iat[t, j], "overhead_resistance_pivots15": G["res_n15"].iat[t, j],
            "overhead_supply_density15": G["supply15"].iat[t, j], "close": raw[sid],
            "structural_stop": stop * f, "stop_distance_pct": sd, "stop_distance_atr": (c[sid] - stop) / atr[sid],
            "room_to_resistance_pct": room, "blue_sky": bool(res == np.inf),
            "room_to_52w_high_pct": hi52[j] / ctx.wd.close.to_numpy()[i, j] - 1,
            "pre_trade_rr": (min(res, 1.0) if np.isfinite(res) else 1.0) / sd if sd > 0 else np.nan,
            "hrr_score": r.hrr_score,
            "one_second_trigger_status": "盤中 tick 才能判斷（本表為收盤後資料；歷史觸發見 Intraday Replay）",
        })
    df = pd.DataFrame(out)
    df["leader_type"] = np.select([(df["rs120_pct"] >= 0.8) & (F["secmkt_20"].iloc[t].reindex(df["symbol"]).to_numpy() > 0),
                                   df["rs120_pct"] >= 0.8], ["SECTOR_CONFIRMED", "INDEPENDENT"], "NOT_LEADER")
    return df.sort_values("hrr_score", ascending=False).reset_index(drop=True)


def portfolio_step(p, F, G, ctx):
    from alpha.trades import attach_weekly, build_mats, trade_stats
    from alpha.verdicts2 import hybrid_verdict, portfolio_verdict
    from pipeline import phase2_portfolio as PF
    tr = pd.read_pickle(P2 / "weinstein_trades.pkl")
    shorts = pd.read_pickle(P2 / "weinstein_shorts_exec.pkl")
    mom = tag_windows(PF.momentum_trades(p))
    M = attach_weekly(build_mats(p, F, G), p, ctx)
    E = exit_specs()
    tb = tr[(tr["variant"] == "TEXTBOOK") & (tr["side"] == "LONG")]
    # ---- overlap ----
    sx = shorts[(shorts["variant"] == "TEXTBOOK") & shorts["executable"]]
    ov = PF.overlap(p, pd.concat([mom, tb, sx], ignore_index=True))
    save_csv(ov, "ALPHA_OVERLAP_ANALYSIS.csv")
    # ---- hybrids ----
    from pipeline import phase2_userspec as US
    hy = tag_windows(PF.hybrids(p, F, G, ctx, tr, mom, M, E))
    hy["engine"] = hy["engine"].replace({"HYBRID_C": "HYBRID_C_ALT", "HYBRID_D": "HYBRID_D_ALT"})
    end_i = int(p.dates.searchsorted(pd.Timestamp(END), side="right") - 1)
    ev_b = pd.read_pickle(P2 / "userspec_stage1_breakouts.pkl")
    hc = US.hybrid_c_userspec(M, ev_b, end_i)
    hd = US.hybrid_d_userspec(p, ctx, M, mom, end_i)
    hy = pd.concat([hy, hc, hd], ignore_index=True)
    hy.to_pickle(P2 / "hybrid_trades.pkl")
    comps = {"MOMENTUM": mom, "WEINSTEIN_LONG_TEXTBOOK": tb.assign(engine="WEINSTEIN_LONG_TEXTBOOK")}
    rows, pf_res = [], {}
    oos = WINDOWS["STRICT_OOS"]
    for name, df in list(comps.items()) + [(k, g) for k, g in hy.groupby("engine")]:
        t = stats_table(df.assign(engine=name), ["engine"])
        res = PF.portfolio(p, df.assign(engine=name if name in PF.PRIORITY else df["engine"]), *oos)
        perf = PF.perf(res["equity"], p.market["adj_close"])
        pf_res[name] = perf
        t["portfolio_oos_cagr"] = perf["cagr"]
        t["portfolio_oos_calmar"] = perf["calmar"]
        t["portfolio_oos_mdd"] = perf["mdd"]
        rows.append(t)
    ht = pd.concat(rows, ignore_index=True)
    hv = {}
    for h in ("HYBRID_A", "HYBRID_B", "HYBRID_C", "HYBRID_D", "HYBRID_C_ALT", "HYBRID_D_ALT"):
        o = ht[(ht["window"] == "STRICT_OOS")].set_index("engine")
        if h not in o.index:
            hv[h] = ("REJECT", "no trades")
            continue
        hv[h] = hybrid_verdict({"pf": o.at[h, "pf"], "ev": o.at[h, "ev"], "calmar": pf_res[h]["calmar"],
                                "pf_comp": [o.at[c, "pf"] for c in comps], "ev_comp": [o.at[c, "ev"] for c in comps],
                                "calmar_comp": [pf_res[c]["calmar"] for c in comps]})
    ht["verdict"] = ht["engine"].map(lambda e: hv.get(e, ("", ""))[0])
    save_csv(ht, "HYBRID_STRATEGY_RESULTS.csv")
    # ---- multi-alpha portfolio ----
    sx_all = shorts[(shorts["variant"] == "TEXTBOOK")]
    book = {
        "MULTI_ALPHA_V1": pd.concat([mom, tb, sx], ignore_index=True),
        "MULTI_ALPHA_LONG_ONLY": pd.concat([mom, tb], ignore_index=True),
        "MOMENTUM_ONLY": mom, "WEINSTEIN_LONG_ONLY": tb,
        "MULTI_ALPHA_THEORETICAL_SHORTS": pd.concat([mom, tb, sx_all], ignore_index=True),
    }
    prow, curves = [], []
    for name, df in book.items():
        for w in ("DISCOVERY", "STRICT_OOS", "FULL_2023_2026"):
            a, b = WINDOWS[w]
            for lab, xc, xs in (("BASE", 0.0, 0.0), ("STRESS_0.70%+50bps", 0.0025, 0.0025)):
                res = PF.portfolio(p, df, a, b, extra_cost=xc, extra_slip=xs)
                pr = PF.perf(res["equity"], p.market["adj_close"])
                taken = res["taken"]
                prow.append({"portfolio": name, "window": w, "cost_case": lab, **pr, "n_trades": len(taken),
                             **{f"n_{e}": int((taken["engine"] == e).sum()) for e in ("MOMENTUM", "W1", "W2", "W3", "S1", "S2")}})
                if lab == "BASE" and w == "FULL_2023_2026":
                    curves.append(res["equity"]["equity"].rename(name))
    pt = pd.DataFrame(prow)
    eqc = pd.concat(curves, axis=1)
    eqc["TAIEX"] = p.market["adj_close"].reindex(eqc.index) / p.market["adj_close"].reindex(eqc.index).iloc[0] * config.INITIAL_CAPITAL
    eqc.to_csv(P2 / "multi_alpha_equity.csv")
    save_csv(pt, "MULTI_ALPHA_PORTFOLIO.csv")
    o = pt[(pt["portfolio"] == "MULTI_ALPHA_V1") & (pt["window"] == "STRICT_OOS")].set_index("cost_case")
    pv = portfolio_verdict({"cagr": o.at["BASE", "cagr"], "sharpe": o.at["BASE", "sharpe"], "mdd": o.at["BASE", "mdd"],
                            "cagr_stress": o.at["STRESS_0.70%+50bps", "cagr"], "ret_2025": o.at["BASE", "ret_2025"],
                            "ret_2026": o.at["BASE", "ret_2026"]})
    out = {"overlap": ov, "hybrids": ht, "hybrid_verdicts": hv, "portfolio": pt, "portfolio_verdict": pv,
           "momentum_trades": mom}
    with open(P2 / "portfolio_report.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return out


def intraday_step(p, F, G, ctx, download: bool = True, reuse: bool = False):
    from alpha.trades import attach_weekly, build_mats
    from pipeline import phase2_intraday as PI
    tr = pd.read_pickle(P2 / "weinstein_trades.pkl")
    setups = pd.read_pickle(P2 / "weinstein_setups.pkl")
    out = PI.run(p, F, G, ctx, tr, setups, download=download, reuse=reuse)
    frz = OUT / "frozen" / "BOLLINGER_EXIT_V1.json"
    b2 = pd.DataFrame()
    if frz.exists():
        b1 = json.loads(frz.read_text(encoding="utf-8"))["exit"]
        M = attach_weekly(build_mats(p, F, G), p, ctx)
        b2 = PI.bollinger_b2_ticks(p, tr, M, b1.replace("BB_B1_", "BB_B2_"), download=download)
        b2.to_csv(P2 / "bollinger_b2_tick_verification.csv", index=False)
    with open(P2 / "intraday_report.pkl", "wb") as fh:
        pickle.dump({"res": out["res"] if out else None, "comp": out["comp"] if out else None,
                     "imp": out["imp"] if out else None, "frozen": out["frozen"] if out else None, "b2": b2}, fh)
    return out, b2


def userspec_step(p, F, G, ctx):
    from alpha.trades import attach_weekly, build_mats
    from pipeline import phase2_userspec as US
    M = attach_weekly(build_mats(p, F, G), p, ctx)
    end_idx = int(p.dates.searchsorted(pd.Timestamp(END), side="right") - 1)
    ev, trades, res = US.hrr_stage2_userspec(p, F, G, ctx, M, end_idx)
    res.insert(0, "definition", "USER_SPEC 7-condition (PART 38; registered §A2)")
    score = pd.read_csv(P2 / "highrr_stage2_score_v1.csv") if (P2 / "highrr_stage2_score_v1.csv").exists() else pd.DataFrame()
    save_csv(pd.concat([res, score], ignore_index=True), "HIGH_RR_STAGE2_RESULTS.csv")
    # failed user-spec setups with controls (beta, ATR, sector, liquidity, market regime, stage age)
    tr = trades["TEXTBOOK_NOVOL"]
    hi = tr[tr["n_conditions"] >= 4].copy()
    if len(hi):
        t_ = p.dates.get_indexer(hi["entry_date"]) - 1
        jj = hi["j"].astype(int).to_numpy()
        hi["beta"] = F["beta"].to_numpy()[t_, jj]
        hi["atr_pct"] = F["atr_pct"].to_numpy()[t_, jj]
        hi["val20"] = F["val20"].to_numpy()[t_, jj]
        hi["sector"] = hi["stock_id"].map(p.sector)
        k = ctx.wd.week_end.searchsorted(p.dates[t_], side="right") - 1
        hi["mkt_stage"] = ctx.mkt["stage"].to_numpy()[k, 0]
        hi["grp_stage"] = ctx.grp_stage_w.to_numpy()[k, jj]
        hi["outcome"] = np.where(hi["ret"] < 0, "FAILED", "SUCCEEDED")
        ctrl_rows = []
        for w, (a, b) in {k_: WINDOWS[k_] for k_ in ("PRE_2020_2022", "DISCOVERY", "STRICT_OOS")}.items():
            x = hi[(hi["entry_date"] >= a) & (hi["entry_date"] <= b)]
            for f in ("beta", "atr_pct", "val20", "mkt_stage", "grp_stage", "stop_dist_pct", "n_conditions"):
                ctrl_rows.append({"row_type": "USERSPEC_CONTROL_CONTRAST", "window": w, "feature": f,
                                  "failed_mean": x[x["outcome"] == "FAILED"][f].mean(),
                                  "winner_control_mean": x[x["outcome"] == "SUCCEEDED"][f].mean(),
                                  "failed_n": int((x["outcome"] == "FAILED").sum()),
                                  "winner_n": int((x["outcome"] == "SUCCEEDED").sum())})
            for col in ("mkt_stage", "grp_stage"):
                for v, g in x.groupby(col):
                    ctrl_rows.append({"row_type": f"USERSPEC_FAIL_RATE_BY_{col.upper()}", "window": w, "feature": col,
                                      "group": v, "n": len(g), "fail_rate": float((g["outcome"] == "FAILED").mean()),
                                      "ev": float(g["ret"].mean())})
        hi["failure_reason"] = np.select(
            [hi["exit_reason"].str.contains("STOP", na=False) & (hi["mkt_stage"].isin([3, 4])),
             hi["exit_reason"].str.contains("STOP", na=False) & (hi["grp_stage"].isin([3, 4])),
             hi["exit_reason"].str.contains("STOP", na=False), hi["exit_reason"].str.contains("STAGE", na=False)],
            ["STOPPED_IN_WEAK_MARKET", "STOPPED_IN_WEAK_GROUP", "STOPPED_STOCK_SPECIFIC", "STAGE_EXIT_LOSS"], "OTHER")
        old_f = pd.read_csv(ROOT / "FAILED_HIGH_RR_SETUPS.csv") if (ROOT / "FAILED_HIGH_RR_SETUPS.csv").exists() else pd.DataFrame()
        if len(old_f) and "row_type" in old_f:
            old_f = old_f[~old_f["row_type"].astype(str).str.startswith("USERSPEC")]
        fl = hi[hi["outcome"] == "FAILED"].assign(row_type="USERSPEC_FAILED_SETUP")
        keep = ["row_type", "stock_id", "entry_date", "exit_date", "exit_reason", "failure_reason", "ret", "mae", "mfe",
                "n_conditions"] + US.COND + ["beta", "atr_pct", "val20", "sector", "mkt_stage", "grp_stage",
                                             "stop_dist_pct", "window"]
        save_csv(pd.concat([old_f, pd.DataFrame(ctrl_rows), fl[keep]], ignore_index=True), "FAILED_HIGH_RR_SETUPS.csv")
    return ev, trades, res


def short_exec_data_step(p):
    """Data-based executability layer for shorts (FinMind margin / short-sale tables)."""
    from alpha.trades import trade_stats
    from data.short_data import load
    sh = pd.read_pickle(P2 / "weinstein_shorts_exec.pkl")
    ids = sh["stock_id"].drop_duplicates().tolist()
    m = load("TaiwanStockMarginPurchaseShortSale", ids)
    d = load("TaiwanDailyShortSaleBalances", ids)
    m["date"] = pd.to_datetime(m["date"])
    d["date"] = pd.to_datetime(d["date"])
    m = m.drop_duplicates(["date", "stock_id"]).set_index(["date", "stock_id"])
    d = d.drop_duplicates(["date", "stock_id"]).set_index(["date", "stock_id"])
    rows = []
    for r in sh.itertuples():
        e = pd.Timestamp(r.entry_date)
        k = (e, r.stock_id)
        on_list = k in m.index
        note = str(m.at[k, "Note"]) if on_list else ""
        rows.append({"in_margin_table": on_list, "note": note.strip(),
                     "short_halt_X": "X" in note,
                     "short_limit": float(m.at[k, "ShortSaleLimit"]) if on_list else np.nan,
                     "margin_short_quota": float(d.at[k, "MarginShortSalesQuota"]) if k in d.index else np.nan,
                     "sbl_quota": float(d.at[k, "SBLShortSalesQuota"]) if k in d.index else np.nan})
    x = pd.concat([sh.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    x["executable_data"] = (x["in_margin_table"] & ~x["short_halt_X"] & (x["short_limit"] > 0)
                            & (x["plain_tick_ok"] | x["exempt_proxy"]))
    x.to_pickle(P2 / "weinstein_shorts_exec_data.pkl")
    out = []
    for lab, sub in (("EXECUTABLE_DATA", x[x["executable_data"]]),):
        t = stats_table(sub, ["engine", "variant", "exit"])
        t["executability"] = lab
        out.append(t)
        t2 = stats_table(sub.assign(engine="S_POOLED"), ["engine", "variant"])
        t2["executability"] = lab
        t2["exit"] = "ENGINE_DEFAULT"
        out.append(t2)
    aud = []
    for (eng, var, w), g in x.groupby(["engine", "variant", "window"]):
        aud.append({"engine": eng, "variant": var, "window": w, "executability": "AUDIT_DATA", "n": len(g),
                    "in_margin_table": g["in_margin_table"].mean(), "short_halt_X": g["short_halt_X"].mean(),
                    "short_limit_pos": (g["short_limit"] > 0).mean(), "plain_tick_ok": g["plain_tick_ok"].mean(),
                    "exempt_proxy": g["exempt_proxy"].mean(), "executable_data": g["executable_data"].mean()})
    old = pd.read_csv(ROOT / "WEINSTEIN_SHORT_RESULTS.csv")
    old = old[~old["executability"].isin(["EXECUTABLE_DATA", "AUDIT_DATA"])]
    save_csv(pd.concat([old] + out + [pd.DataFrame(aud)], ignore_index=True), "WEINSTEIN_SHORT_RESULTS.csv")
    return x


def dash_step(p, F, G, ctx):
    """Radar (PART 47) + examples (PART 49/50) with the requested columns."""
    spec = json.loads((OUT / "frozen" / "HIGH_RR_SCORE_V1.json").read_text(encoding="utf-8"))["features"]
    ev = pd.read_pickle(P2 / "highrr_panel.pkl")
    radar = radar_candidates(p, F, G, ctx, ev, spec)
    save_csv(radar, "HIGH_RR_CANDIDATES.csv")
    examples_step(p, F, G, ctx)
    return radar


def examples_step(p, F, G, ctx):
    """>= 10 successes and >= 10 failures per strategy with PART 49/50 columns."""
    from pipeline.phase2_portfolio import momentum_trades
    tr = pd.read_pickle(P2 / "weinstein_trades.pkl")
    setups = pd.read_pickle(P2 / "weinstein_setups.pkl")
    hy = pd.read_pickle(P2 / "hybrid_trades.pkl") if (P2 / "hybrid_trades.pkl").exists() else pd.DataFrame()
    mom = tag_windows(momentum_trades(p)).assign(variant="FROZEN_V1")
    pol = pd.read_csv(P2 / "intraday_policy_entries.csv", dtype={"stock_id": str}) if (P2 / "intraday_policy_entries.csv").exists() else pd.DataFrame()
    base = pd.concat([tr[tr["variant"].isin(["TEXTBOOK", "MODERNIZED"])], mom,
                      hy[hy["engine"].isin(["HYBRID_A", "HYBRID_B", "HYBRID_C", "HYBRID_D"])] if len(hy) else hy],
                     ignore_index=True)
    rows = []
    why = {"W1": "Stage 1 基底 ≥8 週、MA 不再下降、突破基底壓力（buy-stop）", "W2": "Stage 2 已確立、MA 明顯上升、靠近 MA 整理後再突破",
           "W3": "W1 放量突破後量縮回測突破點", "S1": "Stage 3 頂部跌破支撐（sell-stop）", "S2": "跌破後量縮反彈至跌破點失敗",
           "MOMENTUM": "凍結 V1：Emerging Leader 試單訊號", "HYBRID_A": "動能試單且週線 Stage 2",
           "HYBRID_B": "Weinstein 突破且動能分數 ≥0.9", "HYBRID_C": "Stage1→2 + RS 領先 + 低壓力 + 壓縮 + 放量",
           "HYBRID_D": "動能試單 → Weinstein 確認 → 加碼"}
    G_res = G["res_dist"].to_numpy()
    for (eng, var), g in base.groupby(["engine", "variant"]):
        g = g[g["entry_date"] >= "2023-01-01"]
        if len(g) == 0:
            continue
        for lab, sub in (("SUCCESS", g.nlargest(min(10, len(g)), "ret")), ("FAILURE", g.nsmallest(min(10, len(g)), "ret"))):
            for r in sub.itertuples():
                e = p.dates.get_loc(pd.Timestamp(r.entry_date))
                j = int(r.j) if hasattr(r, "j") and pd.notna(r.j) else p.ids.index(r.stock_id)
                t = int(r.t) if hasattr(r, "t") and pd.notna(getattr(r, "t", np.nan)) else e - 1
                k = ctx.wd.week_end.searchsorted(p.dates[t], side="right") - 1
                sd = getattr(r, "stop_dist_pct", np.nan)
                res = G_res[t, j]
                rr = ((min(res, 1.0) if np.isfinite(res) else 1.0) / sd) if sd and sd > 0 else np.nan
                ptime = ""
                if len(pol):
                    cid = f"{eng}_{r.stock_id}_{pd.Timestamp(r.entry_date).date()}" if eng != "MOMENTUM" else \
                        f"MOM_{r.stock_id}_{pd.Timestamp(r.entry_date).date()}"
                    pp = pol[(pol["cand_id"] == cid) & (pol["policy"] == "LEARNED")]
                    ptime = pp["entry_time"].iloc[0] if len(pp) else ""
                entry_time = ("09:00:00 開盤" if eng in ("W3", "S2", "MOMENTUM", "HYBRID_A", "HYBRID_C", "HYBRID_D")
                              else "盤中觸及 buy/sell-stop（日線模擬）")
                rows.append({
                    "example": lab, "symbol": r.stock_id, "name": p.names.get(r.stock_id, ""), "strategy": eng,
                    "variant": var, "candidate_date": p.dates[t].date(), "stage": int(ctx.S["stage"].to_numpy()[k, j]) if k >= 0 else np.nan,
                    "why_candidate": why.get(eng, ""),
                    "why_high_rr": f"PreTradeRR={rr:.2f}; 停損 {sd * 100:.1f}%; 上方{'無壓力' if res == np.inf else f'壓力 {res * 100:.1f}%'}" if np.isfinite(rr) else "",
                    "pre_trade_rr": rr, "one_second_trigger_time(LEARNED)": ptime, "entry_time": entry_time,
                    "entry_date": pd.Timestamp(r.entry_date).date(), "entry_price": getattr(r, "entry_price", np.nan),
                    "structural_stop": getattr(r, "stop_price", np.nan), "stop_distance_pct": sd,
                    "stop_distance_atr": getattr(r, "stop_dist_atr", np.nan),
                    "confirmation": getattr(r, "first_exit_reason", "") if eng != "HYBRID_D" else ("ADDED" if getattr(r, "added", False) else "NOT_CONFIRMED"),
                    "add": (str(pd.Timestamp(r.add_date).date()) if eng == "HYBRID_D" and pd.notna(getattr(r, "add_date", pd.NaT)) else ""),
                    "exit_date": pd.Timestamp(r.exit_date).date(), "exit_reason": r.exit_reason, "return": r.ret,
                    "mae": getattr(r, "mae", np.nan), "mfe": getattr(r, "mfe", np.nan),
                    "failure_reason": (r.exit_reason if lab == "FAILURE" else ""), "window": getattr(r, "window", "")})
    ex = pd.DataFrame(rows)
    ex.to_csv(P2 / "dash_examples.csv", index=False)
    save_csv(ex, "TRADE_EXAMPLES_PHASE2.csv")
    return ex


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["state", "momentum", "weinstein", "highrr", "userspec", "portfolio", "intraday",
                                     "shortdata", "dash", "all"])
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="intraday: reuse the cached 1-second dataset")
    a = ap.parse_args(argv)
    if a.step == "state":
        build_state()
        return
    p, F, G, ctx = load_state()
    if a.step in ("momentum", "all"):
        momentum(p, F)
    if a.step in ("weinstein", "all"):
        allset, tr, M, E, end_idx = weinstein(p, F, G, ctx)
        rep = weinstein_reports(p, F, G, ctx, allset, tr, M, E, end_idx)
        with open(P2 / "weinstein_report.pkl", "wb") as fh:
            pickle.dump({k: v for k, v in rep.items()}, fh)
        dash_files(p, ctx, allset, tr)
    if a.step in ("highrr", "all"):
        wtr = pd.read_pickle(P2 / "weinstein_trades.pkl") if (P2 / "weinstein_trades.pkl").exists() else None
        from pipeline.phase2_portfolio import momentum_trades
        hr = highrr(p, F, G, ctx, wtr, tag_windows(momentum_trades(p)))
        with open(P2 / "highrr_report.pkl", "wb") as fh:
            pickle.dump({k: v for k, v in hr.items() if k != "ev"}, fh)
    if a.step in ("dash", "all"):
        dash_step(p, F, G, ctx)
    if a.step in ("shortdata", "all"):
        short_exec_data_step(p)
    if a.step in ("userspec", "all"):
        userspec_step(p, F, G, ctx)
    if a.step in ("portfolio", "all"):
        portfolio_step(p, F, G, ctx)
    if a.step in ("intraday", "all"):
        intraday_step(p, F, G, ctx, download=not a.no_download, reuse=a.reuse)


if __name__ == "__main__":
    main()
