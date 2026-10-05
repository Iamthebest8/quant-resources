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
    from pipeline.run_research import load_frozen, run_tl
    from strategy.metrics import campaign_metrics
    from strategy.signals import SignalCache, build_arrays
    cfg, frozen = load_frozen("V1")
    A = build_arrays(p, F)
    sc = SignalCache(p, F)
    ref = pd.read_csv(OUT / "frozen" / "MOMENTUM_LONG_BASELINE_reference_metrics.csv")
    W = {"DISCOVERY": config.DISCOVERY, "EXTENDED": config.EXTENDED_VALIDATION, "STRICT_OOS": config.STRICT_OOS,
         "FULL": (config.RESEARCH_START, END)}
    rows, tls = [], {}
    for k, w in W.items():
        tl = run_tl(A, sc, cfg, *w)
        tls[k] = tl
        m = campaign_metrics(tl, years=(pd.Timestamp(w[1]) - pd.Timestamp(w[0])).days / 365.25)
        rows.append({"window": k, **{x: m.get(x) for x in ("n_probes", "pf", "payoff", "pnl_per_100_slot_days_pct",
                                                            "win_rate", "ev")}})
    rep = pd.DataFrame(rows)
    log("momentum reproduction:\n" + rep.to_string())
    return rep, tls, ref, cfg, frozen


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
        frames.append(simulate_setups(M, st, TEXTBOOK_EXIT[eng], E, "TEXTBOOK", end_idx))
        frames.append(simulate_setups(M, sm, MODERN_EXIT[eng], E, "MODERNIZED", end_idx))
        # exit research on the frozen TEXTBOOK entries
        others = ["STOP_STAGE4_ONLY"] + [k for k in E if k.startswith("BB_")]
        others += [MODERN_EXIT[eng]] if MODERN_EXIT[eng] != TEXTBOOK_EXIT[eng] else []
        for xn in others:
            frames.append(simulate_setups(M, st, xn, E, "TEXTBOOK_ENTRY_EXITRESEARCH", end_idx))
        log(f"  {eng} simulated ({time.time() - t0:.0f}s)")
    tr = pd.concat([f for f in frames if len(f)], ignore_index=True)
    tr = no_overlap(tr)
    tr = tag_windows(tr)
    tr.to_pickle(P2 / "weinstein_trades.pkl")
    log(f"weinstein trades: {len(tr)} in {time.time() - t0:.0f}s")
    return allset, tr, M, E, end_idx


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
    save_csv(h2_res, "HIGH_RR_STAGE2_RESULTS.csv")
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


def highrr_stage2_trades(p, F, G, ctx, sig: pd.DataFrame) -> pd.DataFrame:
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
                res.update({"engine": "HIGH_RR_STAGE2", "variant": "V1", "exit": exn})
                out.append(res)
        df = pd.DataFrame(out)
        if df.empty:
            continue
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df = no_overlap(df)
        df.to_pickle(P2 / f"highrr_stage2_trades_{exn}.pkl")
        rows.append(stats_table(df, ["engine", "exit"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def radar_candidates(p, F, G, ctx, ev, spec) -> pd.DataFrame:
    """Latest week-end: High R/R radar (dashboard Part 47 fields)."""
    from alpha import highrr as HR
    last = ev["date"].max()
    # the event panel only has rows with 40d forward data; rebuild the last week-end cross-section directly
    t = int(p.dates.searchsorted(ctx.wd.week_end[-1]))
    i = len(ctx.wd.week_end) - 1
    st = ctx.S["stage"].to_numpy()[i]
    U = p.universe.to_numpy()[t]
    js = np.nonzero(U & np.isin(st, [1, 2]))[0]
    rows = pd.DataFrame({"t": t, "j": js})
    rows["hrr_score"] = HR.score_at(p, F, G, ctx, spec, rows, ev[ev["date"] == last])
    c = p.c.iloc[t]
    raw = p.raw_c.iloc[t]
    swl = p.l.rolling(20, min_periods=15).min().iloc[t]
    atr = F["atr"].iloc[t]
    out = []
    for r in rows.itertuples():
        sid = p.ids[r.j]
        adj2raw = raw[sid] / c[sid] if c[sid] > 0 else np.nan
        stop = swl[sid] * 0.99
        sd = 1 - stop / c[sid]
        res = G["res_dist"].iat[t, r.j]
        out.append({"date": p.dates[t].date(), "stock_id": sid, "name": p.names.get(sid, ""),
                    "sector": p.sector.get(sid, ""), "stage": int(st[r.j]),
                    "stage_age_weeks": int(ctx.S["ep_len"].to_numpy()[i, r.j]),
                    "close": raw[sid], "hrr_score": r.hrr_score,
                    "structural_stop": stop * adj2raw, "stop_dist_pct": sd, "stop_dist_atr": (c[sid] - stop) / atr[sid],
                    "room_to_resistance_pct": res if np.isfinite(res) else np.nan, "blue_sky": bool(res == np.inf),
                    "pre_trade_rr": (min(res, 1.0) if np.isfinite(res) else 1.0) / sd if sd > 0 else np.nan,
                    "supply15": G["supply15"].iat[t, r.j], "rs120_pct": np.nan,
                    "rs_leads120_10d": G["rs_leads120_10d"].iat[t, r.j], "atr5_20": G["atr5_20"].iat[t, r.j],
                    "vol10_60": G["vol10_60"].iat[t, r.j], "ext_ma60_atr": G["ext_ma60_atr"].iat[t, r.j],
                    "mrs": ctx.mrs.to_numpy()[i, r.j], "secmkt_20": F["secmkt_20"].iat[t, r.j]})
    df = pd.DataFrame(out)
    mk = p.market["adj_close"]
    ex = (p.c / p.c.shift(120) - 1).sub(mk / mk.shift(120) - 1, axis=0).iloc[t]
    pct = ex[p.universe.iloc[t]].rank(pct=True)
    df["rs120_pct"] = df["stock_id"].map(pct)
    df["leader_type"] = np.select([(df["rs120_pct"] >= 0.8) & (df["secmkt_20"] > 0),
                                   (df["rs120_pct"] >= 0.8) & (df["secmkt_20"] <= 0)],
                                  ["SECTOR_CONFIRMED", "INDEPENDENT"], "NOT_LEADER")
    return df.sort_values("hrr_score", ascending=False).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["state", "momentum", "weinstein", "highrr", "all"])
    a = ap.parse_args(argv)
    if a.step == "state":
        build_state()
        return
    p, F, G, ctx = load_state()
    if a.step in ("momentum", "all"):
        momentum(p, F)
    if a.step in ("weinstein", "all"):
        weinstein(p, F, G, ctx)
    if a.step in ("highrr", "all"):
        wtr = pd.read_pickle(P2 / "weinstein_trades.pkl") if (P2 / "weinstein_trades.pkl").exists() else None
        highrr(p, F, G, ctx, wtr)


if __name__ == "__main__":
    main()
