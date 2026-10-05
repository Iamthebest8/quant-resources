"""Compile the 10 phase-2 verdicts from the saved step reports using the PRE-REGISTERED criteria
(alpha/verdicts2.py). python -m pipeline.phase2_report"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from alpha.verdicts2 import exec_verdict  # noqa: E402
from pipeline.phase2 import OUT, P2, log, now_utc, save_csv  # noqa: E402


def _load(name):
    f = P2 / name
    return pickle.load(open(f, "rb")) if f.exists() else {}


def compile_verdicts() -> pd.DataFrame:
    rows = []
    # 1. MOMENTUM LONG
    mom = pd.read_csv(config.ROOT / "MOMENTUM_LONG_RESULTS.csv")
    fv = pd.read_csv(OUT / "FINAL_VERDICTS.csv") if (OUT / "FINAL_VERDICTS.csv").exists() else pd.DataFrame()
    ref = pd.read_csv(OUT / "frozen" / "MOMENTUM_LONG_BASELINE_reference_metrics.csv").set_index("window")
    full_sys = fv[fv.iloc[:, 0].astype(str).str.contains("FULL", case=False)] if len(fv) else fv
    fs_v = str(full_sys.iloc[0].get("verdict", "")) if len(full_sys) else ""
    pf, ev = ref.at["STRICT_OOS", "pf"], ref.at["STRICT_OOS", "ev_slots_per_probe"]
    v = "ACCEPT" if fs_v == "ACCEPT" else ("WATCH" if pf >= 1.0 and ev > 0 else "REJECT")
    rows.append({"verdict_item": "MOMENTUM LONG", "verdict": v,
                 "evidence": f"frozen V1 Strict OOS trade-level PF={pf:.3f}, EV/probe={ev:.4f} slots; phase-1 "
                             f"FULL_SYSTEM verdict={fs_v or 'n/a'}; rerun on 2019-extended panel within rounding"})
    # 2-4 Weinstein entries
    w = _load("weinstein_report.pkl")
    names = {"W1": "W STAGE1→2", "W2": "W CONTINUATION", "W3": "W PULLBACK"}
    for eng, nm in names.items():
        tv = w.get("entry_verdicts", {}).get((eng, "TEXTBOOK"), ("REJECT", "no data"))
        mv = w.get("entry_verdicts", {}).get((eng, "MODERNIZED"), ("REJECT", "no data"))
        rows.append({"verdict_item": nm, "verdict": tv[0], "evidence": f"TEXTBOOK (primary): {tv[1]}",
                     "secondary": f"MODERNIZED: {mv[0]} — {mv[1]}"})
    # 5 HIGH R/R
    h = _load("highrr_report.pkl")
    hv = h.get("verdict", ("REJECT", "not run"))
    rows.append({"verdict_item": "HIGH R/R FILTER", "verdict": hv[0], "evidence": hv[1],
                 "secondary": f"score features: {list((h.get('spec') or {}).values())}"})
    # 6 1-second engine (per strategy; headline = momentum, the only engine with >=100 TEST days unless others have)
    it = _load("intraday_report.pkl")
    comp = it.get("comp")
    per = []
    head = ("REJECT", "not run")
    if comp is not None and len(comp):
        for r in comp.itertuples():
            breakout_day = r.strategy in ("WEINSTEIN_BREAKOUT_TRIGGER", "WEINSTEIN_SHORT_TRIGGER_S1")
            # on breakout-day candidates OPEN is look-ahead, so both comparators are BREAKOUT_IMMEDIATE
            u_open = r.u_breakout if breakout_day else r.u_open
            u_brk = r.u_breakout if np.isfinite(r.u_breakout) else r.u_open
            vv = exec_verdict({"u_learned": r.u_learned, "u_open": u_open, "u_breakout": u_brk, "p": r.p,
                               "trade_ev_learned": r.trade_ev_learned, "trade_ev_daily": r.trade_ev_daily})
            per.append((r.strategy, vv, int(r.test_days)))
        order = {"ACCEPT": 2, "WATCH": 1, "REJECT": 0}
        big = [x for x in per if x[2] >= 100]
        pool = big or per
        head = max(pool, key=lambda x: (order[x[1][0]], x[2]))[1]
    rows.append({"verdict_item": "1-SECOND ENGINE", "verdict": head[0], "evidence": head[1],
                 "secondary": "; ".join(f"{s}: {vv[0]} (TEST days {n})" for s, vv, n in per)})
    # 7 W SHORT
    sv = w.get("short_verdicts", {})
    tv = sv.get(("TEXTBOOK", "EXECUTABLE"), ("REJECT", "no data"))
    rows.append({"verdict_item": "W SHORT", "verdict": tv[0], "evidence": f"TEXTBOOK EXECUTABLE: {tv[1]}",
                 "secondary": "; ".join(f"{k[0]}/{k[1]}: {v_[0]} ({v_[1]})" for k, v_ in sv.items())})
    # 8 Bollinger
    bb = w.get("bb", ("", ("REJECT", "not run"), None))
    rows.append({"verdict_item": "BOLLINGER EXIT", "verdict": bb[1][0], "evidence": f"frozen {bb[0]}: {bb[1][1]}"})
    # 9 HYBRID
    pr = _load("portfolio_report.pkl")
    hv_ = pr.get("hybrid_verdicts", {})
    order = {"ACCEPT": 2, "WATCH": 1, "REJECT": 0}
    best = max(hv_.items(), key=lambda kv: order[kv[1][0]]) if hv_ else ("", ("REJECT", "not run"))
    rows.append({"verdict_item": "HYBRID", "verdict": best[1][0], "evidence": f"best = {best[0]}: {best[1][1]}",
                 "secondary": "; ".join(f"{k}: {v_[0]}" for k, v_ in hv_.items())})
    # 10 portfolio
    pv = pr.get("portfolio_verdict", ("REJECT", "not run"))
    rows.append({"verdict_item": "MULTI-ALPHA PORTFOLIO", "verdict": pv[0], "evidence": pv[1]})
    df = pd.DataFrame(rows)
    for col in ("evidence", "secondary"):
        df[col] = df[col].astype(str).str.replace(r"np\.float64\(([^)]*)\)", lambda m: f"{float(m.group(1)):.3f}",
                                                  regex=True).replace("nan", "")
    df["criteria_file"] = "alpha/verdicts2.py (registered 2026-10-05T17:08Z, commit 70ba58d)"
    df["compiled_utc"] = now_utc()
    save_csv(df, "PHASE2_VERDICTS.csv")
    return df


if __name__ == "__main__":
    print(compile_verdicts().to_string())
