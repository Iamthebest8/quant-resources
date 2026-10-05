"""1-SECOND EXECUTION research (phase 2): WHEN to enter a candidate chosen by the daily/weekly engines.

Candidate days per strategy (the 1-second engine never chooses WHAT):
  MOMENTUM_INTRADAY_TRIGGER   frozen MOMENTUM_LONG_BASELINE probe days       daily entry = OPEN
  WEINSTEIN_BREAKOUT_TRIGGER  W1/W2 TEXTBOOK fills (day the buy-stop crossed) daily entry = BREAKOUT_IMMEDIATE
  WEINSTEIN_PULLBACK_TRIGGER  W3 TEXTBOOK entries                           daily entry = OPEN
  WEINSTEIN_SHORT_TRIGGER     S1 (breakdown day, daily = BREAKOUT_IMMEDIATE) and S2 (OPEN) TEXTBOOK entries

Look-ahead guard: on breakout/breakdown candidate days the day itself is known only because the trigger
crossed, so every policy there (incl. LEARNED) may act ONLY after the first crossing second.
Chronological split: TRAIN 2023-01-01..2024-12-31, TEST 2025-01-01..2026-09-03 (~40% of days).
Model / threshold are fitted on TRAIN only and frozen (outputs/frozen/INTRADAY_TRIGGERS_V1.json) before TEST.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from pipeline.phase2 import COST, END, OUT, P2, SLIP, bootstrap_p, log, now_utc, save_csv  # noqa: E402

ROOT_DIR = config.ROOT if not config.IS_SYNTHETIC else OUT

BREAKOUT_STRATS = {"WEINSTEIN_BREAKOUT_TRIGGER", "WEINSTEIN_SHORT_TRIGGER_S1"}


def raw_factor(p, t, j) -> float:
    c = p.c.iat[t, j]
    r = p.raw_c.iat[t, j]
    return r / c if c > 0 and np.isfinite(r) else np.nan


def candidates_momentum(p, F) -> pd.DataFrame:
    pe = pd.read_csv(OUT / "PROBE_EVENTS.csv", dtype={"stock_id": str}, parse_dates=["signal_date", "probe_date",
                                                                                        "exit_date"])
    rows = []
    ids = {s: j for j, s in enumerate(p.ids)}
    vol20 = p.vol.rolling(20, min_periods=15).mean()
    for r in pe.itertuples():
        j = ids.get(r.stock_id)
        if j is None or r.probe_date not in p.dates:
            continue
        e = p.dates.get_loc(r.probe_date)
        t = e - 1
        f = raw_factor(p, t, j)
        x = p.dates.get_indexer([pd.Timestamp(r.exit_date)])[0] if pd.notna(r.exit_date) else -1
        exit_adj = r.exit_price * raw_factor(p, x, j) if x >= 0 else np.nan   # phase-2 scale (see momentum_trades)
        rows.append({"cand_id": f"MOM_{r.stock_id}_{r.probe_date.date()}", "strategy": "MOMENTUM_INTRADAY_TRIGGER",
                     "stock_id": r.stock_id, "date": str(r.probe_date.date()), "side": 1,
                     "trigger": p.h.iat[t, j] * f, "stop": r.probe_stop, "atr": F["atr"].iat[t, j] * f,
                     "prev_close": p.raw_c.iat[t, j], "adv_lots": vol20.iat[t, j] / 1000,
                     "next_close": p.raw_c.iat[e + 1, j] if e + 1 < len(p.dates) else np.nan,
                     "t": t, "j": j, "e": e, "daily_entry_adj": r.probe_price * raw_factor(p, e, j),
                     "daily_trade_ret": r.pnl_slots / r.probe_size if r.probe_size else np.nan,
                     "exit_price_adj": exit_adj, "baseline": "OPEN"})
    return pd.DataFrame(rows)


def candidates_weinstein(p, F, wtr: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    tb = wtr[wtr["variant"] == "TEXTBOOK"]
    vol20 = p.vol.rolling(20, min_periods=15).mean()
    rows = []
    for r in tb.itertuples():
        e = p.dates.get_loc(r.entry_date)
        t = e - 1
        j = int(r.j)
        f = raw_factor(p, e, j)
        if r.engine in ("W1", "W2"):
            strat, base = "WEINSTEIN_BREAKOUT_TRIGGER", "BREAKOUT_IMMEDIATE"
        elif r.engine == "W3":
            strat, base = "WEINSTEIN_PULLBACK_TRIGGER", "OPEN"
        elif r.engine == "S1":
            strat, base = "WEINSTEIN_SHORT_TRIGGER_S1", "BREAKOUT_IMMEDIATE"
        else:
            strat, base = "WEINSTEIN_SHORT_TRIGGER_S2", "OPEN"
        side = 1 if r.side == "LONG" else -1
        trig_adj = r.entry_price_adj / (1 + side * SLIP)     # daily fill price before slippage
        level = r.level if np.isfinite(r.level) else trig_adj
        st_trig = setups.at[r.setup_idx, "trigger"] if r.setup_idx in setups.index else np.nan
        rows.append({"cand_id": f"{r.engine}_{r.stock_id}_{pd.Timestamp(r.entry_date).date()}", "strategy": strat,
                     "engine": r.engine, "stock_id": r.stock_id, "date": str(pd.Timestamp(r.entry_date).date()),
                     "side": side, "trigger": (st_trig if np.isfinite(st_trig) else level) * f,
                     "stop": r.stop_price, "atr": F["atr"].iat[t, j] * f, "prev_close": p.raw_c.iat[t, j],
                     "adv_lots": vol20.iat[t, j] / 1000,
                     "next_close": p.raw_c.iat[e + 1, j] if e + 1 < len(p.dates) else np.nan,
                     "t": int(r.t), "j": j, "e": e, "daily_entry_adj": r.entry_price_adj,
                     "daily_trade_ret": r.ret, "setup_idx": r.setup_idx, "baseline": base})
    return pd.DataFrame(rows)


def write_requests(c: pd.DataFrame, name: str) -> Path:
    path = OUT / "intraday" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    c[["stock_id", "date"]].drop_duplicates().to_csv(path, index=False)
    return path


def policies(ds: pd.DataFrame, model=None, thr=None, after_cross: bool = False) -> dict:
    from intraday.engine import FEATURES
    from intraday.research import _first
    P = {}
    base = ds[ds["crossed"] > 0] if after_cross else ds
    P["OPEN"] = ds.sort_values(["cand_id", "sec"]).drop_duplicates("cand_id", keep="first")
    crossed = ds["crossed"] > 0
    P["A_BREAKOUT_IMMEDIATE"] = _first(ds, crossed)
    P["B_BREAKOUT_HOLD_60S"] = _first(ds, crossed & (ds["secs_since_cross"] >= 60) & (ds["dist_trigger"] > 0))
    P["C_BREAKOUT_RETEST_HOLD"] = _first(ds, ds["retest_hold"] > 0)
    P["D_BREAKOUT_MICRO_HL"] = _first(ds, crossed & (ds["micro_hl"] > 0) & (ds["dist_trigger"] > 0))
    P["E_BREAKOUT_VWAP_RECLAIM"] = _first(ds, crossed & (ds["vwap_reclaim"] > 0))
    P["F_BREAKOUT_REL_STRENGTH"] = _first(ds, crossed & (ds["rel_r_5m"] > 0.005))
    P["VWAP_RECLAIM_ANY"] = _first(ds, ds["vwap_reclaim"] > 0)
    P["MICRO_HL_ANY"] = _first(ds, ds["micro_hl"] > 0)
    if model is not None and thr is not None and len(base):
        s = model.predict(base[FEATURES].astype(float))
        P["LEARNED"] = _first(base, pd.Series(s >= thr, index=base.index))
    return P


def learned_trade_ev(p, M, E, rows: pd.DataFrame, cands: pd.DataFrame, wtr: pd.DataFrame | None) -> pd.DataFrame:
    """Multi-day trade return when entering at the intraday-selected price.
    Weinstein: exact re-simulation with fixed_entry and the same frozen exit.
    Momentum: same exit price as the frozen V1 campaign, entry price replaced (campaign exits do not depend on
    the entry price except through the fixed stop, which is unchanged) -> approximation, labelled as such."""
    from alpha.trades import simulate
    from pipeline.phase2 import TEXTBOOK_EXIT
    end_idx = int(p.dates.searchsorted(pd.Timestamp(END), side="right") - 1)
    cm = cands.set_index("cand_id")
    out = []
    for r in rows.itertuples():
        c = cm.loc[r.cand_id]
        f = raw_factor(p, int(c.e), int(c.j))
        adj_px = r.entry_price / f * (1 + c.side * SLIP)
        if c.strategy == "MOMENTUM_INTRADAY_TRIGGER":
            # probe-leg return with the campaign's own exit (adds / scaling of V1 are NOT re-simulated)
            ret = c.side * (c.exit_price_adj / adj_px - 1) - COST
            out.append({"cand_id": r.cand_id, "trade_ret_intraday": ret, "trade_ret_daily": c.daily_trade_ret,
                        "method": "probe_leg_same_exit(approx)"})
        else:
            eng = c.engine
            stop_adj = c.stop / f
            res = simulate(M, int(c.j), int(c.t), int(c.side), "open", np.nan, stop_adj, E[TEXTBOOK_EXIT[eng]],
                           end_idx, COST, SLIP, fixed_entry=(int(c.e), adj_px))
            ms = getattr(r, "micro_stop", np.nan)
            res_m = None
            if np.isfinite(ms):
                ms_adj = ms / f * (1 - c.side * 0.001)      # one tick-ish beyond the 5-min swing extreme
                res_m = simulate(M, int(c.j), int(c.t), int(c.side), "open", np.nan, ms_adj, E[TEXTBOOK_EXIT[eng]],
                                 end_idx, COST, SLIP, fixed_entry=(int(c.e), adj_px))
            out.append({"cand_id": r.cand_id, "trade_ret_intraday": res["ret"] if res else np.nan,
                        "trade_ret_daily": c.daily_trade_ret, "method": "exact_resimulation",
                        "trade_ret_micro_stop": res_m["ret"] if res_m else np.nan,
                        "r_multiple_micro_stop": res_m["r_multiple"] if res_m else np.nan,
                        "r_multiple_daily_stop": res["r_multiple"] if res else np.nan,
                        "stop_dist_atr_daily": res["stop_dist_atr"] if res else np.nan,
                        "stop_dist_atr_micro": res_m["stop_dist_atr"] if res_m else np.nan})
    return pd.DataFrame(out)


def run(p, F, G, ctx, wtr: pd.DataFrame, setups: pd.DataFrame, step: int = 15, download: bool = True,
        reuse: bool = False):
    from alpha.trades import attach_weekly, build_mats
    from intraday.engine import FEATURES, utility
    from intraday.research import TRAIN_END, build, choose_threshold, fit_model, summarize
    from pipeline.phase2 import exit_specs
    t0 = time.time()
    cm = candidates_momentum(p, F)
    cw = candidates_weinstein(p, F, wtr, setups)
    cands = pd.concat([cm, cw], ignore_index=True)
    cands = cands[(cands["date"] >= "2023-01-01") & (cands["date"] <= END)
                  & np.isfinite(cands["trigger"]) & np.isfinite(cands["stop"])].drop_duplicates("cand_id")
    write_requests(cw, "tick_requests_weinstein.csv")
    if download:
        from data.ticks import download as dl
        log(f"[intraday] downloading ticks for {cw[['stock_id', 'date']].drop_duplicates().shape[0]} Weinstein days")
        dl(cw[["stock_id", "date"]].drop_duplicates(), log=log)
    dsp = P2 / "intraday_dataset.pkl"
    if reuse and dsp.exists():
        ds = pd.read_pickle(dsp)
        ds = ds[ds["cand_id"].isin(set(cands["cand_id"]))].drop(columns=["baseline"], errors="ignore")
        log(f"[intraday] reused dataset {len(ds):,} rows")
    else:
        ds = build(cands, step=step, log=log)
    if ds.empty:
        log("[intraday] no tick data available")
        return None
    ds = ds.merge(cands[["cand_id", "baseline"]], on="cand_id", how="left")
    ds.to_pickle(P2 / "intraday_dataset.pkl")
    M = attach_weekly(build_mats(p, F, G), p, ctx)
    E = exit_specs()
    frz_path = OUT / "frozen" / "INTRADAY_TRIGGERS_V1.json"
    frozen = json.loads(frz_path.read_text(encoding="utf-8")) if frz_path.exists() else {}
    from data.ticks import sec_to_time
    res_rows, ev_rows, thr_rows, comp_rows, pol_rows = [], [], [], [], []
    models = {}
    strategies = sorted(ds["strategy"].unique())
    for strat in strategies:
        d = ds[ds["strategy"] == strat]
        after = strat in BREAKOUT_STRATS
        tr, te = d[d["split"] == "TRAIN"], d[d["split"] == "TEST"]
        n_tr, n_te = tr["cand_id"].nunique(), te["cand_id"].nunique()
        log(f"[intraday] {strat}: TRAIN days={n_tr} TEST days={n_te}")
        if n_tr < 40 or n_te < 20:
            res_rows.append({"strategy": strat, "note": f"insufficient days TRAIN={n_tr} TEST={n_te}"})
            continue
        fit_on = tr[tr["crossed"] > 0] if after else tr
        model = fit_model(fit_on)
        thr, tab = choose_threshold(model, fit_on)
        if strat in frozen:
            thr = frozen[strat]["threshold"]           # never re-chosen after freezing
        else:
            frozen[strat] = {"threshold": thr, "frozen_utc": now_utc(), "train_end": TRAIN_END,
                             "model": "HistGradientBoostingRegressor(depth4,200it,lr.05,leaf200,l2=1,seed0)",
                             "objective": "utility = MFE_EOD - |MAE_EOD| (lambda=1)", "after_cross_only": after}
        thr_rows.append(tab.assign(strategy=strat, chosen=thr))
        models[strat] = model
        for split, dd in (("TRAIN", tr), ("TEST", te)):
            P = policies(dd, model, thr, after_cross=after)
            for pk, prow in P.items():
                if prow is None or len(prow) == 0:
                    continue
                keep = prow[["cand_id", "stock_id", "date", "sec", "entry_price", "fwd_15m", "fwd_60m", "fwd_eod",
                             "mfe_eod", "mae_eod", "stop_dist_daily_atr", "stop_dist_micro_atr", "trigger", "stop"]].copy()
                keep["utility"] = utility(prow).to_numpy()
                keep["entry_time"] = [sec_to_time(int(x) + 1) for x in keep["sec"]]
                pol_rows.append(keep.assign(strategy=strat, split=split, policy=pk))
            sm = summarize(P, dd["cand_id"].nunique(), split)
            sm.insert(0, "strategy", strat)
            sm["baseline_policy"] = "OPEN" if not after else "A_BREAKOUT_IMMEDIATE"
            res_rows.append(sm)
            if split == "TEST":
                base_k = "A_BREAKOUT_IMMEDIATE" if after else "OPEN"
                L = P.get("LEARNED")
                if L is not None and len(L):
                    u_l = utility(L).to_numpy()
                    p_open = bootstrap_p(u_l, utility(P["OPEN"]).to_numpy())
                    p_brk = bootstrap_p(u_l, utility(P["A_BREAKOUT_IMMEDIATE"]).to_numpy())
                    ev = learned_trade_ev(p, M, E, L, cands, wtr)
                    bl = P[base_k]
                    evb = learned_trade_ev(p, M, E, bl, cands, wtr)
                    comp_rows.append({"strategy": strat, "test_days": dd["cand_id"].nunique(),
                                      "learned_fill_rate": len(L) / dd["cand_id"].nunique(),
                                      "u_learned": float(u_l.mean()),
                                      "u_open": float(utility(P["OPEN"]).mean()),
                                      "u_breakout": float(utility(P["A_BREAKOUT_IMMEDIATE"]).mean())
                                      if len(P["A_BREAKOUT_IMMEDIATE"]) else np.nan,
                                      "p_vs_open": p_open, "p_vs_breakout": p_brk,
                                      "p": (p_brk if after else (max(p_open, p_brk) if np.isfinite(p_brk) else p_open)),
                                      "trade_ev_learned": float(ev["trade_ret_intraday"].mean()),
                                      # same-method comparison: baseline policy (daily entry) re-priced the same way
                                      "trade_ev_daily": float(evb["trade_ret_intraday"].mean()),
                                      "trade_ev_daily_simulator": float(cands[cands["strategy"] == strat].merge(
                                          dd[["cand_id"]].drop_duplicates(), on="cand_id")["daily_trade_ret"].mean()),
                                      "trade_ev_method": ev["method"].iloc[0] if len(ev) else "",
                                      "stop_dist_lod_atr_learned": float(L["stop_dist_lod_atr"].median()),
                                      "lod_stop_hit_learned": float(L["lod_stop_hit"].mean()),
                                      "stop_dist_daily_atr_learned": float(L["stop_dist_daily_atr"].median()),
                                      "stop_dist_micro_atr_learned": float(L["stop_dist_micro_atr"].median()),
                                      "stop_dist_daily_atr_baseline": float(bl["stop_dist_daily_atr"].median()),
                                      "micro_stop_hit_learned": float(L["micro_stop_hit"].mean()),
                                      "daily_stop_hit_learned": float(L["daily_stop_hit"].mean()),
                                      "learned_median_time": L["time"].sort_values().iloc[len(L) // 2]})
                    ev_rows.append(ev.assign(strategy=strat, policy="LEARNED"))
                    ev_rows.append(evb.assign(strategy=strat, policy=base_k))
    frz_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if pol_rows:
        pd.concat(pol_rows, ignore_index=True).to_csv(P2 / "intraday_policy_entries.csv", index=False)
    res = pd.concat([r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in res_rows], ignore_index=True)
    comp = pd.DataFrame(comp_rows)
    save_csv(res, "INTRADAY_TRIGGER_RESULTS.csv")
    if comp_rows:
        comp.to_csv(P2 / "intraday_learned_vs_baseline.csv", index=False)
    if ev_rows:
        pd.concat(ev_rows, ignore_index=True).to_csv(P2 / "intraday_trade_ev.csv", index=False)
    if thr_rows:
        pd.concat(thr_rows, ignore_index=True).to_csv(P2 / "intraday_threshold_selection.csv", index=False)
    # dataset sample CSV (full dataset is a pickle, too large for git)
    pick = (ds[["strategy", "cand_id"]].drop_duplicates().groupby("strategy")["cand_id"]
            .apply(lambda x: list(x.sample(min(4, len(x)), random_state=1))))
    smp = ds[ds["cand_id"].isin({c for v in pick for c in v})]
    smp.to_csv(ROOT_DIR / "INTRADAY_ENTRY_EVENT_DATASET.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    log(f"wrote INTRADAY_ENTRY_EVENT_DATASET.csv sample ({len(smp)} rows; full dataset: outputs/phase2/intraday_dataset.pkl)")
    # feature importance proxy: permutation on TEST utility for each model
    imp_rows = []
    for strat, model in models.items():
        te = ds[(ds["strategy"] == strat) & (ds["split"] == "TEST")]
        if strat in BREAKOUT_STRATS:
            te = te[te["crossed"] > 0]
        if len(te) < 200:
            continue
        X = te[FEATURES].astype(float)
        y = utility(te).to_numpy(float)
        ok = np.isfinite(y)
        base_corr = np.corrcoef(model.predict(X[ok]), y[ok])[0, 1]
        rng = np.random.default_rng(0)
        for f in FEATURES:
            Xp = X[ok].copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            imp_rows.append({"strategy": strat, "feature": f,
                             "test_corr_drop": base_corr - np.corrcoef(model.predict(Xp), y[ok])[0, 1]})
    imp = pd.DataFrame(imp_rows)
    if len(imp):
        imp.to_csv(P2 / "intraday_feature_importance.csv", index=False)
    log(f"[intraday] done in {time.time() - t0:.0f}s")
    return {"res": res, "comp": comp, "imp": imp, "ds": ds, "cands": cands, "frozen": frozen}


def bollinger_b2_ticks(p, tr: pd.DataFrame, M, exit_name: str, download: bool = True) -> pd.DataFrame:
    """1-second verification of Bollinger B2 exits (Strict OOS): exact crossing second of yesterday's band and the
    fill at the next second vs the daily stop-proxy fill used in the simulator."""
    from data.ticks import bars_1s, sec_to_time
    n_, k_ = int(exit_name.split("_")[2]), float(exit_name.split("_")[3])
    up, dn = M.BB[(n_, k_)]
    sub = tr[(tr["exit"] == exit_name) & tr["exit_reason"].str.startswith("BB_B2", na=False)
             & (tr["window"] == "STRICT_OOS")].copy()
    if sub.empty:
        return pd.DataFrame()
    req = pd.DataFrame({"stock_id": sub["stock_id"], "date": sub["exit_date"].dt.strftime("%Y-%m-%d")})
    write_requests(req, "tick_requests_bollinger_b2.csv")
    if download:
        from data.ticks import download as dl
        dl(req.drop_duplicates(), log=log, index_too=False)
    rows = []
    for r in sub.itertuples():
        d = p.dates.get_loc(r.exit_date)
        j = int(r.j)
        side = 1 if r.side == "LONG" else -1
        f = raw_factor(p, d, j)
        lvl = (up[d - 1, j] if side > 0 else dn[d - 1, j]) * f
        b = bars_1s(r.stock_id, str(pd.Timestamp(r.exit_date).date()))
        if b is None or not np.isfinite(lvl):
            rows.append({"stock_id": r.stock_id, "exit_date": r.exit_date, "status": "NO_TICKS"})
            continue
        px = b["close"].to_numpy(float)
        lo = b["low"].to_numpy(float)
        hi = b["high"].to_numpy(float)
        hit = (lo <= lvl) if side > 0 else (hi >= lvl)
        hit &= b["has_trade"].to_numpy()
        if not hit.any():
            rows.append({"stock_id": r.stock_id, "exit_date": r.exit_date, "status": "NO_CROSS_IN_TICKS"})
            continue
        s = int(np.argmax(hit))
        fill = px[min(s + 1, len(px) - 1)]
        o = px[int(np.argmax(b["has_trade"].to_numpy()))]
        proxy = min(o, lvl) if side > 0 else max(o, lvl)
        rows.append({"stock_id": r.stock_id, "exit_date": r.exit_date, "status": "OK", "band_level_raw": lvl,
                     "cross_second": s, "cross_time": sec_to_time(s), "fill_1s_raw": fill, "proxy_fill_raw": proxy,
                     "fill_diff_pct": side * (fill / proxy - 1), "close_raw": px[-1],
                     "close_vs_fill_pct": side * (px[-1] / fill - 1)})
    return pd.DataFrame(rows)
