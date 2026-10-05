"""End-to-end research pipeline.

    python -m pipeline.run_research                 # discovery (if not frozen) + validation
    python -m pipeline.run_research --refreeze V2   # deliberate new version (logged)

Discipline (Parts 36-41):
  * Phase DISCOVERY uses only 2023-01-01 ~ 2024-12-31 rows, purged by the 40D label horizon,
    and campaigns force-closed at 2024-12-31. It selects features + rules and FREEZES V1
    (outputs/frozen/V1.json, with config hash). A frozen version is never silently overwritten.
  * Phase VALIDATION loads the frozen config and evaluates every period; nothing it computes
    feeds back into the rules.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from engine.features import compute_features, cs_pct  # noqa: E402
from engine.panel import build_panel  # noqa: E402
from pipeline.verdicts import CRITERIA, decide  # noqa: E402
from research import analysis as AN  # noqa: E402
from research import studies as ST  # noqa: E402
from research.labels import compute_labels, period_label, research_frame  # noqa: E402
from strategy.config import StrategyConfig  # noqa: E402
from strategy.engine import trade_level  # noqa: E402
from strategy.metrics import campaign_metrics  # noqa: E402
from strategy.portfolio import simulate_portfolio  # noqa: E402
from strategy.selection import SignalCache, staged_selection, window_idx  # noqa: E402
from strategy.signals import build_arrays, condition_frames, discovery_score  # noqa: E402

OUT = config.OUT_DIR
FROZEN = OUT / "frozen"
SIG_DIR = OUT / "signals"
FROZEN.mkdir(parents=True, exist_ok=True)
SIG_DIR.mkdir(parents=True, exist_ok=True)
SLIP = config.BASE_SLIPPAGE_BPS / 1e4
COST = config.BASE_COST


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.6g")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class DecisionLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, version, phase, step, decision, value="", window="", evidence="", cfg_hash="", ts=None):
        self.rows.append({"timestamp_utc": ts or now_utc(), "version": version, "phase": phase, "step": step,
                          "decision": decision, "value": value, "data_window": window, "evidence": evidence,
                          "config_hash": cfg_hash})

    def save(self):
        save(pd.DataFrame(self.rows), "STRATEGY_DECISION_LOG.csv")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run_tl(A, sc, cfg, start, end, cost=COST, slip=SLIP, keep=False, live=False):
    sig, rank, _ = sc.get(cfg)
    s, e = window_idx(A.dates, start, end)
    return trade_level(A, sig, cfg, s, e, slip, cost, rank, keep_campaigns=keep, live=live)


def run_pf(A, sc, cfg, start, end, cost=COST, slip_bps=config.BASE_SLIPPAGE_BPS, positions=False, live=False):
    sig, rank, _ = sc.get(cfg)
    s, e = window_idx(A.dates, start, end)
    return simulate_portfolio(A, sig, rank, cfg, s, e, cost=cost, slip_bps=slip_bps, record_positions=positions,
                              live=live)


def years_of(start, end):
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25


def add_period(df: pd.DataFrame, col="probe_date") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df["period"] = period_label(df[col])
    df["year"] = df[col].dt.year
    return df


def bootstrap_p(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 3) -> float:
    """One-sided p-value for mean(a) > mean(b) via bootstrap of the difference."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    diffs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)])
    return float((diffs <= 0).mean())


# ---------------------------------------------------------------------------
# PHASE 1: DISCOVERY -> freeze
# ---------------------------------------------------------------------------
def discovery_phase(p, F, R, A, sc, version: str, dlog: DecisionLog) -> StrategyConfig:
    log(f"=== DISCOVERY phase ({config.DISCOVERY[0]} ~ {config.DISCOVERY[1]}) -> {version}")
    D = ST.purged(R, *config.DISCOVERY, p.dates)
    sel = ST.select_discovery_features(D)
    save(sel["table"].assign(window="DISCOVERY_purged"), "FEATURE_LIFT_DISCOVERY.csv")
    feats = tuple(v["feature"] for v in sel["families"].values())
    if not feats:
        feats = ("outp_10", "srs_10", "rs_accel")
        dlog.add(version, "DISCOVERY", "features", "NO family passed lift>=1.10; fallback to pre-declared trio",
                 ",".join(feats), "2023-2024 purged")
    for fam, v in sel["families"].items():
        dlog.add(version, "DISCOVERY", "discovery_feature", f"family {fam} -> {v['feature']}",
                 f"lift={v['lift']}", "2023-2024 purged", "vol-controlled lift(leader20) top quintile")
    cfg, stages = staged_selection(A, sc, feats, config.DISCOVERY, version=version, log=log)
    save(stages, f"SELECTION_STAGES_{version}.csv")
    for st, g in stages.groupby("stage"):
        row = g[g["selected"]].iloc[0]
        dlog.add(version, "DISCOVERY", f"stage{st}_{row['stage_name']}", "selected", row["params"],
                 "2023-2024", f"objective={row['objective']:.4f}; n_candidates={len(g)}")
    frozen = {"version": version, "frozen_utc": now_utc(), "config": cfg.to_dict(), "hash": cfg.hash(),
              "data_last_date": str(p.dates[-1].date()), "selection_window": config.DISCOVERY,
              "discovery_features": sel["families"], "criteria": CRITERIA, "decisions": dlog.rows,
              "data_source": config.DATA_SOURCE}
    (FROZEN / f"{version}.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
    dlog.add(version, "DISCOVERY", "FREEZE", f"{version} frozen before any OOS evaluation", cfg.short(),
             "2023-2024", "", cfg.hash())
    log(f"FROZEN {version}: {cfg.short()} hash={cfg.hash()}")
    return cfg


# ---------------------------------------------------------------------------
# PHASE 2: VALIDATION
# ---------------------------------------------------------------------------
def validation_phase(p, F, L, R, A, sc, cfg: StrategyConfig, dlog: DecisionLog) -> dict:
    V = cfg.version
    log(f"=== VALIDATION phase for {V} ({cfg.short()})")
    res: dict = {"cfg": cfg}
    U = p.universe
    disc = discovery_score(F, U, cfg.disc_features)
    conds = condition_frames(p, F, cfg, disc_pct=disc)
    idx = pd.MultiIndex.from_frame(R[["date", "stock_id"]])
    R["disc_pct"] = disc.stack(future_stack=True).reindex(idx).to_numpy(dtype="float32")
    R["probe_signal"] = conds["probe_signal"].stack(future_stack=True).reindex(idx).to_numpy()

    periods = {
        "DISCOVERY": ST.purged(R, *config.DISCOVERY, p.dates),
        "STRICT_OOS": ST.window(R, *config.STRICT_OOS),
        **{f"Y{y}": ST.window(R, *w) for y, w in config.YEARS.items()},
    }
    # ---------------- research studies -------------------------------------------------
    log("studies: response surfaces / independent strength / resilience / sector")
    surf = ST.feature_response_surface(R, periods)
    mkt_fams = ["MKT_REL", "PERSIST", "ACCEL", "INDEP", "RESIL", "UPPART", "ASYM", "TRAD_RS", "STRUCT", "VOLUME"]
    lift_rows = []
    for pn, D in periods.items():
        t = ST.feature_lift_table(D)
        t["period"] = pn
        lift_rows.append(t)
        flag = D["disc_pct"] >= cfg.disc_q
        for lab in ("leader20", "leader30", "leader40", "bad10"):
            l_, d_, n_ = ST.stratified_lift(D, flag, lab)
            lift_rows.append(pd.DataFrame([{"family": "DISCOVERY_SCORE", "feature": f"disc_pct>={cfg.disc_q}",
                                            "period": pn, f"lift_{lab}_volctl": l_, "n_top": n_,
                                            f"diff_{lab}_volctl": d_}]))
    lifts = pd.concat(lift_rows, ignore_index=True)
    save(lifts, "FEATURE_LIFT_BY_PERIOD.csv")
    save(surf[surf["family"].isin(mkt_fams)], "MARKET_RELATIVE_STRENGTH.csv")
    sec_tab = ST.sector_decomposition(periods)
    save(pd.concat([surf[surf["family"] == "SECTOR_REL"].assign(table="feature_quintiles"),
                    sec_tab.assign(table="stock_vs_sector_x_sector_vs_market")], ignore_index=True),
         "SECTOR_RELATIVE_STRENGTH.csv")
    save(ST.independent_strength_surface(periods), "INDEPENDENT_STRENGTH_SURFACE.csv")
    inc = pd.DataFrame([{"period": pn, **ST.independent_strength_increment(D)} for pn, D in periods.items()])
    save(inc, "INDEPENDENT_STRENGTH_INCREMENT.csv")
    save(ST.resilience_study(periods), "RESILIENCE_PARTICIPATION.csv")
    save(ST.regime_profile(periods), "REGIME_PROFILE.csv")
    res.update(lifts=lifts, inc=inc, sec_tab=sec_tab)

    # ---------------- discovery events -----------------------------------------------------
    de = R[(R["disc_pct"] >= cfg.disc_q)].copy()
    prev = disc.shift(1).stack(future_stack=True).reindex(pd.MultiIndex.from_frame(de[["date", "stock_id"]]))
    de = de[(prev.to_numpy() < cfg.disc_q) | np.isnan(prev.to_numpy())]
    keep = ["date", "stock_id", "sector", "period", "regime", "disc_pct", "probe_signal", "cex_10", "srs_10",
            "outp_10", "rs_accel", "is_cnt10", "dr60", "up60", "rs_20", "rs_60", "atr_pct", "beta", "val20",
            "fwd_ret_20", "fwd_ret_40", "fwd_maxc_40", "fwd_mae_40", "leader20", "leader30", "leader40"]
    de = de[[c for c in keep if c in de.columns]]
    de.insert(2, "name", de["stock_id"].map(p.names))
    save(de, "DISCOVERY_EVENTS.csv")
    res["discovery_events"] = de

    # ---------------- trade-level V1 ---------------------------------------------------------
    log("trade-level runs")
    W = {"DISCOVERY": config.DISCOVERY, "EXTENDED": config.EXTENDED_VALIDATION, "STRICT_OOS": config.STRICT_OOS,
         "FULL": (config.RESEARCH_START, config.BACKTEST_END)}
    tl = {k: add_period(run_tl(A, sc, cfg, *w)) for k, w in W.items()}
    live_df, live_camps = run_tl(A, sc, cfg, config.RESEARCH_START, str(p.dates[-1].date()), keep=True, live=True)
    tl["LIVE"] = add_period(live_df)
    res["tl"] = tl
    tm = {k: campaign_metrics(v, years=years_of(*W[k]) if k in W else None) for k, v in tl.items()}
    res["tm"] = tm
    save(pd.DataFrame([{"window": k, **m} for k, m in tm.items()]), "TRADE_LEVEL_SUMMARY.csv")
    full = tl["FULL"]

    # ---------------- event files ------------------------------------------------------------
    def nm(df):
        df = df.copy()
        df.insert(1, "name", df["stock_id"].map(p.names))
        df.insert(2, "sector", df["stock_id"].map(p.sector))
        return df
    sf = AN.signal_features(p, F, conds, full)
    why_cols = ["disc_pct", "rank_score", "cex_10", "srs_10", "outp_10", "rs_accel", "is_cnt10", "ext_ma20",
                "atr_pct", "regime"]

    def why_probe(r):
        return (f"Disc={r.disc_pct:.2f} 10D超額={r.cex_10:+.1%} 產業超額={r.srs_10:+.1%} "
                f"outperf10={r.outp_10:.0%} RSaccel={r.rs_accel:+.4f} 觸發={cfg.trigger} regime={r.regime}")

    why_failed_map = {"F1_PROBE_LOW": "跌破 Probe Low（盤中停損）", "F2_STRUCTURE": "收盤跌破 MA10 且低於成本（短結構失敗）",
                      "F3_RS_FAILURE": f"試單後相對大盤落後 ≤ {cfg.f3_thresh:.0%}（RS 失敗）",
                      "PROBE_MAX_LIFE": f"{cfg.probe_max_life} 日內未確認", "END_OF_WINDOW": "期間結束",
                      "DELISTED/SUSPENDED": "停止交易"}
    sf["why_probe"] = sf.apply(why_probe, axis=1) if len(sf) else []
    probes = nm(sf)
    save(probes, "PROBE_EVENTS.csv")
    failed = probes[probes["final_state"] == "FAILED_PROBE"].copy()
    failed["why_failed"] = failed["exit_reason"].map(lambda x: why_failed_map.get(x, x if not str(x).startswith(
        "F4") else f"時間停損：{x[-3:-1]}日內未上漲"))
    save(failed[["stock_id", "name", "sector", "period", "signal_date", "probe_date", "probe_price", "probe_size",
                 "probe_stop", "probe_risk", "exit_date", "exit_price", "exit_reason", "why_failed", "why_probe",
                 "ret_on_invested", "pnl_slots", "probe_mae", "probe_mfe", "holding_days"] + why_cols],
         "FAILED_PROBES.csv")
    conf = probes[probes["confirmed"]]
    save(conf[["stock_id", "name", "sector", "period", "probe_date", "probe_price", "confirm_date", "confirm_price",
               "confirm_rule", "days_probe_to_confirm", "probe_mfe", "probe_mae", "add_date", "add_price",
               "exit_date", "exit_reason", "ret_on_invested", "pnl_slots"]], "CONFIRMATION_EVENTS.csv")
    adds = probes[probes["n_adds"] > 0]
    save(adds[["stock_id", "name", "sector", "period", "probe_date", "probe_price", "confirm_date", "add_date",
               "add_price", "add_premium", "add2_date", "add2_price", "add_mfe", "add_mae", "add_leg_ret",
               "exit_date", "exit_price", "exit_reason", "ret_on_invested", "pnl_slots"]], "ADD_EVENTS.csv")
    fulls = probes[probes["full"]]
    save(fulls[["stock_id", "name", "sector", "period", "probe_date", "confirm_date", "add_date", "add2_date",
                "max_size", "exit_date", "exit_reason", "holding_days", "mfe", "mae", "ret_on_invested",
                "pnl_slots"]], "FULL_POSITION_EVENTS.csv")
    res["probes"] = probes

    # ---------------- success / failure case tables (Parts 53-54) -------------------------------
    succ, fl = [], []
    for pn in ("DISCOVERY", "STRICT_OOS"):
        g = probes[probes["period"] == pn]
        succ.append(g[g["pnl"] > 0].sort_values("ret_on_invested", ascending=False).head(12))
        gf = failed[failed["period"] == pn]
        rep = pd.concat([gf.sort_values("ret_on_invested").head(4),
                         gf.iloc[(gf["ret_on_invested"] - gf["ret_on_invested"].median()).abs().argsort()[:4]]
                         if len(gf) else gf,
                         gf.sort_values("disc_pct", ascending=False).head(4)]).drop_duplicates(
            ["stock_id", "probe_date"])
        fl.append(rep.head(12))
    case_cols = ["period", "stock_id", "name", "sector", "probe_date", "probe_price", "probe_size", "confirm_date",
                 "confirm_rule", "add_date", "add_price", "exit_date", "exit_price", "exit_reason", "ret_on_invested",
                 "pnl_slots", "mfe", "mae", "holding_days"]
    save(pd.concat(succ)[case_cols], "CASE_SUCCESS.csv")
    save(pd.concat(fl)[case_cols[:6] + ["exit_date", "exit_price", "exit_reason", "ret_on_invested", "probe_mae",
                                        "holding_days", "why_probe", "why_failed"]], "CASE_FAILED.csv")
    res["case_success"], res["case_failed"] = pd.concat(succ), pd.concat(fl)

    # ---------------- yearly & architecture comparisons ------------------------------------------
    log("yearly / architecture / exit comparisons")
    yrows = []
    for y, w in config.YEARS.items():
        g = full[full["probe_date"].dt.year == int(y[:4])]
        yrows.append({"year": y, "level": "trade", **campaign_metrics(g, years=years_of(*w))})
    arch_rows = []
    for pn, w in (("DISCOVERY", config.DISCOVERY), ("STRICT_OOS", config.STRICT_OOS)):
        for arch in ("A", "B", "C", "P25_ONLY", "P50_ONLY"):
            c2 = cfg.with_(add_arch=arch)
            m = campaign_metrics(run_tl(A, sc, c2, *w), years=years_of(*w))
            pm = run_pf(A, sc, c2, *w)["metrics"]
            arch_rows.append({"period": pn, "variant_type": "add_architecture", "variant": arch,
                              "is_frozen_choice": arch == cfg.add_arch, **m,
                              **{f"pf_{k}": v for k, v in pm.items() if not k.startswith("c_")}})
        for ex in ("HS", "TF", "MS", "MP"):
            c2 = cfg.with_(exit=ex)
            m = campaign_metrics(run_tl(A, sc, c2, *w), years=years_of(*w))
            arch_rows.append({"period": pn, "variant_type": "exit", "variant": ex, "is_frozen_choice": ex == cfg.exit,
                              **m})
        for cf in ("price", "rs", "persist", "price+rs", "price+persist", "trend"):
            c2 = cfg.with_(confirm=cf)
            m = campaign_metrics(run_tl(A, sc, c2, *w), years=years_of(*w))
            arch_rows.append({"period": pn, "variant_type": "confirmation", "variant": cf,
                              "is_frozen_choice": cf == cfg.confirm, **m})
        for fl_, ts in (("F1", 5), ("F1+F2", 5), ("F1+F3", 5), ("F1+F4", 3), ("F1+F4", 5), ("F1+F4", 10),
                        ("F1+F2+F3", 5), ("F1+F2+F3+F4", 5)):
            c2 = cfg.with_(fail=fl_, time_stop=ts)
            m = campaign_metrics(run_tl(A, sc, c2, *w), years=years_of(*w))
            arch_rows.append({"period": pn, "variant_type": "failed_probe_rule",
                              "variant": fl_ + (f"({ts})" if "F4" in fl_ else ""),
                              "is_frozen_choice": (fl_ == cfg.fail and (ts == cfg.time_stop or "F4" not in fl_)), **m})
        for k_ in (2, 3, 4):
            for ca, rv in (("A", 1.0), ("B", 1.0), ("B", 1.5)):
                c2 = cfg.with_(max_probes=k_, capital_arch=ca, probe_reserve=rv)
                pm = run_pf(A, sc, c2, *w)["metrics"]
                arch_rows.append({"period": pn, "variant_type": "portfolio_capacity",
                                  "variant": f"max{k_}|cap{ca}{'' if ca == 'A' else rv}",
                                  "is_frozen_choice": (k_ == cfg.max_probes and ca == cfg.capital_arch and
                                                       (ca == "A" or rv == cfg.probe_reserve)),
                                  **{f"pf_{kk}": v for kk, v in pm.items() if not kk.startswith("c_")}})
    arch = pd.DataFrame(arch_rows)
    res["arch"] = arch
    save(arch, "VARIANT_COMPARISON_REPORTED_NOT_SELECTED.csv")

    # ---------------- portfolio ---------------------------------------------------------------
    log("portfolio runs")
    pfr = {k: run_pf(A, sc, cfg, *w, positions=(k == "FULL")) for k, w in W.items()}
    pfr["LIVE"] = run_pf(A, sc, cfg, config.RESEARCH_START, str(p.dates[-1].date()), positions=True, live=True)
    res["pf"] = pfr
    eq = pfr["FULL"]["equity"].copy()
    m_close = p.market["close"].reindex(eq.index)
    eq["taiex_norm"] = m_close / m_close.iloc[0] * config.INITIAL_CAPITAL
    eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
    eq = eq.reset_index()
    save(eq, "FULL_PORTFOLIO_BACKTEST.csv")
    psum = []
    for k, r_ in pfr.items():
        e_ = r_["equity"]["equity"]
        bm = p.market["close"].reindex(e_.index)
        psum.append({"window": k, "start": str(e_.index[0].date()), "end": str(e_.index[-1].date()),
                     "taiex_return": float(bm.iloc[-1] / bm.iloc[0] - 1), **r_["metrics"]})
    psum = pd.DataFrame(psum)
    save(psum, "PORTFOLIO_SUMMARY.csv")
    # yearly portfolio returns from the continuous FULL run
    e_full = pfr["FULL"]["equity"]["equity"]
    for y, w in config.YEARS.items():
        seg = e_full[(e_full.index >= w[0]) & (e_full.index <= w[1])]
        prev = e_full[e_full.index < w[0]]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        bm = p.market["close"]
        bseg = bm[(bm.index >= w[0]) & (bm.index <= w[1])]
        bprev = bm[bm.index < w[0]]
        yrows.append({"year": y, "level": "portfolio(continuous)", "return": float(seg.iloc[-1] / base - 1),
                      "mdd": float((seg / seg.cummax() - 1).min()),
                      "sharpe": float(seg.pct_change().mean() / seg.pct_change().std() * np.sqrt(252)),
                      "taiex_return": float(bseg.iloc[-1] / (bprev.iloc[-1] if len(bprev) else bseg.iloc[0]) - 1)})
    yearly = pd.DataFrame(yrows)
    save(yearly, "YEARLY_VALIDATION.csv")
    res["yearly"] = yearly
    res["psum"] = psum

    # ---------------- costs -----------------------------------------------------------------
    log("cost / slippage grid")
    crow = []
    for pn, w in (("DISCOVERY", config.DISCOVERY), ("STRICT_OOS", config.STRICT_OOS),
                  ("EXTENDED", config.EXTENDED_VALIDATION)):
        for c_ in config.COST_GRID:
            for sb in config.SLIPPAGE_GRID_BPS:
                m = campaign_metrics(run_tl(A, sc, cfg, *w, cost=c_, slip=sb / 1e4), years=years_of(*w))
                pm = run_pf(A, sc, cfg, *w, cost=c_, slip_bps=sb)["metrics"]
                crow.append({"period": pn, "round_trip_cost": c_, "slippage_bps_per_side": sb,
                             "n_probes": m["n_probes"], "pf": m["pf"], "payoff": m["payoff"],
                             "ev_slots_per_probe": m["ev_slots_per_probe"],
                             "pnl_per_100_slot_days": m["pnl_per_100_slot_days"], "win_rate": m["win_rate"],
                             "pf_cagr": pm.get("cagr"), "pf_mdd": pm.get("mdd"), "pf_sharpe": pm.get("sharpe"),
                             "pf_campaign_pf": pm.get("c_pf")})
    costs = pd.DataFrame(crow)
    save(costs, "PROBE_COST_ROBUSTNESS.csv")
    res["costs"] = costs

    # ---------------- placebo (matched controls) ---------------------------------------------------
    log("matched-control placebo")
    prow, placebo_p = [], {}
    for pn, w in (("DISCOVERY", config.DISCOVERY), ("STRICT_OOS", config.STRICT_OOS)):
        s_, e_ = window_idx(A.dates, *w)
        ctrl = AN.matched_controls(p, F, A, conds, tl[pn], cfg, e_, SLIP, COST)
        mp_, mc_ = campaign_metrics(tl[pn], years_of(*w)), campaign_metrics(ctrl, years_of(*w))
        pv = bootstrap_p(tl[pn]["pnl_slots"].to_numpy(), ctrl["pnl_slots"].to_numpy()) if len(ctrl) else np.nan
        placebo_p[pn] = pv
        for grp, m in (("V1_PROBES", mp_), ("MATCHED_CONTROLS", mc_)):
            prow.append({"period": pn, "group": grp, "bootstrap_p_probe_gt_control": pv, **m})
        res[f"ctrl_{pn}"] = ctrl
    plac = pd.DataFrame(prow)
    save(plac, "PLACEBO_MATCHED_CONTROLS.csv")
    res["placebo"] = plac

    # ---------------- right tail ---------------------------------------------------------------
    rt = []
    srcs = [("V1", tl["DISCOVERY"], "DISCOVERY"), ("V1", tl["STRICT_OOS"], "STRICT_OOS"),
            ("MATCHED_CONTROLS", res["ctrl_DISCOVERY"], "DISCOVERY"),
            ("MATCHED_CONTROLS", res["ctrl_STRICT_OOS"], "STRICT_OOS")]
    for y in config.YEARS:
        srcs.append(("V1", full[full["probe_date"].dt.year == int(y[:4])], f"Y{y}"))
    for name, df, pn in srcs:
        if df is None or df.empty:
            continue
        r_ = df["ret_on_invested"]
        m = campaign_metrics(df)
        rt.append({"source": name, "period": pn, "n": len(df), "mean": r_.mean(), "median": r_.median(),
                   **{f"p{q}": r_.quantile(q / 100) for q in (1, 5, 25, 75, 90, 95, 99)}, "max": r_.max(),
                   "ge10": m["ge10"], "ge20": m["ge20"], "ge30": m["ge30"], "ge40": m["ge40"],
                   "top1pct_share": m["top1pct_share"], "top5pct_share": m["top5pct_share"],
                   "top5_trades_share": m["top5_trades_share"], "pnl_ex_top5_slots": m["pnl_ex_top5_slots"],
                   "skew": m["skew"], "pf": m["pf"], "payoff": m["payoff"]})
    save(pd.DataFrame(rt), "PROBE_RIGHT_TAIL.csv")
    res["right_tail"] = pd.DataFrame(rt)

    # ---------------- capital efficiency ------------------------------------------------------------
    ce = []
    for k, r_ in pfr.items():
        m = r_["metrics"]
        ce.append({"window": k, "variant": "V1(frozen)", "pnl_per_100_slot_days_pct": m.get("pnl_per_100_slot_days_pct"),
                   "trade_pnl_per_100_slot_days_slots": m.get("c_pnl_per_100_slot_days"),
                   "avg_occupancy": m.get("avg_occupancy"), "probe_capital_utilization": m.get("probe_capital_utilization"),
                   "avg_probe_capital_pct": m.get("avg_probe_capital_pct"), "failed_probe_slot_days": m.get("failed_probe_slot_days"),
                   "turnover_annual": m.get("turnover_annual"), "opportunity_cost_pct": m.get("opportunity_cost_pct"),
                   "avg_cash_pct": m.get("avg_cash_pct"), "missed_for_capacity": m.get("missed_for_capacity"),
                   "cagr": m.get("cagr"), "mdd": m.get("mdd"), "sharpe": m.get("sharpe")})
    ar = arch[arch["variant_type"].isin(["add_architecture", "portfolio_capacity"])]
    for r_ in ar.itertuples():
        ce.append({"window": r_.period, "variant": f"{r_.variant_type}:{r_.variant}",
                   "pnl_per_100_slot_days_pct": getattr(r_, "pf_pnl_per_100_slot_days_pct", np.nan),
                   "trade_pnl_per_100_slot_days_slots": getattr(r_, "pnl_per_100_slot_days", np.nan),
                   "avg_occupancy": getattr(r_, "pf_avg_occupancy", np.nan),
                   "probe_capital_utilization": getattr(r_, "pf_probe_capital_utilization", np.nan),
                   "failed_probe_slot_days": getattr(r_, "pf_failed_probe_slot_days", np.nan),
                   "turnover_annual": getattr(r_, "pf_turnover_annual", np.nan),
                   "opportunity_cost_pct": getattr(r_, "pf_opportunity_cost_pct", np.nan),
                   "cagr": getattr(r_, "pf_cagr", np.nan), "mdd": getattr(r_, "pf_mdd", np.nan),
                   "sharpe": getattr(r_, "pf_sharpe", np.nan)})
    save(pd.DataFrame(ce), "PROBE_CAPITAL_EFFICIENCY.csv")

    # ---------------- leader capture / optionality / false positives ------------------------------
    log("leader capture / optionality / false positives")
    eps = pd.concat([AN.leader_episodes(p, L, thr) for thr in (0.2, 0.3, 0.4)], ignore_index=True)
    cap_tl = AN.capture(eps, full, p.dates, label="trade_level(unconstrained)")
    pft = pfr["FULL"]["trades"]
    cap_pf = AN.capture(eps, pft, p.dates, label="10slot_portfolio")
    cap = pd.concat([cap_tl, cap_pf], ignore_index=True)
    base = {}
    for src, camps_ in (("trade_level(unconstrained)", full), ("10slot_portfolio", pft)):
        for thr in (0.2, 0.3, 0.4):
            e_ = eps[eps["threshold"] == thr]
            base[(src, thr, "ALL")] = AN.random_capture_baseline(e_, camps_, U)
    csum = AN.capture_summary(cap, base)
    save(csum, "LEADER_CAPTURE_RATE.csv")
    epi = cap.copy()
    epi.insert(1, "name", epi["stock_id"].map(p.names))
    save(epi.sort_values(["source", "threshold", "peak_gain"], ascending=[True, True, False]), "LEADER_EPISODES.csv")
    res.update(capture=csum, episodes=epi)
    opt = AN.optionality_table(full, cap_tl)
    save(opt, "LEADER_OPTIONALITY.csv")
    res["optionality"] = opt
    fpa = AN.false_positive_analysis(sf)
    sbf = AN.strong_but_failed(R)
    fp_all = pd.concat([fpa.assign(source="V1_probes"), sbf.assign(source="top5pct_discovery_label_based")],
                       ignore_index=True)
    save(fp_all, "FALSE_POSITIVE_ANALYSIS.csv")
    res["fp"] = fp_all

    # ---------------- expanding validation (robustness only) ----------------------------------------
    log("expanding-window validation (robustness only)")
    exp_rows = []
    for tr0, tr1, te0, te1 in config.EXPANDING_WINDOWS:
        Dtr = ST.purged(R, tr0, tr1, p.dates)
        sel = ST.select_discovery_features(Dtr)
        feats = tuple(v["feature"] for v in sel["families"].values()) or cfg.disc_features
        c_x, _ = staged_selection(A, sc, feats, (tr0, tr1), version=f"EXP_{tr1[:4]}", log=lambda *a: None)
        m = campaign_metrics(run_tl(A, sc, c_x, te0, te1), years=years_of(te0, te1))
        pm = run_pf(A, sc, c_x, te0, te1)["metrics"]
        m1 = campaign_metrics(run_tl(A, sc, cfg, te0, te1), years=years_of(te0, te1))
        exp_rows.append({"train": f"{tr0}~{tr1}", "test": f"{te0}~{te1}", "selected_config": c_x.short(),
                         "selected_features": ",".join(feats), "test_n": m["n_probes"], "test_pf": m["pf"],
                         "test_payoff": m["payoff"], "test_ev": m["ev_slots_per_probe"],
                         "test_pnl_per_100_slot_days": m["pnl_per_100_slot_days"],
                         "test_portfolio_cagr": pm.get("cagr"), "test_portfolio_mdd": pm.get("mdd"),
                         "test_portfolio_sharpe": pm.get("sharpe"),
                         "frozen_V1_test_pf": m1["pf"], "frozen_V1_test_pnl_per_100_slot_days": m1["pnl_per_100_slot_days"]})
    expd = pd.DataFrame(exp_rows)
    save(expd, "EXPANDING_VALIDATION.csv")
    res["expanding"] = expd

    # ---------------- verdict evidence ------------------------------------------------------------
    lo = lifts[(lifts["family"] == "DISCOVERY_SCORE")]

    def lift_of(pn, lab):
        x = lo[(lo["period"] == pn)][f"lift_{lab}_volctl"].dropna()
        return float(x.iloc[0]) if len(x) else np.nan
    mo, md = tm["STRICT_OOS"], tm["DISCOVERY"]
    oos = tl["STRICT_OOS"]
    oos_conf = oos[oos["confirmed"]]
    po = pfr["STRICT_OOS"]["metrics"]
    stress = costs[(costs["period"] == "STRICT_OOS") & (costs["round_trip_cost"] == 0.007) &
                   (costs["slippage_bps_per_side"] == 50)].iloc[0]
    noadd_arch = "P25_ONLY" if cfg.probe_size <= 0.25 else "P50_ONLY"
    na = arch[(arch["period"] == "STRICT_OOS") & (arch["variant"] == noadd_arch)].iloc[0]
    yv = yearly[yearly["level"] == "portfolio(continuous)"].set_index("year")["return"]
    evidence = {
        "discovery": {"oos_lift20": lift_of("STRICT_OOS", "leader20"), "oos_lift30": lift_of("STRICT_OOS", "leader30"),
                      "disc_lift20": lift_of("DISCOVERY", "leader20"), "oos_lift_mae10": lift_of("STRICT_OOS", "bad10")},
        "probe": {"pf": mo.get("pf", np.nan), "ev": mo.get("ev_slots_per_probe", np.nan), "n": mo.get("n_probes", 0),
                  "placebo_p": placebo_p.get("STRICT_OOS", np.nan)},
        "failed": {"avg": mo.get("avg_failed_loss_ret", np.nan), "worst5": mo.get("failed_worst5pct_ret", np.nan),
                   "hold": mo.get("failed_holding_median", np.nan), "disc_avg": md.get("avg_failed_loss_ret", np.nan)},
        "confirm": {"add_pf": mo.get("add_leg_pf", np.nan),
                    "p20_conf": float((oos_conf["ret_on_invested"] >= 0.2).mean()) if len(oos_conf) else np.nan,
                    "p20_all": float((oos["ret_on_invested"] >= 0.2).mean()) if len(oos) else np.nan,
                    "prem_med": mo.get("add_premium_median", np.nan), "rr_med": mo.get("add_reward_risk_at_add", np.nan)},
        "add": {"pnl": mo.get("total_pnl_slots", np.nan), "pnl_noadd": na["total_pnl_slots"],
                "eff": mo.get("pnl_per_100_slot_days", np.nan), "eff_noadd": na["pnl_per_100_slot_days"],
                "pf": mo.get("pf", np.nan), "pf_noadd": na["pf"], "calmar": po.get("calmar", np.nan),
                "calmar_noadd": na.get("pf_calmar", np.nan)},
        "system": {"cagr": po.get("cagr"), "sharpe": po.get("sharpe"), "mdd": po.get("mdd"), "pf": po.get("c_pf"),
                   "cagr_stress": stress["pf_cagr"], "pf_stress": stress["pf_campaign_pf"],
                   "ex_top5": po.get("c_pnl_ex_top5_slots"), "ret_2025": yv.get("2025", np.nan),
                   "ret_2026": yv.get("2026YTD", np.nan)},
    }
    verdicts = decide(evidence)
    res["evidence"], res["verdicts"] = evidence, verdicts
    rows = [{"component": k, "verdict": v[0], "evidence": v[1], "criteria_ACCEPT": CRITERIA[k]["ACCEPT"],
             "criteria_WATCH": CRITERIA[k]["WATCH"]} for k, v in verdicts.items()]
    for k, v in verdicts.items():
        dlog.add(V, "VALIDATION", f"verdict_{k}", v[0], "", "STRICT_OOS 2025-01-01~2026-09-03", v[1], cfg.hash())
    oos_rows = [{"section": "trade_level", "metric": k, "value": v} for k, v in mo.items()]
    oos_rows += [{"section": "portfolio", "metric": k, "value": v} for k, v in po.items()]
    oos_rows += [{"section": "verdict", "metric": r["component"], "value": r["verdict"], "note": r["evidence"]}
                 for r in rows]
    save(pd.DataFrame(oos_rows), "STRICT_OOS_2025_2026.csv")
    save(pd.DataFrame(rows), "FINAL_VERDICTS.csv")

    # ---------------- dashboard data -------------------------------------------------------------
    log("dashboard data files")
    write_dashboard_data(p, F, conds, cfg, A, live_camps, pfr["LIVE"], res)
    return res


# ---------------------------------------------------------------------------
STATE_ZH = {"WATCH": "觀察", "EMERGING": "Emerging", "PROBE_READY": "可試單", "PROBED": "已試單",
            "WAIT_CONFIRM": "等待確認", "CONFIRMED": "趨勢確認", "ADD_READY": "可加碼", "FULL": "正式持股",
            "FAILED": "試單失敗", "EXIT": "出場"}


def campaign_state_rows(A, camps) -> pd.DataFrame:
    """Per (stock, date) state derived ONLY from events up to that date (PIT)."""
    rows = []
    for cp in camps:
        sid = A.ids[cp.j]
        rows.append({"date": A.dates[cp.t], "stock_id": sid, "state": "PROBE_READY", "detail": "probe 條件成立（收盤）",
                     "camp_id": f"{sid}_{A.dates[cp.t].date()}"})
        cid = f"{sid}_{A.dates[cp.t].date()}"
        for r in cp.daily:
            d = r["day"]
            st = r["state"]
            if st == "PROBE":
                s_ = "PROBED" if d - cp.entry_day <= 1 else "WAIT_CONFIRM"
            elif st == "CONFIRMED":
                s_ = "ADD_READY" if "ADD" in r.get("pending", "") else "CONFIRMED"
            elif st == "ADD":
                s_ = "ADD_READY" if "ADD2" in r.get("pending", "") else "CONFIRMED"
            elif st == "FULL":
                s_ = "FULL"
            else:
                continue
            row = {"date": A.dates[d], "stock_id": sid, "state": s_, "camp_id": cid,
                   **{k: v for k, v in r.items() if k not in ("day", "state")}}
            row["stop_adj"] = row.get("stop")
            for k in ("close", "stop", "avg_cost"):
                if k in row:
                    row[k] = cp.raw_px(d, row[k])
            rows.append(row)
        if cp.exit_day >= 0 and not cp.exit_reason.startswith("END"):
            st = "FAILED" if cp.state == "FAILED_PROBE" else "EXIT"
            for k in range(0, 5):
                dd = cp.exit_day + k
                if dd < len(A.dates):
                    rows.append({"date": A.dates[dd], "stock_id": sid, "state": st, "camp_id": cid,
                                 "detail": cp.exit_reason, "exit_price": cp.raw_px(cp.exit_day, cp.exit_px)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # one row per (stock, date): later events in the same day win
    pri = {"PROBE_READY": 0, "PROBED": 1, "WAIT_CONFIRM": 1, "CONFIRMED": 2, "ADD_READY": 2, "FULL": 3,
           "FAILED": 4, "EXIT": 4}
    df["pri"] = df["state"].map(pri)
    df = df.sort_values(["stock_id", "date", "pri"]).drop_duplicates(["stock_id", "date"], keep="last")
    return df.drop(columns="pri")


def write_dashboard_data(p, F, conds, cfg, A, live_camps, pf_live, res) -> None:
    U = p.universe
    start = pd.Timestamp(config.RESEARCH_START) - pd.Timedelta(days=30)
    mask = U.copy() | (conds["disc_pct"] >= cfg.watch_q)
    mask.loc[mask.index < start] = False
    idx = mask.stack()
    idx = idx[idx].index
    cols = {
        "disc_pct": conds["disc_pct"], "rank_score": conds["rank_score"], "probe_signal": conds["probe_signal"],
        "c_universe": conds["c_universe"], "c_no_disposition": conds["c_no_disposition"],
        "c_probe_score": conds["c_probe_score"], "c_trigger": conds["c_trigger"], "c_regime": conds["c_regime"],
        "c_stop_ok": conds["c_stop_ok"], "stop_dist": conds["stop_dist"],
        "cex_10": F["cex_10"], "srs_10": F["srs_10"], "rs_20": F["rs_20"], "rs_40": F["rs_40"], "rs_60": F["rs_60"],
        "rs_accel": F["rs_accel"], "outp_10": F["outp_10"], "atr_pct": F["atr_pct"], "val20": F["val20"],
        "val_pct": F["val_pct"], "trend_ok": F["trend_ok"], "hl_flag": F["hl_flag"], "hh_flag": F["hh_flag"],
        "dist_h20": F["dist_h20"], "dist_h60": F["dist_h60"], "ma20": F["ma20"], "ma10": F["ma10"],
        "low3": F["low3"], "rsl_high20": F["rsl_high20"], "relhigh": F["relhigh"], "is_event": F["is_event"],
        "is_cnt10": F["is_cnt10"], "dr60": F["dr60"], "up60": F["up60"], "beta": F["beta"],
        "disp": p.disp, "disp_minutes": p.disp_minutes, "vol_comparable": F["vol_comparable"],
        "close": p.raw_c, "turnover_pct": F.get("turnover_pct", F["val_pct"]),
    }
    sig = pd.DataFrame({k: v.stack(future_stack=True).reindex(idx) for k, v in cols.items()})
    sig.index.names = ["date", "stock_id"]
    sig = sig.reset_index()
    for c in sig.columns:
        if sig[c].dtype == np.float64:
            sig[c] = sig[c].astype(np.float32)
    sig.to_parquet(SIG_DIR / "signals_daily.parquet", index=False)
    states = campaign_state_rows(A, live_camps)
    states.to_parquet(SIG_DIR / "states_daily.parquet", index=False)
    camps = pd.DataFrame([c.record() for c in live_camps if c.legs])
    camps.to_parquet(SIG_DIR / "campaigns_live.parquet", index=False)
    ev = []
    for c in live_camps:
        for (d, e, px, sz, why) in c.events:
            ev.append({"camp_id": f"{A.ids[c.j]}_{A.dates[c.t].date()}", "stock_id": A.ids[c.j], "date": A.dates[d],
                       "event": e, "price": c.raw_px(d, px), "price_adj": px, "size": sz, "reason": why})
    pd.DataFrame(ev).to_parquet(SIG_DIR / "campaign_events.parquet", index=False)
    # prices for charts (adjusted OHLC + raw close)
    px = pd.DataFrame({"open": p.o.stack(), "high": p.h.stack(), "low": p.l.stack(), "close": p.c.stack()})
    px["raw_close"] = p.raw_c.stack().reindex(px.index)
    px["raw_open"] = p.raw_o.stack().reindex(px.index)
    fac = (p.raw_c / p.c)
    px["raw_high"] = (p.h * fac).stack().reindex(px.index)
    px["raw_low"] = (p.l * fac).stack().reindex(px.index)
    px["value"] = p.val.stack().reindex(px.index)
    px.index.names = ["date", "stock_id"]
    px = px.reset_index()
    px = px[px["date"] >= start - pd.Timedelta(days=120)]
    for c in ("open", "high", "low", "close", "raw_close", "raw_open", "raw_high", "raw_low", "value"):
        px[c] = px[c].astype(np.float32)
    px.to_parquet(SIG_DIR / "prices.parquet", index=False)
    mk = F["mkt"].copy()
    mk["close"] = p.market["close"]
    mk["open"] = p.market["open"]
    mk["high"] = p.market["high"]
    mk["low"] = p.market["low"]
    mk.index.name = "date"
    mk.reset_index().to_parquet(SIG_DIR / "market.parquet", index=False)
    sec = F["sec_level"].stack().rename("sec_level").reset_index()
    sec.columns = ["date", "stock_id", "sec_level"]
    sec = sec[sec["date"] >= start - pd.Timedelta(days=120)]
    sec.to_parquet(SIG_DIR / "sector_level.parquet", index=False)
    pf_live["equity"].reset_index().to_parquet(SIG_DIR / "portfolio_live_equity.parquet", index=False)
    pf_live.get("positions", pd.DataFrame()).to_parquet(SIG_DIR / "portfolio_live_positions.parquet", index=False)
    pf_live["trades"].to_parquet(SIG_DIR / "portfolio_live_trades.parquet", index=False)
    info = pd.DataFrame({"stock_id": p.ids, "name": p.names.reindex(p.ids).to_numpy(),
                         "sector": p.sector.reindex(p.ids).to_numpy(), "type": p.stock_type.reindex(p.ids).to_numpy()})
    info.to_parquet(SIG_DIR / "stock_info.parquet", index=False)
    dtab = p.meta.get("disposition_table")
    if dtab is not None:
        dtab.to_parquet(SIG_DIR / "disposition.parquet", index=False)
    from data.finmind_client import FinMindCache
    man = FinMindCache(client=None).manifest
    meta = {"data_source": config.DATA_SOURCE, "last_data_date": str(p.dates[-1].date()),
            "finmind_last_refresh_utc": man.get("last_refresh_utc"), "signals_built_utc": now_utc(),
            "config": cfg.to_dict(), "config_hash": cfg.hash(), "version": cfg.version,
            "market_source": p.meta.get("market_source")}
    (SIG_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
def load_frozen(version: str) -> tuple[StrategyConfig, dict] | tuple[None, None]:
    f = FROZEN / f"{version}.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text(encoding="utf-8"))
    return StrategyConfig.from_dict(d["config"]), d


def prepare(log_=log):
    t0 = time.time()
    p = build_panel(log=log_)
    F = compute_features(p, log=log_)
    L = compute_labels(p)
    R = research_frame(p, F, L)
    log_(f"prepared panel/features/labels/research frame in {time.time() - t0:.0f}s; R rows={len(R):,}")
    return p, F, L, R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="V1")
    ap.add_argument("--refreeze", default=None, help="create a NEW version label (e.g. V2) by re-running Discovery")
    ap.add_argument("--skip-case", action="store_true")
    ap.add_argument("--note", default=None, help="reason for a validation re-run (logged)")
    a = ap.parse_args(argv)
    p, F, L, R = prepare()
    A = build_arrays(p, F)
    sc = SignalCache(p, F)
    dlog = DecisionLog()
    version = a.refreeze or a.version
    cfg, frozen = load_frozen(version)
    if cfg is None:
        cfg = discovery_phase(p, F, R, A, sc, version, dlog)
    else:
        log(f"using FROZEN {version} (frozen {frozen['frozen_utc']}, hash {frozen['hash']}) — not re-selected")
        dlog.rows.extend(frozen.get("decisions", []))
        dlog.add(version, "VALIDATION", "load_frozen", "frozen config reused unchanged", cfg.short(), "",
                 f"frozen_utc={frozen['frozen_utc']}", frozen["hash"])
        if a.note:
            dlog.add(version, "VALIDATION", "rerun_note", a.note, "", "", "rules/thresholds unchanged", frozen["hash"])
    hist_f = FROZEN / f"{version}_runs.jsonl"
    if hist_f.exists():                       # every earlier validation run of this frozen version
        for line in hist_f.read_text(encoding="utf-8").splitlines():
            h = json.loads(line)
            dlog.add(version, "VALIDATION_HISTORY", "validation_run", h.get("note", ""),
                     json.dumps(h.get("verdicts", {}), ensure_ascii=False), "", h.get("source", ""),
                     h.get("config_hash", ""), ts=h.get("timestamp_utc"))
    run_ts = now_utc()
    res = validation_phase(p, F, L, R, A, sc, cfg, dlog)
    with hist_f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp_utc": run_ts, "note": a.note or "validation run", "config_hash": cfg.hash(),
                             "verdicts": {k: v[0] for k, v in res["verdicts"].items()}, "source": "pipeline"},
                            ensure_ascii=False) + "\n")
    if not a.skip_case:
        from research.case_study import run_case
        res["case"] = run_case(p, F, cfg, A, sc, res)
    from pipeline.reports import write_reports
    write_reports(p, F, cfg, res)
    dlog.save()
    log("ALL DONE")
    return res


if __name__ == "__main__":
    main()
