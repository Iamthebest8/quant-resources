"""PRE-REGISTERED decision criteria (written before any Strict-OOS result was computed).

These thresholds are part of the frozen V1 research protocol. Changing them after looking
at 2025-2026 results requires a new version label (V2, V3, ...) — see STRATEGY_DECISION_LOG.csv.
The final ACCEPT / WATCH / REJECT is driven by STRICT OOS (2025-01-01 ~ 2026-09-03).
"""
from __future__ import annotations

import numpy as np

CRITERIA = {
    "DISCOVERY": {
        "ACCEPT": "OOS vol-controlled lift(leader20) >= 1.5 AND lift(leader30) >= 1.5 AND OOS lift >= 0.7 x Discovery lift "
                  "AND lift(MAE<=-10%) < lift(leader20)",
        "WATCH": "OOS vol-controlled lift(leader20) >= 1.2",
    },
    "PROBE": {
        "ACCEPT": "OOS trade-level PF >= 1.3 AND EV/probe > 0 AND >= 50 probes AND EV beats matched-control placebo "
                  "(bootstrap one-sided p < 0.10)",
        "WATCH": "OOS PF >= 1.0 AND EV/probe > 0",
    },
    "FAILED_PROBE_EXIT": {
        "ACCEPT": "OOS avg failed-probe return >= -6% AND worst-5% >= -12% AND median failed holding <= 10D "
                  "AND OOS avg failed loss not worse than 1.5 x Discovery",
        "WATCH": "OOS avg failed-probe return >= -8%",
    },
    "CONFIRMATION": {
        "ACCEPT": "OOS add-leg PF >= 1.3 AND P(ret>=20% | confirmed) >= 2 x P(ret>=20% | all probes) "
                  "AND median add premium <= 10% AND median add reward/risk >= 1.5",
        "WATCH": "OOS add-leg PF >= 1.0",
    },
    "ADD": {
        "ACCEPT": "OOS with-add vs probe-only (same probe size): total PnL higher AND PnL/100 slot-days higher "
                  "AND PF not more than 10% lower AND portfolio Calmar not lower",
        "WATCH": "OOS with-add total PnL higher",
    },
    "FULL_SYSTEM": {
        "ACCEPT": "OOS 10-slot portfolio @0.45%+25bps: CAGR > 0, Sharpe >= 0.8, MDD >= -25%, campaign PF >= 1.3; "
                  "@0.70%+50bps: CAGR > 0 and PF >= 1.1; PnL ex-top-5 campaigns > 0; 2025 and 2026YTD both > 0; "
                  "PROBE verdict not REJECT",
        "WATCH": "OOS portfolio CAGR > 0 and campaign PF >= 1.1",
    },
}


def _ok(x, cond):
    try:
        return bool(np.isfinite(x) and cond(x))
    except TypeError:
        return False


def decide(ev: dict) -> dict:
    """ev: evidence dict assembled by the validation phase. Returns {component: (verdict, reasons)}."""
    out = {}
    # DISCOVERY
    d = ev.get("discovery", {})
    l20, l30, ld, lm = d.get("oos_lift20"), d.get("oos_lift30"), d.get("disc_lift20"), d.get("oos_lift_mae10")
    if (_ok(l20, lambda x: x >= 1.5) and _ok(l30, lambda x: x >= 1.5) and _ok(ld, lambda x: True)
            and l20 >= 0.7 * ld and _ok(lm, lambda x: x < l20)):
        v = "ACCEPT"
    elif _ok(l20, lambda x: x >= 1.2):
        v = "WATCH"
    else:
        v = "REJECT"
    out["DISCOVERY"] = (v, f"OOS lift20={l20:.2f} lift30={l30:.2f} (Discovery lift20={ld:.2f}) MAE10 lift={lm:.2f}"
                        if all(isinstance(x, float) for x in (l20, l30, ld, lm)) else str(d))
    # PROBE
    p = ev.get("probe", {})
    pf, evp, n, pv = p.get("pf", np.nan), p.get("ev", np.nan), p.get("n", 0), p.get("placebo_p", np.nan)
    if _ok(pf, lambda x: x >= 1.3) and _ok(evp, lambda x: x > 0) and n >= 50 and _ok(pv, lambda x: x < 0.10):
        v = "ACCEPT"
    elif _ok(pf, lambda x: x >= 1.0) and _ok(evp, lambda x: x > 0):
        v = "WATCH"
    else:
        v = "REJECT"
    out["PROBE"] = (v, f"OOS PF={pf:.2f} EV/probe={evp:.4f} slots n={n} placebo p={pv:.3f}")
    # FAILED PROBE EXIT
    f = ev.get("failed", {})
    al, w5, hd, ald = f.get("avg", np.nan), f.get("worst5", np.nan), f.get("hold", np.nan), f.get("disc_avg", np.nan)
    if (_ok(al, lambda x: x >= -0.06) and _ok(w5, lambda x: x >= -0.12) and _ok(hd, lambda x: x <= 10)
            and _ok(ald, lambda x: True) and al >= 1.5 * ald):
        v = "ACCEPT"
    elif _ok(al, lambda x: x >= -0.08):
        v = "WATCH"
    else:
        v = "REJECT"
    out["FAILED_PROBE_EXIT"] = (v, f"OOS avg={al:.2%} worst5%={w5:.2%} median hold={hd} (Discovery avg={ald:.2%})")
    # CONFIRMATION
    c = ev.get("confirm", {})
    apf, r_c, r_a, prem, rr = (c.get("add_pf", np.nan), c.get("p20_conf", np.nan), c.get("p20_all", np.nan),
                               c.get("prem_med", np.nan), c.get("rr_med", np.nan))
    if (_ok(apf, lambda x: x >= 1.3) and _ok(r_c, lambda x: True) and _ok(r_a, lambda x: x > 0) and r_c >= 2 * r_a
            and _ok(prem, lambda x: x <= 0.10) and _ok(rr, lambda x: x >= 1.5)):
        v = "ACCEPT"
    elif _ok(apf, lambda x: x >= 1.0):
        v = "WATCH"
    else:
        v = "REJECT"
    out["CONFIRMATION"] = (v, f"OOS add-leg PF={apf:.2f} P(>=20%|conf)={r_c:.2%} vs all={r_a:.2%} "
                              f"add premium med={prem:.2%} R/R med={rr:.2f}")
    # ADD
    a = ev.get("add", {})
    tp, tp0 = a.get("pnl", np.nan), a.get("pnl_noadd", np.nan)
    ef, ef0 = a.get("eff", np.nan), a.get("eff_noadd", np.nan)
    pf1, pf0 = a.get("pf", np.nan), a.get("pf_noadd", np.nan)
    cal, cal0 = a.get("calmar", np.nan), a.get("calmar_noadd", np.nan)
    if (all(np.isfinite(x) for x in (tp, tp0, ef, ef0, pf1, pf0, cal, cal0)) and tp > tp0 and ef > ef0
            and pf1 >= 0.9 * pf0 and cal >= cal0):
        v = "ACCEPT"
    elif np.isfinite(tp) and np.isfinite(tp0) and tp > tp0:
        v = "WATCH"
    else:
        v = "REJECT"
    out["ADD"] = (v, f"OOS PnL with add={tp:.2f} vs probe-only={tp0:.2f} slots; eff {ef:.3f} vs {ef0:.3f}; "
                     f"PF {pf1:.2f} vs {pf0:.2f}; Calmar {cal:.2f} vs {cal0:.2f}")
    # FULL SYSTEM
    s = ev.get("system", {})
    conds = [
        _ok(s.get("cagr"), lambda x: x > 0), _ok(s.get("sharpe"), lambda x: x >= 0.8),
        _ok(s.get("mdd"), lambda x: x >= -0.25), _ok(s.get("pf"), lambda x: x >= 1.3),
        _ok(s.get("cagr_stress"), lambda x: x > 0), _ok(s.get("pf_stress"), lambda x: x >= 1.1),
        _ok(s.get("ex_top5"), lambda x: x > 0), _ok(s.get("ret_2025"), lambda x: x > 0),
        _ok(s.get("ret_2026"), lambda x: x > 0), out["PROBE"][0] != "REJECT",
    ]
    if all(conds):
        v = "ACCEPT"
    elif _ok(s.get("cagr"), lambda x: x > 0) and _ok(s.get("pf"), lambda x: x >= 1.1):
        v = "WATCH"
    else:
        v = "REJECT"
    names = ["CAGR>0", "Sharpe>=0.8", "MDD>=-25%", "PF>=1.3", "stress CAGR>0", "stress PF>=1.1", "ex-top5>0",
             "2025>0", "2026YTD>0", "PROBE!=REJECT"]
    out["FULL_SYSTEM"] = (v, "; ".join(f"{n}:{'✔' if c_ else '✘'}" for n, c_ in zip(names, conds)))
    return out
