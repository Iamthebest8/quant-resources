"""PRE-REGISTERED decision criteria for research phase 2 (Weinstein / High R/R / 1-second execution).

Written before any Weinstein / High-R/R / intraday result for 2025-2026 was computed. All verdicts are driven by
STRICT OOS (2025-01-01 ~ 2026-09-03) for daily engines and by the chronological TEST split (same dates) for the
1-second engine. Base cost 0.45% round trip + 25 bps slippage per side; shorts additionally pay 0.08% fee and
2%/yr borrow. Changing any threshold after seeing OOS requires a new version label.
"""
from __future__ import annotations

import numpy as np

CRITERIA2 = {
    "MOMENTUM_LONG": {
        "ACCEPT": "frozen MOMENTUM_LONG_BASELINE meets the phase-1 FULL_SYSTEM criteria in Strict OOS",
        "WATCH": "OOS trade-level PF >= 1.0 and EV > 0",
    },
    "WEINSTEIN_ENTRY": {   # applied separately to W1 (stage 1->2), W2 (continuation), W3 (pullback)
        "ACCEPT": "OOS n >= 40, PF >= 1.5, payoff >= 2.5, EV > 0, EV beats matched non-signal controls "
                  "(bootstrap one-sided p < 0.10), EV > 0 in both 2025 and 2026YTD",
        "WATCH": "OOS PF >= 1.15 and EV > 0",
    },
    "HIGH_RR_FILTER": {
        "ACCEPT": "OOS: frozen High-R/R top tercile vs all events of the same engine: higher EV AND higher payoff AND "
                  "higher P(ret>=20%), and filtered PF >= 1.5",
        "WATCH": "OOS filtered EV higher than unfiltered",
    },
    "ONE_SECOND_EXECUTION": {
        "ACCEPT": "TEST split: frozen LEARNED trigger beats OPEN and BREAKOUT_IMMEDIATE on mean utility "
                  "(MFE-|MAE|, EOD) with bootstrap p < 0.10 AND the multi-day trade EV with learned entry >= EV "
                  "with the daily entry",
        "WATCH": "TEST utility of LEARNED > OPEN",
    },
    "WEINSTEIN_SHORT": {
        "ACCEPT": "OOS n >= 30, PF >= 1.3, EV > 0 after short costs, AND short executability verified "
                  "(not available -> capped at WATCH)",
        "WATCH": "OOS PF >= 1.0 and EV > 0",
    },
    "BOLLINGER_EXIT": {
        "ACCEPT": "OOS on the same frozen entries: frozen Bollinger variant beats the textbook exit on "
                  "PnL/100 slot-days AND PF, while keeping >= 75% of the textbook exit's >=30% winners",
        "WATCH": "OOS PF higher than textbook exit",
    },
    "HYBRID": {
        "ACCEPT": "OOS frozen hybrid beats BOTH its component engines on PF and EV, and its 10-slot "
                  "portfolio Calmar >= the better component's",
        "WATCH": "OOS hybrid beats at least one component on PF and EV",
    },
    "MULTI_ALPHA_PORTFOLIO": {
        "ACCEPT": "OOS 10-slot: CAGR > 0, Sharpe >= 0.8, MDD >= -25%, CAGR > 0 at 0.70%+50bps, "
                  "2025 > 0 and 2026YTD > 0",
        "WATCH": "OOS CAGR > 0 and Sharpe >= 0.4",
    },
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def entry_verdict(ev: dict) -> tuple[str, str]:
    n, pf, pay, e, p, y25, y26 = (ev.get(k) for k in ("n", "pf", "payoff", "ev", "placebo_p", "ev_2025", "ev_2026"))
    n, pf, pay, e, p, y25, y26 = (_f(x) for x in (n, pf, pay, e, p, y25, y26))
    if n >= 40 and pf >= 1.5 and pay >= 2.5 and e > 0 and p < 0.10 and y25 > 0 and y26 > 0:
        v = "ACCEPT"
    elif pf >= 1.15 and e > 0:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"n={n:.0f} PF={pf:.2f} payoff={pay:.2f} EV={e:.4f} placebo p={p:.3f} EV2025={y25:.4f} EV2026={y26:.4f}"


def highrr_verdict(ev: dict) -> tuple[str, str]:
    e1, e0, p1, p0, g1, g0, pf1 = (_f(ev.get(k)) for k in ("ev_f", "ev_all", "pay_f", "pay_all", "ge20_f", "ge20_all", "pf_f"))
    if e1 > e0 and p1 > p0 and g1 > g0 and pf1 >= 1.5:
        v = "ACCEPT"
    elif e1 > e0:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"EV {e1:.4f} vs {e0:.4f}; payoff {p1:.2f} vs {p0:.2f}; P(>=20%) {g1:.3f} vs {g0:.3f}; PF {pf1:.2f}"


def exec_verdict(ev: dict) -> tuple[str, str]:
    u, uo, ub, p, tr, td = (_f(ev.get(k)) for k in ("u_learned", "u_open", "u_breakout", "p", "trade_ev_learned", "trade_ev_daily"))
    if u > uo and u > ub and p < 0.10 and tr >= td:
        v = "ACCEPT"
    elif u > uo:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"utility learned {u:.4f} vs open {uo:.4f} vs breakout {ub:.4f} (p={p:.3f}); trade EV {tr:.4f} vs {td:.4f}"


def short_verdict(ev: dict, executability_verified: bool = False) -> tuple[str, str]:
    n, pf, e = (_f(ev.get(k)) for k in ("n", "pf", "ev"))
    if n >= 30 and pf >= 1.3 and e > 0:
        v = "ACCEPT" if executability_verified else "WATCH"
    elif pf >= 1.0 and e > 0:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"n={n:.0f} PF={pf:.2f} EV={e:.4f}; executability verified={executability_verified}"


def bb_verdict(ev: dict) -> tuple[str, str]:
    eff, eff0, pf, pf0, w30, w30_0 = (_f(ev.get(k)) for k in ("eff", "eff_text", "pf", "pf_text", "n30", "n30_text"))
    keep = w30 >= 0.75 * w30_0 if w30_0 > 0 else True
    if eff > eff0 and pf > pf0 and keep:
        v = "ACCEPT"
    elif pf > pf0:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"PnL/100sd {eff:.3f} vs textbook {eff0:.3f}; PF {pf:.2f} vs {pf0:.2f}; >=30% winners {w30:.0f} vs {w30_0:.0f}"


def hybrid_verdict(ev: dict) -> tuple[str, str]:
    pf, e, cal = (_f(ev.get(k)) for k in ("pf", "ev", "calmar"))
    pfs, es, cals = ev.get("pf_comp", []), ev.get("ev_comp", []), ev.get("calmar_comp", [])
    beats_all = all(pf > a and e > b for a, b in zip(pfs, es))
    beats_one = any(pf > a and e > b for a, b in zip(pfs, es))
    if beats_all and cal >= max(cals or [-np.inf]):
        v = "ACCEPT"
    elif beats_one:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"hybrid PF {pf:.2f} EV {e:.4f} Calmar {cal:.2f}; components PF {pfs} EV {es} Calmar {cals}"


def portfolio_verdict(ev: dict) -> tuple[str, str]:
    cagr, sh, mdd, cs, y25, y26 = (_f(ev.get(k)) for k in ("cagr", "sharpe", "mdd", "cagr_stress", "ret_2025", "ret_2026"))
    if cagr > 0 and sh >= 0.8 and mdd >= -0.25 and cs > 0 and y25 > 0 and y26 > 0:
        v = "ACCEPT"
    elif cagr > 0 and sh >= 0.4:
        v = "WATCH"
    else:
        v = "REJECT"
    return v, f"CAGR {cagr:.3f} Sharpe {sh:.2f} MDD {mdd:.3f} stress CAGR {cs:.3f} 2025 {y25:.3f} 2026 {y26:.3f}"
