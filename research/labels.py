"""Forward-looking labels. RESEARCH ONLY — never imported by strategy/ signal code.

Entry convention: signal at close t -> fill at open t+1 (adjusted open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from engine.features import FEATURE_FAMILIES, cs_pct
from engine.panel import Panel

HORIZONS = (5, 10, 20, 40)


def _fwd_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """max over rows t+1..t+n"""
    return x[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)


def _fwd_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x[::-1].rolling(n, min_periods=1).min()[::-1].shift(-1)


def compute_labels(p: Panel) -> dict[str, pd.DataFrame]:
    L: dict[str, pd.DataFrame] = {}
    c = p.c.ffill(limit=config.LABEL_HORIZON)       # delisted / suspended -> last price
    lo = p.l.fillna(c)
    entry = p.o.shift(-1)
    L["entry"] = entry
    T = len(p.dates)
    pos = pd.Series(np.arange(T), index=p.dates)
    m = p.market["adj_close"]
    m_entry = p.market["adj_open"].shift(-1)
    for n in HORIZONS:
        valid = pd.DataFrame(np.repeat((pos + n < T).to_numpy()[:, None], len(p.ids), axis=1),
                             index=p.dates, columns=p.ids) & entry.notna()
        L[f"fwd_ret_{n}"] = (c.shift(-n) / entry - 1).where(valid)
        L[f"fwd_maxc_{n}"] = (_fwd_max(c, n) / entry - 1).where(valid)
        L[f"fwd_mae_{n}"] = (_fwd_min(lo, n) / entry - 1).where(valid)
        mret = (m.shift(-n) / m_entry - 1)
        L[f"fwd_ex_{n}"] = L[f"fwd_ret_{n}"].sub(mret, axis=0)
    for thr in (20, 30, 40):
        L[f"leader{thr}"] = (L["fwd_maxc_40"] >= thr / 100).astype(float).where(L["fwd_maxc_40"].notna())
        L[f"leader{thr}_20d"] = (L["fwd_maxc_20"] >= thr / 100).astype(float).where(L["fwd_maxc_20"].notna())
    # "clean" leader: +20% max gain before a -10% drawdown is ever hit within 40D
    L["bad10"] = (L["fwd_mae_40"] <= -0.10).astype(float).where(L["fwd_mae_40"].notna())
    return L


def period_label(d: pd.Series) -> pd.Series:
    out = pd.Series("PRE", index=d.index)
    out[(d >= config.DISCOVERY[0]) & (d <= config.DISCOVERY[1])] = "DISCOVERY"
    out[(d >= config.STRICT_OOS[0]) & (d <= config.STRICT_OOS[1])] = "STRICT_OOS"
    out[d > config.STRICT_OOS[1]] = "POST"
    return out


def research_frame(p: Panel, F: dict, L: dict, start: str = config.RESEARCH_START) -> pd.DataFrame:
    """Long frame of universe stock-days with candidate features, controls, labels."""
    U = p.universe
    feats = sorted({f for fam in FEATURE_FAMILIES.values() for f in fam})
    extra = ["beta", "atr_pct", "hv20", "val20", "ret_60", "ext_ma20", "ret_20", "cs_rank20", "secmkt_10",
             "is_event", "ex_1", "clv", "rs_20", "rs_40", "rs_60", "rs_120", "rs_10", "trend_ok", "relhigh",
             "rsl_high20", "px_high20", "gap", "mkt_not_up", "cap_vel", "srs_20"]
    cols = {}
    mask = U.copy()
    mask.loc[mask.index < pd.Timestamp(start)] = False
    idx = mask.stack()
    idx = idx[idx]
    for f in dict.fromkeys(feats + extra):
        if f == "gap":
            x = p.raw_o / (p.raw_c.shift(1)) - 1
        elif f in F:
            x = F[f]
        else:
            continue
        cols[f] = x.stack(future_stack=True).reindex(idx.index).astype("float32")
        if f"pct_{f}" in F:
            cols[f"pct_{f}"] = F[f"pct_{f}"].stack(future_stack=True).reindex(idx.index).astype("float32")
    for k, x in L.items():
        if k == "entry":
            continue
        cols[k] = x.stack(future_stack=True).reindex(idx.index).astype("float32")
    # strata controls (cross-sectional terciles within universe)
    for f, name in (("atr_pct", "atr_t3"), ("beta", "beta_t3"), ("val20", "liq_t3")):
        pr = cs_pct(F[f], U).stack(future_stack=True).reindex(idx.index)
        cols[name] = np.ceil(pr.clip(1e-9, 1) * 3).astype("float32")
    R = pd.DataFrame(cols)
    R.index.names = ["date", "stock_id"]
    R = R.reset_index()
    mk = F["mkt"]
    R["regime"] = R["date"].map(mk["regime"])
    R["m_day"] = R["date"].map(mk["m_day"])
    R["m_ret"] = R["date"].map(mk["m_ret"]).astype("float32")
    R["sector"] = R["stock_id"].map(p.sector)
    R["year"] = R["date"].dt.year
    R["period"] = period_label(R["date"])
    return R
