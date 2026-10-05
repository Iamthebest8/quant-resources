"""PIT / look-ahead test: features and probe signals at date d computed on data truncated at d
must equal those computed on the full dataset.

    EL_DATA_SOURCE=synthetic python -m pytest -q tests/test_pit.py     (fast, offline)
    python -m pytest -q tests/test_pit.py                              (real cache)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.features import compute_features  # noqa: E402
from engine.panel import build_panel  # noqa: E402
from strategy.config import StrategyConfig  # noqa: E402
from strategy.signals import condition_frames  # noqa: E402

KEYS = ["cex_10", "srs_10", "rs_accel", "outp_10", "dr60", "up60", "is_cnt10", "atr_pct", "beta", "rs_pct_20",
        "dist_h60", "relhigh_cnt20", "low3", "ma20", "pct_cex_10", "val_pct"]
CFG = StrategyConfig(disc_features=("cex_10", "srs_10", "rs_accel"))


def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    same_nan = np.isnan(a) == np.isnan(b)
    return same_nan.all() and np.allclose(a[both], b[both], rtol=1e-5, atol=1e-7)


def test_features_are_point_in_time():
    p_full = build_panel(log=lambda *a: None)
    F_full = compute_features(p_full, log=lambda *a: None)
    c_full = condition_frames(p_full, F_full, CFG)
    for d in (p_full.dates[-260], p_full.dates[-40], p_full.dates[-1]):
        p_cut = build_panel(end=str(d.date()), log=lambda *a: None)
        F_cut = compute_features(p_cut, log=lambda *a: None)
        c_cut = condition_frames(p_cut, F_cut, CFG)
        assert p_cut.dates[-1] == d
        assert (p_cut.universe.loc[d] == p_full.universe.loc[d, p_cut.ids]).all(), "universe leaks"
        for k in KEYS:
            assert _close(F_cut[k].loc[d], F_full[k].loc[d, p_cut.ids]), f"{k} differs at {d.date()}"
        for k in ("disc_pct", "rank_score"):
            assert _close(c_cut[k].loc[d], c_full[k].loc[d, p_cut.ids]), f"{k} differs at {d.date()}"
        assert (c_cut["probe_signal"].loc[d] == c_full["probe_signal"].loc[d, p_cut.ids]).all(), "probe leaks"
        assert (p_cut.disp_next.loc[d] == p_full.disp_next.loc[d, p_cut.ids]).all(), "disposition leaks"
        assert F_cut["mkt"].loc[d, "regime"] == F_full["mkt"].loc[d, "regime"]
