"""Cross-check: spread-based total return (engine.panel) vs FinMind TaiwanStockPriceAdj.

    python -m data.audit_adjust      -> outputs/ADJUSTMENT_CHECK.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from data.finmind_client import FinMindCache, FinMindClient  # noqa: E402
from engine.panel import build_panel  # noqa: E402

SAMPLE = ["2330", "2317", "2454", "3653", "2881", "1101", "2603", "3008", "6669", "3231", "8299", "5274", "6488",
          "1795", "2002"]


def main():
    cache = FinMindCache(FinMindClient())
    p = build_panel(log=lambda *a: None)
    rows = []
    for sid in SAMPLE:
        if sid not in p.ids:
            continue
        adj = cache.series("TaiwanStockPriceAdj", sid, "2023-01-01")
        if adj.empty:
            continue
        a = adj.set_index(pd.to_datetime(adj["date"]))["close"].astype(float)
        ra = a.pct_change()
        rm = p.r[sid].reindex(ra.index)
        both = ra.notna() & rm.notna()
        diff = (ra[both] - rm[both]).abs()
        big = diff > 0.002
        cum_a = a.iloc[-1] / a.iloc[0] - 1
        cm = p.c[sid].reindex(a.index).dropna()
        rows.append({"stock_id": sid, "days": int(both.sum()), "corr": float(np.corrcoef(ra[both], rm[both])[0, 1]),
                     "mean_abs_diff_bp": float(diff.mean() * 1e4), "days_diff_gt_20bp": int(big.sum()),
                     "max_abs_diff": float(diff.max()), "worst_day": str(diff.idxmax().date()),
                     "cum_ret_finmind_adj": cum_a, "cum_ret_spread_method": float(cm.iloc[-1] / cm.iloc[0] - 1)})
    cache.save_manifest()
    df = pd.DataFrame(rows)
    df.to_csv(config.OUT_DIR / "ADJUSTMENT_CHECK.csv", index=False, encoding="utf-8-sig")
    print(df.to_string())


if __name__ == "__main__":
    main()
