"""Global configuration for the EMERGING LEADER PROBE STRATEGY project.

All paths, research periods and data-source switches live here so that every
module (download -> PIT dataset -> research -> strategy -> portfolio -> dashboard)
uses exactly the same definitions.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data source switch.
#   finmind   : real data downloaded from FinMind into data/cache (default)
#   synthetic : offline, FinMind-schema synthetic data used ONLY to test code
#               paths. Never use synthetic results for any trading decision.
# ---------------------------------------------------------------------------
DATA_SOURCE = os.getenv("EL_DATA_SOURCE", "finmind").lower()
if DATA_SOURCE not in ("finmind", "synthetic"):
    raise RuntimeError(f"EL_DATA_SOURCE must be finmind or synthetic, got {DATA_SOURCE}")
IS_SYNTHETIC = DATA_SOURCE == "synthetic"

_suffix = "_synthetic" if IS_SYNTHETIC else ""
CACHE_DIR = ROOT / "data" / f"cache{_suffix}"          # raw FinMind cache
RAW_DIR = CACHE_DIR / "raw"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
PIT_DIR = ROOT / "data" / f"pit{_suffix}"              # clean PIT research dataset
OUT_DIR = ROOT / ("outputs_synthetic" if IS_SYNTHETIC else "outputs")  # CSV / signal files
DOC_DIR = ROOT if not IS_SYNTHETIC else OUT_DIR       # Markdown reports
LOG_DIR = ROOT / "logs"

for _p in (CACHE_DIR, RAW_DIR, PIT_DIR, OUT_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Periods (Parts 36-41)
# ---------------------------------------------------------------------------
DATA_START = "2021-06-01"          # warm-up for RS120 / 250D highs / beta
RESEARCH_START = "2023-01-01"
DISCOVERY = ("2023-01-01", "2024-12-31")
EXTENDED_VALIDATION = ("2024-01-01", "2026-09-03")
STRICT_OOS = ("2025-01-01", "2026-09-03")
BACKTEST_END = "2026-09-03"
YEARS = {
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026YTD": ("2026-01-01", "2026-09-03"),
}
EXPANDING_WINDOWS = [
    # (train_start, train_end, test_start, test_end)
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2023-01-01", "2025-12-31", "2026-01-01", "2026-09-03"),
]
CASE_SYMBOL = "3653"
CASE_PERIOD = ("2026-07-01", "2026-09-30")

# Forward label horizon (trading days) used for research labels + purge/embargo
LABEL_HORIZON = 40

# ---------------------------------------------------------------------------
# Universe (common stocks only; ETFs / warrants / TDR excluded)
# ---------------------------------------------------------------------------
MIN_PRICE = 10.0                    # TWD
MIN_AVG_VALUE_20D = 30_000_000      # 20D average traded value, TWD
MIN_LISTED_DAYS = 120               # trading days of history (avoids IPO no-limit window)
PRICE_LIMIT = 0.10                  # daily price limit in TW market

# Portfolio
INITIAL_CAPITAL = 10_000_000
N_SLOTS = 10                        # 1 normal position = 10% of equity
MAX_PARTICIPATION = 0.10            # max fraction of a day's traded value per fill

# Costs (Part 42). Round-trip cost is split half on buy, half on sell.
COST_GRID = [0.0030, 0.0045, 0.0070, 0.0100]
SLIPPAGE_GRID_BPS = [0, 25, 50]     # per side, adverse
BASE_COST = 0.0045
BASE_SLIPPAGE_BPS = 25
