"""Strategy configuration (versioned; V1 is frozen after Discovery selection)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

ARCH_SIZES = {
    # architecture -> (probe size, sizes after each add) in normal-position units (1.0 = 1 slot = 10% equity)
    "A": (0.25, (1.0,)),
    "B": (0.50, (1.0,)),
    "C": (0.25, (0.5, 1.0)),
    "P25_ONLY": (0.25, ()),   # diagnostics: probe but never add
    "P50_ONLY": (0.50, ()),
}


@dataclass(frozen=True)
class StrategyConfig:
    version: str = "V1"
    # --- Discovery (Part 21) ---
    disc_features: tuple = ()            # features (pct_ columns) averaged into the Discovery score
    watch_q: float = 0.80                # 觀察
    disc_q: float = 0.90                 # Emerging (Discovery)
    # --- Probe (Part 22) ---
    probe_q: float = 0.90                # Discovery-score percentile needed for a probe
    trigger: str = "rs_high"             # rs_high | price_high | indep
    regime_filter: str = "all"           # all | not_up
    max_stop_dist: float = 0.08          # reasonable stop distance (signal close -> probe low)
    max_gap: float = 0.06                # do not chase an open > +6% above signal close
    # --- Failed probe (Part 24) ---
    fail: str = "F1+F2+F3"               # F1 probe low, F2 structure, F3 RS failure, F4 time stop
    time_stop: int = 5
    f3_thresh: float = -0.04
    probe_max_life: int = 20
    # --- Confirmation (Part 26/27) ---
    confirm: str = "price+rs"            # price | rs | persist | price+rs | price+persist (trend = diagnostic)
    min_days_confirm: int = 2
    # --- Add (Part 28) ---
    add_arch: str = "A"
    # --- Exit (Part 30) ---
    exit: str = "TF"                     # HS hard stop | TF trend failure | MS MA/structure | MP MFE protection
    mfe_trigger: float = 0.15
    cooldown: int = 5
    # --- Portfolio (Part 43-45) ---
    max_probes: int = 3
    capital_arch: str = "A"              # A shared 10 slots | B reserve probe bucket
    probe_reserve: float = 1.0           # slot-equivalents reserved for probes in architecture B
    notes: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def probe_size(self) -> float:
        return ARCH_SIZES[self.add_arch][0]

    @property
    def add_targets(self) -> tuple:
        return ARCH_SIZES[self.add_arch][1]

    def with_(self, **kw) -> "StrategyConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["disc_features"] = list(self.disc_features)
        return d

    def short(self) -> str:
        return (f"q{self.probe_q:.2f}|{self.trigger}|{self.regime_filter}|{self.fail}"
                f"{'' if 'F4' not in self.fail else f'({self.time_stop})'}|{self.confirm}|{self.add_arch}|"
                f"{self.exit}|cap{self.capital_arch}{self.max_probes}"
                f"{'' if self.capital_arch == 'A' else f'r{self.probe_reserve}'}")

    def hash(self) -> str:
        d = self.to_dict()
        d.pop("notes", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @staticmethod
    def from_dict(d: dict) -> "StrategyConfig":
        d = dict(d)
        d["disc_features"] = tuple(d.get("disc_features", ()))
        return StrategyConfig(**d)
