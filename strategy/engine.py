"""EMERGING_LEADER_STATE_MACHINE — one campaign per stock.

States: DISCOVERY(0) -> PROBE(1) -> FAILED_PROBE(2A) | CONFIRMED(2B) -> ADD(3) -> FULL(4) -> EXIT(5)

Timing (no look-ahead):
  * signal evaluated on close t  -> orders for open t+1
  * F1 / hard stop are resting stop orders: fill at min(open, stop) when low <= stop
  * all other exit / add decisions are taken on close d -> executed at open d+1
  * locked limit-up (can't buy) / locked limit-down (can't sell) days postpone the order
Sizes are in "slots" (1.0 = one normal position = 10% of equity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy.config import StrategyConfig
from strategy.signals import Arrays

NaN = float("nan")


@dataclass
class Leg:
    day: int
    px: float
    size: float          # slots
    notional: float      # currency (= size * slot_value)
    kind: str            # PROBE | ADD1 | ADD2


@dataclass
class Campaign:
    A: Arrays
    j: int
    t: int                          # signal day index
    cfg: StrategyConfig
    slip: float                     # per side, fraction
    cost: float                     # round trip fraction (half per side)
    slot_value: float = 1.0
    end_idx: int = 0
    rank: float = NaN
    # --- dynamic state ---
    state: str = "SIGNAL"
    legs: list = field(default_factory=list)
    pending: list = field(default_factory=list)     # [(kind, target_size, reason)]
    stop: float = NaN
    probe_stop0: float = NaN
    base_high: float = NaN
    rs20_sig: float = NaN
    entry_day: int = -1
    entry_px: float = NaN
    mkt_entry: float = NaN
    sec_entry: float = NaN
    confirm_day: int = -1
    confirm_px: float = NaN
    confirm_rule: str = ""
    exit_day: int = -1
    exit_px: float = NaN
    exit_reason: str = ""
    max_close: float = NaN
    max_high: float = NaN
    min_low: float = NaN
    probe_mfe: float = NaN
    probe_mae: float = NaN
    below_ma20: int = 0
    outp_days: int = 0
    days_obs: int = 0
    slot_days: float = 0.0
    events: list = field(default_factory=list)      # (day, event, price, size, reason)
    blocked_adds: int = 0
    skipped: str = ""
    track: bool = False                              # record daily diagnostics (dashboard / case study)
    daily: list = field(default_factory=list)
    last_conf: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def size(self) -> float:
        return sum(l.size for l in self.legs)

    def avg_cost(self) -> float:
        n = sum(l.notional for l in self.legs)
        return sum(l.notional * l.px for l in self.legs) / n if n else NaN

    def is_open(self) -> bool:
        return self.state in ("PROBE", "CONFIRMED", "ADD", "FULL")

    def is_probe(self) -> bool:
        return self.state == "PROBE"

    # ------------------------------------------------------------------
    def init_signal(self) -> bool:
        A, j, t = self.A, self.j, self.t
        self.stop = A.LOW3[t, j]
        self.probe_stop0 = self.stop
        self.base_high = A.HI20[t, j]
        self.rs20_sig = A.RS20[t, j]
        if not np.isfinite(self.stop):
            self.skipped = "no_stop"
            return False
        self.pending = [("ENTRY", self.cfg.probe_size, "probe_signal")]
        self.events.append((t, "DISCOVERY/PROBE_SIGNAL", A.C[t, j], 0.0, "probe conditions met at close"))
        return True

    # ------------------------------------------------------------------
    def on_open(self, d: int, alloc=None) -> None:
        """Execute pending orders at the open of day d. alloc(kind, size) -> granted size."""
        if not self.pending:
            return
        A, j = self.A, self.j
        o = A.O[d, j]
        if not np.isfinite(o):
            return  # no trade today -> keep pending
        keep = []
        for kind, target, reason in self.pending:
            if kind == "ENTRY":
                gap_ok = o <= A.C[self.t, j] * (1 + self.cfg.max_gap)
                if A.LOCK_UP[d, j] or A.G[d, j] >= 1.095 or not gap_ok or o <= self.stop:
                    self.state = "SKIPPED"
                    self.skipped = ("limit_up_open" if (A.LOCK_UP[d, j] or A.G[d, j] >= 1.095)
                                    else "gap_too_large" if not gap_ok else "open_below_stop")
                    self.pending = []
                    return
                size = target if alloc is None else alloc("PROBE", target)
                if size <= 1e-9:
                    self.state = "SKIPPED"
                    self.skipped = "no_capacity"
                    self.pending = []
                    return
                px = o * (1 + self.slip)
                self.legs.append(Leg(d, px, size, size * self.slot_value, "PROBE"))
                self.state = "PROBE"
                self.entry_day, self.entry_px = d, px
                self.mkt_entry = A.MKT_O[d] if np.isfinite(A.MKT_O[d]) else A.MKT_C[d - 1]
                self.sec_entry = A.SEC[d - 1, j]
                self.max_close, self.max_high, self.min_low = px, px, px
                self.events.append((d, "PROBE", px, size, f"probe {size:.2f} slot @open"))
            elif kind == "EXIT":
                if A.LOCK_DN[d, j]:
                    keep.append((kind, target, reason))
                    continue
                self._close_out(d, o * (1 - self.slip), reason)
                self.pending = []
                return
            elif kind in ("ADD1", "ADD2"):
                if A.LOCK_UP[d, j] or A.G[d, j] >= 1.095:
                    self.blocked_adds += 1
                    continue   # cancel add (do not chase a locked limit-up)
                want = target - self.size()
                if want <= 1e-9:
                    continue
                size = want if alloc is None else alloc(kind, want)
                if size < want - 1e-9:
                    self.blocked_adds += 1
                if size > 1e-9:
                    px = o * (1 + self.slip)
                    self.legs.append(Leg(d, px, size, size * self.slot_value, kind))
                    self.events.append((d, kind, px, size, reason))
                full = self.size() >= max(self.cfg.add_targets[-1] if self.cfg.add_targets else 0, 1.0) - 1e-6
                self.state = "FULL" if full else "ADD"
                if full:
                    self.events.append((d, "FULL", px if size > 1e-9 else o, self.size(), "full position"))
        self.pending = keep

    # ------------------------------------------------------------------
    def on_intraday(self, d: int) -> None:
        if not self.is_open():
            return
        A, j = self.A, self.j
        lo = A.L[d, j]
        if not np.isfinite(lo):
            return
        if lo <= self.stop:
            o = A.O[d, j]
            if A.LOCK_DN[d, j]:
                self.pending = [("EXIT", 0, "F1_PROBE_LOW" if self.is_probe() else "HARD_STOP")]
                return
            px = min(o, self.stop) if d != self.entry_day else self.stop
            self._close_out(d, px * (1 - self.slip), "F1_PROBE_LOW" if self.is_probe() else "HARD_STOP")

    # ------------------------------------------------------------------
    def on_close(self, d: int) -> None:
        if not self.is_open():
            return
        self._on_close(d)
        if self.track and self.legs:
            A, j = self.A, self.j
            c = A.C[d, j]
            row = {"day": d, "state": self.state, "close": c, "stop": self.stop, "size": self.size(),
                   "avg_cost": self.avg_cost(), "unrealized": c / self.avg_cost() - 1 if np.isfinite(c) else NaN,
                   "pending": ",".join(k for k, _, _ in self.pending)}
            row.update({f"conf_{k}": v for k, v in self.last_conf.items()})
            if np.isfinite(c):
                row["dist_ma20"] = c / A.MA20[d, j] - 1
                row["dist_ma10"] = c / A.MA10[d, j] - 1
                row["dist_stop"] = c / self.stop - 1 if self.stop > 0 else NaN
                row["mkt_rel_since_probe"] = (c / self.entry_px) / (A.MKT_C[d] / self.mkt_entry) - 1
            self.daily.append(row)

    def _on_close(self, d: int) -> None:
        A, j, cfg = self.A, self.j, self.cfg
        c = A.C[d, j]
        if not np.isfinite(c):
            if d >= A.last_valid[j] >= 0 or d >= self.end_idx:
                self._force_end(d)
            return
        self.slot_days += self.size()
        self.days_obs += 1
        self.max_close = max(self.max_close, c)
        if np.isfinite(A.H[d, j]):
            self.max_high = max(self.max_high, A.H[d, j])
        if np.isfinite(A.L[d, j]):
            self.min_low = min(self.min_low, A.L[d, j])
        mkt_rel = (c / self.entry_px) / (A.MKT_C[d] / self.mkt_entry) - 1
        if d > self.entry_day and np.isfinite(A.C[d - 1, j]):
            r_s = c / A.C[d - 1, j] - 1
            r_m = A.MKT_C[d] / A.MKT_C[d - 1] - 1
            self.outp_days += int(r_s > r_m)
        if d >= self.end_idx or (A.last_valid[j] >= 0 and d >= A.last_valid[j]):
            self._close_out(d, c * (1 - self.slip), "END_OF_WINDOW" if d >= self.end_idx else "DELISTED/SUSPENDED")
            return

        if self.state == "PROBE":
            self.probe_mfe = self.max_high / self.entry_px - 1
            self.probe_mae = self.min_low / self.entry_px - 1
            held = d - self.entry_day + 1
            if held >= cfg.min_days_confirm:
                rule = self._confirm(d, c, mkt_rel, held)
                if rule:
                    self.state = "CONFIRMED"
                    self.confirm_day, self.confirm_px, self.confirm_rule = d, c, rule
                    self.stop = max(self.stop, A.LOW3[d, j]) if np.isfinite(A.LOW3[d, j]) else self.stop
                    self.events.append((d, "CONFIRMED", c, self.size(), rule))
                    if cfg.add_targets:
                        self.pending = [("ADD1", cfg.add_targets[0], f"confirmation ({rule})")]
                    return
            reason = self._fail(d, c, mkt_rel, held)
            if reason:
                self.pending = [("EXIT", 0, reason)]
            return

        # ---- confirmed / add / full -------------------------------------------------
        reason = self._exit_rule(d, c)
        if reason:
            self.pending = [("EXIT", 0, reason)]
            return
        if (len(cfg.add_targets) > 1 and self.state == "ADD" and d >= self.confirm_day + 3
                and not any(k == "ADD2" for k, _, _ in self.pending)):
            hi_since = np.nanmax(A.H[self.confirm_day:d, j]) if d > self.confirm_day else np.inf
            if c > hi_since and A.TREND[d, j] > 0 and c > self.avg_cost() * 1.03:
                self.pending = [("ADD2", cfg.add_targets[1], "second confirmation: new high + trend")]

    # ------------------------------------------------------------------
    def _confirm(self, d, c, mkt_rel, held) -> str:
        A, j = self.A, self.j
        res = {}
        res["price"] = (c > self.base_high) and (c >= self.max_close - 1e-12) and (c > self.entry_px)
        res["rs"] = (A.RSL[d, j] >= A.RSLMAX40[d, j] - 1e-9) and (A.RS20[d, j] > self.rs20_sig) and \
            (d >= 5 and A.RS40[d, j] > A.RS40[d - 5, j])
        sec_rel = (c / self.entry_px) / (A.SEC[d, j] / self.sec_entry) - 1 if self.sec_entry > 0 else mkt_rel
        res["persist"] = held >= 3 and self.days_obs > 0 and (self.outp_days / max(held - 1, 1) >= 0.6) and \
            mkt_rel >= 0.03 and sec_rel > 0
        res["trend"] = A.TREND[d, j] > 0
        self.last_conf = {k: bool(v) for k, v in res.items()}
        parts = self.cfg.confirm.split("+")
        return self.cfg.confirm if all(res.get(p, False) for p in parts) else ""

    def _fail(self, d, c, mkt_rel, held) -> str:
        A, j, cfg = self.A, self.j, self.cfg
        rules = cfg.fail.split("+")
        if "F2" in rules and c < A.MA10[d, j] and c < self.entry_px:
            return "F2_STRUCTURE"
        if "F3" in rules and mkt_rel <= cfg.f3_thresh:
            return "F3_RS_FAILURE"
        if "F4" in rules and held >= cfg.time_stop and c <= self.entry_px:
            return f"F4_TIME_{cfg.time_stop}D"
        if held >= cfg.probe_max_life:
            return "PROBE_MAX_LIFE"
        return ""

    def _exit_rule(self, d, c) -> str:
        A, j, cfg = self.A, self.j, self.cfg
        ma20, ma10 = A.MA20[d, j], A.MA10[d, j]
        ex = cfg.exit
        if ex in ("TF", "MP"):
            self.below_ma20 = self.below_ma20 + 1 if c < ma20 else 0
            if self.below_ma20 >= 2:
                return "TREND_FAILURE_MA20x2"
        if ex == "MS":
            swing = np.nanmin(A.L[max(d - 10, 0):d, j]) if d > 0 else np.nan
            if c < ma10:
                return "MA10_EXIT"
            if np.isfinite(swing) and c < swing:
                return "STRUCTURE_BREAK"
        if ex == "MP":
            ac = self.avg_cost()
            peak = self.max_close / ac - 1
            if peak >= cfg.mfe_trigger and c < ac * (1 + 0.5 * peak):
                return "MFE_PROTECTION"
        return ""

    # ------------------------------------------------------------------
    def _close_out(self, d: int, px: float, reason: str) -> None:
        if not self.legs:
            self.state = "SKIPPED"
            return
        was_probe = self.state == "PROBE"
        self.exit_day, self.exit_px, self.exit_reason = d, px, reason
        self.state = "FAILED_PROBE" if was_probe and not reason.startswith(("END", "DELIST")) else "EXIT"
        if was_probe and reason.startswith(("END", "DELIST")):
            self.state = "EXIT"
        self.events.append((d, "FAILED_PROBE" if self.state == "FAILED_PROBE" else "EXIT", px, self.size(), reason))
        self.pending = []

    def _force_end(self, d: int) -> None:
        A, j = self.A, self.j
        k = d
        while k > self.entry_day and not np.isfinite(A.C[k, j]):
            k -= 1
        self._close_out(d, A.C[k, j] * (1 - self.slip), "DELISTED/SUSPENDED")

    # ------------------------------------------------------------------
    def pnl(self) -> float:
        """Currency PnL net of costs (half round-trip cost on each side)."""
        if not self.legs or not np.isfinite(self.exit_px):
            return NaN
        half = self.cost / 2
        tot = 0.0
        for l in self.legs:
            gross = l.notional * (self.exit_px / l.px - 1)
            tot += gross - half * l.notional - half * l.notional * self.exit_px / l.px
        return tot

    def mtm_value(self, d: int) -> float:
        """Market value at close d (for portfolio equity)."""
        A, j = self.A, self.j
        c = A.C[d, j]
        if not np.isfinite(c):
            k = d
            while k > 0 and not np.isfinite(A.C[k, j]):
                k -= 1
            c = A.C[k, j]
        return sum(l.notional * c / l.px for l in self.legs)

    def record(self) -> dict:
        A, j = self.A, self.j
        dates = A.dates
        inv = sum(l.notional for l in self.legs)
        pnl = self.pnl()
        probe = self.legs[0] if self.legs else None
        adds = [l for l in self.legs if l.kind != "PROBE"]
        add_px = adds[0].px if adds else NaN
        exit_px = self.exit_px
        rec = {
            "stock_id": A.ids[j], "signal_date": dates[self.t],
            "probe_date": dates[self.entry_day] if self.entry_day >= 0 else pd.NaT,
            "probe_price": self.entry_px, "probe_size": probe.size if probe else NaN,
            "probe_stop": self.probe_stop0,
            "probe_risk": 1 - self.probe_stop0 / self.entry_px if self.legs else NaN,
            "rank_score": self.rank,
            "confirmed": self.confirm_day >= 0,
            "confirm_date": dates[self.confirm_day] if self.confirm_day >= 0 else pd.NaT,
            "confirm_price": self.confirm_px, "confirm_rule": self.confirm_rule,
            "days_probe_to_confirm": (self.confirm_day - self.entry_day) if self.confirm_day >= 0 else NaN,
            "n_adds": len(adds),
            "add_date": dates[adds[0].day] if adds else pd.NaT, "add_price": add_px,
            "add2_date": dates[adds[1].day] if len(adds) > 1 else pd.NaT,
            "add2_price": adds[1].px if len(adds) > 1 else NaN,
            "full": any(e[1] == "FULL" for e in self.events),
            "max_size": self.size(),
            "exit_date": dates[self.exit_day] if self.exit_day >= 0 else pd.NaT,
            "exit_price": exit_px, "exit_reason": self.exit_reason, "final_state": self.state,
            "holding_days": (self.exit_day - self.entry_day + 1) if self.exit_day >= 0 else NaN,
            "invested": inv, "pnl": pnl,
            "pnl_slots": pnl / self.slot_value if self.slot_value else pnl,
            "ret_on_invested": pnl / inv if inv else NaN,
            "probe_leg_ret": (exit_px / probe.px - 1) if probe else NaN,
            "mfe": self.max_high / self.entry_px - 1 if self.legs else NaN,
            "mae": self.min_low / self.entry_px - 1 if self.legs else NaN,
            "probe_mfe": self.probe_mfe, "probe_mae": self.probe_mae,
            "add_premium": add_px / self.entry_px - 1 if adds else NaN,
            "add_leg_ret": exit_px / add_px - 1 if adds else NaN,
            "add_mfe": (np.nanmax(A.H[adds[0].day:self.exit_day + 1, j]) / add_px - 1) if adds else NaN,
            "add_mae": (np.nanmin(A.L[adds[0].day:self.exit_day + 1, j]) / add_px - 1) if adds else NaN,
            "slot_days": self.slot_days, "blocked_adds": self.blocked_adds, "skipped": self.skipped,
        }
        return rec


# ---------------------------------------------------------------------------
# Trade-level (unconstrained) simulation: every probe signal becomes a campaign,
# one active campaign per stock, cooldown after exit.
# ---------------------------------------------------------------------------
def run_campaign_alone(A: Arrays, j: int, t: int, cfg: StrategyConfig, end_idx: int, slip: float, cost: float,
                       rank: float = NaN, track: bool = False) -> Campaign | None:
    cp = Campaign(A=A, j=j, t=t, cfg=cfg, slip=slip, cost=cost, end_idx=end_idx, rank=rank, track=track)
    if not cp.init_signal():
        return cp
    d = t + 1
    while d <= end_idx:
        cp.on_open(d)
        if cp.state in ("SKIPPED", "EXIT", "FAILED_PROBE"):
            break
        cp.on_intraday(d)
        if not cp.is_open():
            break
        cp.on_close(d)
        if not cp.is_open() and not cp.pending:
            break
        d += 1
    if cp.is_open():
        cp._close_out(end_idx, A.C[end_idx, j] * (1 - slip) if np.isfinite(A.C[end_idx, j]) else cp.entry_px,
                      "END_OF_WINDOW")
    return cp


def trade_level(A: Arrays, signal: np.ndarray, cfg: StrategyConfig, start_idx: int, end_idx: int,
                slip: float, cost: float, rank: np.ndarray | None = None, keep_campaigns: bool = False):
    """Run all probe signals in [start_idx, end_idx] independently (capacity-free)."""
    recs, camps = [], []
    T, N = signal.shape
    sig_days = {j: np.where(signal[start_idx:end_idx, j])[0] + start_idx for j in range(N)}
    for j, days in sig_days.items():
        if len(days) == 0:
            continue
        free_from = start_idx
        for t in days:
            if t < free_from:
                continue
            cp = run_campaign_alone(A, j, t, cfg, end_idx, slip, cost,
                                    rank[t, j] if rank is not None else NaN, track=keep_campaigns)
            if cp is None or not cp.legs:
                continue
            recs.append(cp.record())
            if keep_campaigns:
                camps.append(cp)
            free_from = cp.exit_day + cfg.cooldown + 1
    df = pd.DataFrame(recs)
    if keep_campaigns:
        return df, camps
    return df
