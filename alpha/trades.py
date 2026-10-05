"""Generic structural-stop trade simulator for the Weinstein / High R/R research (long and short).

Execution semantics (daily data, no look-ahead):
  * setup known at close of day t; orders live from the open of t+1
  * BUY-STOP (breakout): fill at open if open >= trigger (gap; skipped if gap > max_gap), else at the
    trigger when high >= trigger; order expires after `order_days`
  * MARKET-ON-OPEN: fill at open t+1 (pullback / continuation re-entries)
  * initial STRUCTURAL STOP is a resting stop: fill at min(open, stop) when low <= stop (long)
  * close-based exit decisions at close d are executed at open d+1
  * locked limit-up (long entry) / locked limit-down (long exit) postpone or cancel
Short side mirrors everything (sell-stop below support; buy-to-cover stops above).
Returns are on adjusted prices (dividends included); sizes in slots (1.0 = 10% equity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NaN = float("nan")


@dataclass
class ExitSpec:
    name: str
    trail: str = "none"         # none | ma150 | swing | ma150_swing (textbook trailing below/above MA & swing)
    trend_fail: bool = False    # close beyond MA20 two days -> exit
    bb: str = "none"            # none | B1 (bar-close fail-to-hold)
    bb_n: int = 20
    bb_k: float = 2.0
    bb_partial: float = 1.0     # fraction exited on BB failure (1.0 full, 0.5 partial)
    mfe_protect: float = 0.0    # >0: after peak gain >= this, exit if give back half
    max_hold: int = 0           # 0 = none
    stage_exit: str = "none"    # none | investor | trader | modern  (weekly Weinstein stage rules)
    weak_vol: str = "none"      # none | weekly | daily  (book p.104-105: weak-volume breakout handling)
    ma10w_exit: bool = False    # weekly close beyond the 10-week MA -> exit next open (MODERNIZED)
    breakeven_r: float = 0.0    # >0: once MFE >= k*R move the stop to the entry price


@dataclass
class Mats:
    """numpy matrices (dates x stocks) needed by the simulator."""
    dates: pd.DatetimeIndex
    ids: list
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    RAW_C: np.ndarray
    G: np.ndarray            # open / prev close
    LOCK_UP: np.ndarray
    LOCK_DN: np.ndarray
    MA20: np.ndarray
    MA150: np.ndarray
    ATR: np.ndarray
    SWL: np.ndarray          # most recent confirmed swing low (PIT)
    SWH: np.ndarray          # most recent confirmed swing high (PIT)
    BB: dict                 # (n,k) -> (upper, lower)
    VAL: np.ndarray
    last_valid: np.ndarray
    WE: np.ndarray | None = None          # bool: day is the last trading day of its week
    STAGE_WE: np.ndarray | None = None    # weekly stage on week-end days (else 0)
    EPL_WE: np.ndarray | None = None      # episode min weekly low (stage-3 support) on week-end days
    EPH_WE: np.ndarray | None = None      # episode max weekly high (stage-1 resistance) on week-end days
    MA10W_WE: np.ndarray | None = None
    VOLS_W: np.ndarray | None = None      # weekly volume-strength flag on week-end days (1/0/nan)
    VOLS_D: np.ndarray | None = None      # daily breakout-volume flag (vol >= 2x prior-5d avg)

    @property
    def T(self):
        return self.C.shape[0]


def attach_weekly(M: "Mats", p, ctx) -> "Mats":
    """Add Weinstein weekly matrices (aligned on week-end trading days) to M."""
    from alpha.weekly import week_end_index
    we = week_end_index(p.dates)
    is_we = (we.to_numpy() == p.dates.to_numpy())
    pos = p.dates.get_indexer(ctx.wd.week_end)
    T, N = M.C.shape
    def put(frame, fill=np.nan):
        a = np.full((T, N), fill, dtype=float)
        a[pos] = frame.to_numpy(float)
        return a
    M.WE = is_we
    M.STAGE_WE = put(ctx.S["stage"], 0)
    M.EPL_WE = put(ctx.S["ep_minl"])
    M.EPH_WE = put(ctx.S["ep_maxh"])
    M.MA10W_WE = put(ctx.S["ma10w"])
    M.VOLS_W = put(ctx.vol_strong_w.astype(float))
    v = p.vol.where(~p.disp)
    M.VOLS_D = (v >= 2.0 * v.shift(1).rolling(5, min_periods=4).mean()).astype(float).where(v.notna()).to_numpy()
    return M


def build_mats(p, F, G) -> Mats:
    from strategy.signals import build_arrays
    A = build_arrays(p, F)
    c = p.c
    bb = {}
    for n, k in ((20, 2.0), (20, 2.5), (30, 2.0)):
        ma = c.rolling(n, min_periods=int(n * 0.8)).mean()
        sd = c.rolling(n, min_periods=int(n * 0.8)).std()
        bb[(n, k)] = ((ma + k * sd).to_numpy(), (ma - k * sd).to_numpy())
    swh = G.get("swing_high10")
    return Mats(dates=p.dates, ids=list(p.ids), O=A.O, H=A.H, L=A.L, C=A.C, RAW_C=A.RAW_C, G=A.G,
                LOCK_UP=A.LOCK_UP, LOCK_DN=A.LOCK_DN, MA20=F["ma20"].to_numpy(), MA150=G["ma150"].to_numpy(),
                ATR=F["atr"].to_numpy(), SWL=G["swing_low10"].to_numpy(),
                SWH=swh.to_numpy() if swh is not None else np.full_like(A.C, np.nan), BB=bb, VAL=A.VAL,
                last_valid=A.last_valid)


def simulate(M: Mats, j: int, t: int, side: int, entry: str, trigger: float, stop: float, ex: ExitSpec,
             end_idx: int, cost: float, slip: float, order_days: int = 5, max_gap: float = 0.06,
             short_fee: float = 0.0008, borrow_annual: float = 0.02, size: float = 1.0,
             fixed_entry: tuple | None = None, limit: float | None = None, level: float | None = None) -> dict | None:
    """side=+1 long, -1 short. entry in {'stop','open','fixed'}. fixed_entry=(day_idx, ADJUSTED price) is used
    when an intraday engine chose the exact entry second/price. Returns a trade record or None (no fill)."""
    O, H, L, C = M.O[:, j], M.H[:, j], M.L[:, j], M.C[:, j]
    half = cost / 2
    # ---------------- entry ----------------
    e = -1
    px = NaN
    armed = False              # stop-limit: stop triggered by a gap beyond the limit, limit order resting
    if fixed_entry is not None:
        e, px = int(fixed_entry[0]), float(fixed_entry[1])
        order_days = 0
    for d in range(t + 1, min(t + 1 + order_days, end_idx + 1)):
        o = O[d]
        if not np.isfinite(o):
            continue
        if side > 0 and (M.LOCK_UP[d, j] or M.G[d, j] >= 1.095):
            if entry == "open":
                return None
            continue
        if side < 0 and (M.LOCK_DN[d, j] or M.G[d, j] <= 0.905):
            if entry == "open":
                return None
            continue
        if entry == "open":
            e, px = d, o
        elif entry == "stoplimit":
            lim = limit
            if side > 0:
                if armed or o >= trigger:
                    if o <= lim:
                        e, px = d, o
                    elif L[d] <= lim:
                        e, px = d, lim
                    armed = True
                elif H[d] >= trigger:
                    e, px = d, trigger
            else:
                if armed or o <= trigger:
                    if o >= lim:
                        e, px = d, o
                    elif H[d] >= lim:
                        e, px = d, lim
                    armed = True
                elif L[d] <= trigger:
                    e, px = d, trigger
        elif side > 0:
            if o >= trigger:
                if o > trigger * (1 + max_gap):
                    return None
                e, px = d, o
            elif H[d] >= trigger:
                e, px = d, trigger
        else:
            if o <= trigger:
                if o < trigger * (1 - max_gap):
                    return None
                e, px = d, o
            elif L[d] <= trigger:
                e, px = d, trigger
        if e >= 0:
            break
        if side > 0 and np.isfinite(L[d]) and L[d] < stop:        # setup invalidated before trigger
            return None
        if side < 0 and np.isfinite(H[d]) and H[d] > stop:
            return None
    if e < 0:
        return None
    if fixed_entry is None:
        px = px * (1 + side * slip)
    if (side > 0 and stop >= px) or (side < 0 and stop <= px) or not np.isfinite(stop):
        return None
    risk = abs(px - stop) / px
    atr_e = M.ATR[t, j]
    # ---------------- manage ----------------
    pos = 1.0
    realized = 0.0             # sum of fraction * gross return
    exits = []
    cur_stop = stop
    peak = px
    trough = px
    mfe = 0.0
    mae = 0.0
    pending = None             # (fraction, reason)
    bb_armed = False
    bbu, bbl = M.BB[(ex.bb_n, ex.bb_k)] if ex.bb != "none" else (None, None)
    below = 0
    bb_done = False
    weak = None                # weak-volume breakout flag (None = not yet known)
    s3_done = False
    stop_tag = "STRUCTURAL_STOP"
    lvl = level if level is not None and np.isfinite(level) else (trigger if np.isfinite(trigger) else px)
    slot_days = 0.0
    d = e
    last = min(end_idx, M.T - 1)
    exit_day = -1
    while d <= last:
        o, hi, lo, c = O[d], H[d], L[d], C[d]
        if not np.isfinite(c):
            if 0 <= M.last_valid[j] < M.T - 1 and d >= M.last_valid[j]:
                k = d
                while k > e and not np.isfinite(C[k]):
                    k -= 1
                realized += pos * side * (C[k] / px - 1)
                exits.append((d, pos, C[k], "DELISTED"))
                pos = 0.0
                exit_day = d
                break
            d += 1
            continue
        # pending close-based exit at today's open
        if pending is not None and d > e and np.isfinite(o):
            blocked = (side > 0 and M.LOCK_DN[d, j]) or (side < 0 and M.LOCK_UP[d, j])
            if not blocked:
                frac, why = pending
                q = min(frac, pos)
                xp = o * (1 - side * slip)
                realized += q * side * (xp / px - 1)
                exits.append((d, q, xp, why))
                pos -= q
                pending = None
                if pos <= 1e-9:
                    exit_day = d
                    break
        # resting stop
        hit = (side > 0 and lo <= cur_stop) or (side < 0 and hi >= cur_stop)
        if hit:
            locked = (side > 0 and M.LOCK_DN[d, j]) or (side < 0 and M.LOCK_UP[d, j])
            if not locked:
                xp = (min(o, cur_stop) if side > 0 else max(o, cur_stop)) if d != e else cur_stop
                xp = xp * (1 - side * slip)
                realized += pos * side * (xp / px - 1)
                exits.append((d, pos, xp, stop_tag))
                pos = 0.0
                exit_day = d
                break
        # Bollinger B2: intraday fail-to-hold = resting stop at YESTERDAY's band once armed (1-second engine
        # refines the exact second / price on tick data; daily simulation uses this stop-proxy fill)
        if ex.bb == "B2" and bb_armed and not bb_done and d > e and bbu is not None:
            lv = bbu[d - 1, j] if side > 0 else bbl[d - 1, j]
            if np.isfinite(lv) and ((side > 0 and lo <= lv) or (side < 0 and hi >= lv)):
                locked = (side > 0 and M.LOCK_DN[d, j]) or (side < 0 and M.LOCK_UP[d, j])
                if not locked:
                    xp = (min(o, lv) if side > 0 else max(o, lv)) * (1 - side * slip)
                    q = min(ex.bb_partial, pos)
                    realized += q * side * (xp / px - 1)
                    exits.append((d, q, xp, "BB_B2_INTRADAY" if q >= pos - 1e-9 else "BB_B2_INTRADAY_PARTIAL"))
                    pos -= q
                    bb_done = True
                    if pos <= 1e-9:
                        exit_day = d
                        break
        slot_days += pos * size
        peak = max(peak, hi) if side > 0 else peak
        trough = min(trough, lo) if side < 0 else trough
        fav = (hi / px - 1) if side > 0 else (1 - lo / px)
        adv = (lo / px - 1) if side > 0 else (1 - hi / px)
        mfe = max(mfe, fav)
        mae = min(mae, adv)
        if d >= end_idx:
            realized += pos * side * (c * (1 - side * slip) / px - 1)
            exits.append((d, pos, c, "END_OF_WINDOW"))
            pos = 0.0
            exit_day = d
            break
        # ---- close-based exit rules ----
        why = None
        ma20, ma150 = M.MA20[d, j], M.MA150[d, j]
        if ex.trend_fail:
            cond = (c < ma20) if side > 0 else (c > ma20)
            below = below + 1 if cond else 0
            if below >= 2:
                why = "TREND_FAIL_MA20x2"
        if ex.mfe_protect > 0:
            g = (c / px - 1) * side
            pk = (peak / px - 1) if side > 0 else (1 - trough / px)
            if pk >= ex.mfe_protect and g < 0.5 * pk:
                why = why or "MFE_PROTECT"
        if ex.max_hold and d - e + 1 >= ex.max_hold:
            why = why or "TIME"
        if ex.bb == "B2" and not bb_done and bbu is not None:
            up, dn = bbu[d, j], bbl[d, j]
            if (side > 0 and c > up) or (side < 0 and c < dn):
                bb_armed = True
        if ex.bb == "B1" and not bb_done and bbu is not None:
            up, dn = bbu[d, j], bbl[d, j]
            if side > 0:
                if c > up:
                    bb_armed = True
                elif bb_armed and c < up:
                    if why is None:
                        if ex.bb_partial >= 0.999:
                            why = "BB_FAIL_TO_HOLD"
                        else:
                            pending = (ex.bb_partial, "BB_FAIL_TO_HOLD_PARTIAL")
                            bb_done = True
            else:
                if c < dn:
                    bb_armed = True
                elif bb_armed and c > dn:
                    if why is None:
                        if ex.bb_partial >= 0.999:
                            why = "BB_FAIL_TO_HOLD"
                        else:
                            pending = (ex.bb_partial, "BB_FAIL_TO_HOLD_PARTIAL")
                            bb_done = True
        # ---- Weinstein weekly / volume rules ----
        if ex.weak_vol != "none" and weak is None:
            if ex.weak_vol == "daily":
                f = M.VOLS_D[e, j]
                weak = bool(f == 0) if np.isfinite(f) else False
            elif M.WE[d]:
                f = M.VOLS_W[d, j]
                weak = bool(f == 0) if np.isfinite(f) else False
        if weak:
            if (side > 0 and c < lvl) or (side < 0 and c > lvl):
                why = why or "WEAK_VOL_BACK_BELOW_BREAKOUT"
            elif side * (c - px) >= (atr_e if np.isfinite(atr_e) else np.inf):
                why = why or "WEAK_VOL_FIRST_RALLY"
        if ex.stage_exit != "none" and M.WE is not None and M.WE[d]:
            stg = M.STAGE_WE[d, j]
            bad_full, bad_half = (4, 3) if side > 0 else (2, 1)
            if stg == bad_full:
                why = why or (f"STAGE{bad_full}_EXIT" if side > 0 else "STAGE2_COVER")
            elif stg == bad_half and not s3_done and ex.stage_exit in ("investor", "trader"):
                s3_done = True
                if ex.stage_exit == "trader":
                    why = why or ("STAGE3_EXIT" if side > 0 else "STAGE1_COVER")
                else:
                    pending = (0.5, "STAGE3_HALF" if side > 0 else "STAGE1_HALF_COVER")
                    lv = M.EPL_WE[d, j] * 0.99 if side > 0 else M.EPH_WE[d, j] * 1.01
                    if np.isfinite(lv):
                        if side > 0 and lv > cur_stop and lv < c:
                            cur_stop, stop_tag = lv, "STAGE3_SUPPORT_STOP"
                        if side < 0 and lv < cur_stop and lv > c:
                            cur_stop, stop_tag = lv, "STAGE1_RESIST_STOP"
            if ex.ma10w_exit:
                m10 = M.MA10W_WE[d, j]
                if np.isfinite(m10) and ((side > 0 and c < m10) or (side < 0 and c > m10)):
                    why = why or "MA10W_EXIT"
        if ex.breakeven_r > 0 and risk > 0 and mfe >= ex.breakeven_r * risk:
            if side > 0 and px > cur_stop and px < c:
                cur_stop, stop_tag = px, "BREAKEVEN_STOP"
            if side < 0 and px < cur_stop and px > c:
                cur_stop, stop_tag = px, "BREAKEVEN_STOP"
        if why is not None:
            pending = (1.0, why)
        # ---- trailing stop update (raise only) ----
        if ex.trail != "none":
            cand = []
            if ex.trail in ("ma150", "ma150_swing") and np.isfinite(ma150):
                cand.append(ma150 * (0.97 if side > 0 else 1.03))
            if ex.trail in ("swing", "ma150_swing"):
                sw = M.SWL[d, j] if side > 0 else M.SWH[d, j]
                if np.isfinite(sw):
                    cand.append(sw * (0.995 if side > 0 else 1.005))
            if cand:
                lvl_t = min(cand) if ex.trail == "ma150_swing" and side > 0 else (max(cand) if side < 0 and ex.trail == "ma150_swing" else cand[0])
                if side > 0 and lvl_t > cur_stop and lvl_t < c:
                    cur_stop, stop_tag = lvl_t, "TRAIL_STOP"
                if side < 0 and lvl_t < cur_stop and lvl_t > c:
                    cur_stop, stop_tag = lvl_t, "TRAIL_STOP"
        d += 1
    if pos > 1e-9:          # loop ended without exit (data end)
        k = min(last, M.T - 1)
        while k > e and not np.isfinite(C[k]):
            k -= 1
        realized += pos * side * (C[k] / px - 1)
        exits.append((k, pos, C[k], "END_OF_DATA"))
        exit_day = k
    hold = exit_day - e + 1
    fee = cost + (short_fee + borrow_annual * hold / 252 if side < 0 else 0.0)
    ret = realized - fee
    raw = lambda dd, pp: pp * M.RAW_C[dd, j] / M.C[dd, j] if np.isfinite(M.RAW_C[dd, j]) and M.C[dd, j] > 0 else NaN
    first_exit = exits[0] if exits else (exit_day, 1.0, NaN, "")
    return {
        "stock_id": M.ids[j], "side": "LONG" if side > 0 else "SHORT", "setup_date": M.dates[t],
        "entry_date": M.dates[e], "entry_price": raw(e, px), "entry_price_adj": px,
        "stop_price": raw(e, stop), "stop_dist_pct": risk, "stop_dist_atr": abs(px - stop) / atr_e if atr_e > 0 else NaN,
        "exit_date": M.dates[exit_day], "exit_reason": exits[-1][3] if exits else "",
        "first_exit_reason": first_exit[3], "n_exit_legs": len(exits),
        "holding_days": hold, "ret": ret, "r_multiple": ret / risk if risk > 0 else NaN,
        "mfe": mfe, "mae": mae, "mfe_r": mfe / risk if risk > 0 else NaN, "mae_r": mae / risk if risk > 0 else NaN,
        "slot_days": slot_days * size, "pnl_slots": ret * size, "size": size,
    }


def trade_stats(df: pd.DataFrame, years: float | None = None) -> dict:
    """Metrics emphasising payoff / right tail rather than win rate."""
    if df is None or len(df) == 0:
        return {"n": 0}
    r = df["ret"]
    w, l_ = r[r > 0], r[r < 0]
    srt = df["pnl_slots"].sort_values(ascending=False)
    tot = df["pnl_slots"].sum()
    n = len(df)
    k1, k5 = max(1, round(n * 0.01)), max(1, round(n * 0.05))
    if years is None:
        span = (df["entry_date"].max() - df["entry_date"].min()).days / 365.25 if n > 1 else 1
        years = max(span, 0.25)
    return {
        "n": n, "trades_per_year": n / years, "win_rate": float((r > 0).mean()),
        "pf": float(w.sum() / -l_.sum()) if l_.sum() < 0 else np.inf,
        "payoff": float(w.mean() / -l_.mean()) if len(w) and len(l_) else np.nan,
        "ev": float(r.mean()), "median_ret": float(r.median()),
        "avg_winner": float(w.mean()) if len(w) else np.nan, "avg_loser": float(l_.mean()) if len(l_) else np.nan,
        "avg_r": float(df["r_multiple"].mean()), "median_r": float(df["r_multiple"].median()),
        "mfe_mean": float(df["mfe"].mean()), "mae_mean": float(df["mae"].mean()),
        "mfe_mae_ratio": float(df["mfe"].mean() / -df["mae"].mean()) if df["mae"].mean() < 0 else np.nan,
        "stop_rate": float(df["exit_reason"].isin(["STRUCTURAL_STOP"]).mean()),
        "first_exit_stop_rate": float(df["first_exit_reason"].isin(["STRUCTURAL_STOP"]).mean()) if "first_exit_reason" in df else np.nan,
        "stop_dist_pct_med": float(df["stop_dist_pct"].median()), "stop_dist_atr_med": float(df["stop_dist_atr"].median()),
        "median_hold": float(df["holding_days"].median()),
        "ge10": float((r >= 0.10).mean()), "ge20": float((r >= 0.20).mean()), "ge30": float((r >= 0.30).mean()),
        "ge40": float((r >= 0.40).mean()), "n_ge20": int((r >= 0.20).sum()), "n_ge40": int((r >= 0.40).sum()),
        "top1pct_share": float(srt.iloc[:k1].sum() / tot) if tot > 0 else np.nan,
        "top5pct_share": float(srt.iloc[:k5].sum() / tot) if tot > 0 else np.nan,
        "pnl_ex_top5": float(srt.iloc[5:].sum()),
        "pnl_per_100_slot_days": float(100 * tot / df["slot_days"].sum()) if df["slot_days"].sum() > 0 else np.nan,
        "total_pnl_slots": float(tot),
    }
