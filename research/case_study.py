"""3653 健策 case study / regression test (Parts 49-52).

The frozen V1 rules are applied unchanged. Nothing here feeds back into any threshold.
Every row of the timeline uses only information available at that day's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data.finmind_client import FinMindCache
from strategy.engine import trade_level
from strategy.selection import window_idx
from strategy.signals import condition_frames, discovery_score

SECTOR_5S = {"電子零組件業": "ElectronicPartsComponents", "半導體業": "Semiconductor", "電腦及週邊設備業":
             "ComputerPeripheralEquipment", "光電業": "Optoelectronic", "通信網路業": "CommunicationsInternet",
             "電子通路業": "ElectronicProductsDistribution", "資訊服務業": "InformationService",
             "其他電子業": "OtherElectronic", "電子工業": "Electronic"}


def intraday_case(symbol: str, sector: str, dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalized intraday paths (stock / TAIEX / sector index) + VWAP diagnostics per day."""
    cache = FinMindCache(client=None)
    paths, summ = [], []
    sec_id = SECTOR_5S.get(sector)
    for d in dates:
        ds = str(d.date())
        k = cache.read("TaiwanStockKBar", f"{symbol}_{ds}")
        if k.empty:
            continue
        k = k.copy()
        k["ts"] = pd.to_datetime(ds + " " + k["minute"].astype(str))
        k = k.sort_values("ts")
        k["pv"] = k["close"] * k["volume"]
        k["vwap"] = k["pv"].cumsum() / k["volume"].cumsum().replace(0, np.nan)
        base = k["open"].iloc[0]
        s = pd.DataFrame({"ts": k["ts"], "stock": k["close"] / base * 100, "vwap": k["vwap"] / base * 100})
        m5 = cache.read("TaiwanVariousIndicators5Seconds", f"TAIEX5S_{ds}")
        if not m5.empty:
            m5 = m5.copy()
            m5["ts"] = pd.to_datetime(m5["date"])
            m1 = m5.set_index("ts")["TAIEX"].resample("1min").last().ffill()
            s["market"] = s["ts"].map(m1 / m1.iloc[0] * 100)
        sx = cache.read("TaiwanStockEvery5SecondsIndex", f"SECTOR5S_{ds}")
        if not sx.empty and sec_id is not None:
            sx = sx[sx["stock_id"] == sec_id].copy()
            if len(sx):
                sx["ts"] = pd.to_datetime(sx["date"].astype(str) + " " + sx["time"].astype(str))
                s1 = sx.set_index("ts")["price"].resample("1min").last().ffill()
                s["sector"] = s["ts"].map(s1 / s1.iloc[0] * 100)
        s["date"] = d
        s["rel_spread_vs_market"] = s["stock"] - s.get("market", np.nan)
        paths.append(s)
        above = (k["close"] >= k["vwap"]).mean()
        n_hi = int((k["high"] > k["high"].cummax().shift(1)).sum())
        mid = len(k) // 2
        lows = k["low"].rolling(30, min_periods=10).min()
        summ.append({"date": d, "pct_time_above_vwap": above, "close_vs_vwap": k["close"].iloc[-1] / k["vwap"].iloc[-1] - 1,
                     "vwap_slope": k["vwap"].iloc[-1] / k["vwap"].iloc[min(30, len(k) - 1)] - 1,
                     "intraday_new_high_count": n_hi,
                     "intraday_higher_lows": bool(lows.dropna().is_monotonic_increasing) if lows.notna().sum() > 3 else False,
                     "low_to_close_recovery": k["close"].iloc[-1] / k["low"].min() - 1,
                     "morning_weakness_recovered": bool(k["low"].iloc[:60].min() < base * 0.99 and k["close"].iloc[-1] > base),
                     "afternoon_strength": k["close"].iloc[-1] / k["close"].iloc[mid] - 1,
                     "stock_oc": k["close"].iloc[-1] / base - 1,
                     "market_oc": (s["market"].iloc[-1] / 100 - 1) if "market" in s and s["market"].notna().any() else np.nan,
                     "sector_oc": (s["sector"].iloc[-1] / 100 - 1) if "sector" in s and s["sector"].notna().any() else np.nan})
    return (pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()), pd.DataFrame(summ)


def run_case(p, F, cfg, A, sc, res, symbol: str = config.CASE_SYMBOL, period=config.CASE_PERIOD) -> dict:
    out: dict = {"symbol": symbol, "found": symbol in p.ids}
    if symbol not in p.ids:
        out["reason"] = "資料中沒有此股票"
        return out
    disc = discovery_score(F, p.universe, cfg.disc_features)
    conds = condition_frames(p, F, cfg, disc_pct=disc)
    ctx0 = pd.Timestamp(period[0]) - pd.Timedelta(days=45)
    d0, d1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    dates = p.dates[(p.dates >= ctx0) & (p.dates <= d1)]
    j = p.ids.index(symbol)
    # campaigns for this stock only, PIT, from the same frozen engine (live window to data end)
    sig, rank, _ = sc.get(cfg)
    only = np.zeros_like(sig)
    only[:, j] = sig[:, j]
    s, e = window_idx(A.dates, config.RESEARCH_START, str(p.dates[-1].date()))
    camps_df, camps = trade_level(A, only, cfg, s, e, config.BASE_SLIPPAGE_BPS / 1e4, config.BASE_COST, rank,
                                  keep_campaigns=True)
    daily = {}
    events = []
    for cp in camps:
        for r in cp.daily:
            daily[A.dates[r["day"]]] = r
        for (d, ev, px, sz, why) in cp.events:
            events.append({"date": A.dates[d], "event": ev, "price": px, "size": sz, "reason": why})
    ev = pd.DataFrame(events)
    mk = F["mkt"]
    c = p.c[symbol]
    base_d = dates[dates >= d0][0]
    sec_lvl = F["sec_level"][symbol]
    rows = []
    pos_live = res.get("pf", {}).get("LIVE", {}).get("positions", pd.DataFrame())
    pos_live = pos_live[pos_live["stock_id"] == symbol] if len(pos_live) else pos_live
    for d in dates:
        r = {"date": d, "in_case_period": d >= d0, "raw_close": p.raw_c.at[d, symbol], "adj_close": c.at[d],
             "ret": p.r.at[d, symbol], "market_close": p.market.at[d, "close"], "m_ret": mk.at[d, "m_ret"],
             "regime": mk.at[d, "regime"],
             "stock_norm": c.at[d] / c.at[base_d] * 100,
             "market_norm": p.market.at[d, "adj_close"] / p.market.at[base_d, "adj_close"] * 100,
             "sector_norm": sec_lvl.at[d] / sec_lvl.at[base_d] * 100,
             "excess_1d": F["ex_1"].at[d, symbol], "cex_10": F["cex_10"].at[d, symbol],
             "srs_10": F["srs_10"].at[d, symbol], "rs_20": F["rs_20"].at[d, symbol], "rs_60": F["rs_60"].at[d, symbol],
             "rs_accel": F["rs_accel"].at[d, symbol], "outp_10": F["outp_10"].at[d, symbol],
             "is_event": F["is_event"].at[d, symbol], "atr_pct": F["atr_pct"].at[d, symbol],
             "disc_pct": disc.at[d, symbol], "rank_score": conds["rank_score"].at[d, symbol],
             "watch": bool(disc.at[d, symbol] >= cfg.watch_q), "emerging": bool(disc.at[d, symbol] >= cfg.disc_q)}
        for k in ("c_universe", "c_no_disposition", "c_probe_score", "c_trigger", "c_trig_rs_high",
                  "c_trig_price_high", "c_trig_indep", "c_regime", "c_stop_ok", "probe_signal"):
            r[k] = bool(conds[k].at[d, symbol])
        r["stop_dist"] = conds["stop_dist"].at[d, symbol]
        r["disposition"] = bool(p.disp.at[d, symbol])
        dd = daily.get(d)
        if dd:
            r.update({"campaign_state": dd["state"], "position_size": dd["size"], "stop": dd["stop"],
                      "dist_stop": dd.get("dist_stop"), "unrealized": dd.get("unrealized"),
                      **{k: v for k, v in dd.items() if k.startswith("conf_")}})
        evd = ev[ev["date"] == d] if len(ev) else ev
        r["events"] = " | ".join(f"{x.event}@{x.price:.1f}" for x in evd.itertuples()) if len(evd) else ""
        if len(pos_live):
            pl = pos_live[pos_live["date"] == d]
            r["portfolio_state"] = pl["state"].iloc[0] if len(pl) else ""
            r["portfolio_size"] = pl["size_slots"].iloc[0] if len(pl) else 0.0
        rows.append(r)
    tl = pd.DataFrame(rows)
    # intraday (only if minute data was downloaded)
    ip, isum = intraday_case(symbol, p.sector.get(symbol, ""), list(dates[dates >= d0]))
    if len(isum):
        tl = tl.merge(isum, on="date", how="left")
    out.update(timeline=tl, events=ev, campaigns=camps_df, intraday=ip, intraday_summary=isum)
    # key dates (PIT)
    cp_ = tl[tl["in_case_period"]]

    def first(mask, col="date"):
        x = cp_[mask]
        return x[col].iloc[0] if len(x) else None
    out["first_watch"] = first(cp_["watch"])
    out["first_emerging"] = first(cp_["emerging"])
    out["first_watch_ctx"] = tl[tl["watch"]]["date"].iloc[0] if tl["watch"].any() else None
    out["first_probe_signal"] = first(cp_["probe_signal"])
    inper = ev[(ev["date"] >= ctx0) & (ev["date"] <= d1)] if len(ev) else ev
    for key, name in (("probe", "PROBE"), ("confirm", "CONFIRMED"), ("add", "ADD1"), ("add2", "ADD2"),
                      ("full", "FULL"), ("failed", "FAILED_PROBE"), ("exit", "EXIT")):
        x = inper[inper["event"] == name] if len(inper) else inper
        out[key] = (x["date"].iloc[0], float(x["price"].iloc[0]), float(x["size"].iloc[0]), x["reason"].iloc[0]) \
            if len(x) else None
    # why not captured: count failing probe conditions on days with disc >= watch
    fail_counts = {}
    for k in ("c_universe", "c_no_disposition", "c_probe_score", "c_trigger", "c_regime", "c_stop_ok"):
        fail_counts[k] = int((~cp_[k]).sum())
    out["fail_counts"] = fail_counts
    # closest call to the stop while in probe
    pr = tl[tl.get("campaign_state", pd.Series(dtype=str)) == "PROBE"] if "campaign_state" in tl else tl.iloc[0:0]
    out["closest_stop"] = float(pr["dist_stop"].min()) if len(pr) and "dist_stop" in pr else None
    # breakaway day: first day stock_norm - market_norm > 5 and stays positive afterwards
    gap = tl["stock_norm"] - tl["market_norm"]
    brk = None
    for i in range(len(tl)):
        if tl["in_case_period"].iloc[i] and gap.iloc[i] > 5 and (gap.iloc[i:] > 0).all():
            brk = tl["date"].iloc[i]
            break
    out["breakaway"] = brk
    return out
