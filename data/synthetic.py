"""OFFLINE SYNTHETIC DATA in FinMind schema — for code-path testing ONLY.

    EL_DATA_SOURCE=synthetic python -m data.synthetic

Writes into data/cache_synthetic/raw with exactly the same layout as the real
FinMind cache, so every downstream module can be exercised without network.

!!! Synthetic prices are random numbers with planted patterns. Any metric computed
!!! on them says NOTHING about the real Taiwan market. Outputs go to
!!! outputs_synthetic/ and the dashboard shows a red SYNTHETIC banner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("EL_DATA_SOURCE", "synthetic")

import config  # noqa: E402
from data.finmind_client import FinMindCache  # noqa: E402

SECTORS = ["半導體業", "電子零組件業", "電腦及週邊設備業", "光電業", "通信網路業", "其他電子業", "電子通路業",
           "資訊服務業", "電機機械", "生技醫療業", "塑膠工業", "鋼鐵工業", "航運業", "金融保險業", "建材營造業",
           "食品工業", "紡織纖維", "化學工業", "汽車工業", "觀光餐旅", "貿易百貨", "油電燃氣業", "綠能環保", "其他業"]


def _ohlc(ref: np.ndarray, ret: np.ndarray, gap_frac: np.ndarray, rng, vol: np.ndarray):
    close = ref * (1 + ret)
    open_ = ref * (1 + ret * gap_frac + rng.normal(0, 0.002, ret.shape))
    lim_hi, lim_lo = ref * 1.1, ref * 0.9
    open_ = np.clip(open_, lim_lo, lim_hi)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1, ret.shape)) * vol * 0.45)
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1, ret.shape)) * vol * 0.45)
    return open_, np.clip(hi, None, lim_hi), np.clip(lo, lim_lo, None), close


def generate(n_stocks: int = 700, seed: int = 7, end: str = "2026-10-02") -> None:
    assert config.IS_SYNTHETIC, "must run with EL_DATA_SOURCE=synthetic"
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(config.DATA_START, end)
    # remove a few holidays each year (Lunar New Year week etc.)
    hol = set()
    for yr in range(dates[0].year, dates[-1].year + 1):
        lny = pd.Timestamp(f"{yr}-02-0{rng.integers(1, 6)}")
        hol |= set(pd.bdate_range(lny, lny + pd.Timedelta(days=6)))
        hol |= set(rng.choice(dates[(dates.year == yr)], 4, replace=False))
    dates = pd.DatetimeIndex([d for d in dates if d not in hol])
    T = len(dates)

    # ---- market: regime-switching ---------------------------------------------------
    regime = np.zeros(T, int)
    P = np.array([[0.97, 0.025, 0.005], [0.02, 0.96, 0.02], [0.01, 0.04, 0.95]])
    for t in range(1, T):
        regime[t] = rng.choice(3, p=P[regime[t - 1]])
    mu = np.array([0.0012, 0.0, -0.0013])[regime]
    sd = np.array([0.009, 0.0075, 0.014])[regime]
    mret = mu + sd * rng.standard_t(5, T) / np.sqrt(5 / 3)
    mret = np.clip(mret, -0.09, 0.09)

    # ---- sectors ---------------------------------------------------------------------
    S = len(SECTORS)
    sdrift = np.zeros((T, S))
    for t in range(1, T):
        sdrift[t] = 0.985 * sdrift[t - 1] + rng.normal(0, 0.00025, S)
    sret = sdrift + rng.normal(0, 0.006, (T, S))

    # ---- stocks ----------------------------------------------------------------------
    N = n_stocks
    ids = sorted(set(rng.choice(np.arange(1101, 9960), N - 1, replace=False).astype(str)) - {"3653"})[:N - 1]
    ids = sorted(ids + ["3653"])
    N = len(ids)
    sector = rng.integers(0, S, N)
    sector[ids.index("3653")] = SECTORS.index("電子零組件業")
    beta = rng.uniform(0.5, 1.6, N)
    gamma = rng.uniform(0.6, 1.3, N)
    ivol = rng.uniform(0.010, 0.032, N)
    idio = rng.standard_t(4, (T, N)) / np.sqrt(2) * ivol
    drift = np.zeros((T, N))
    down_beta_mult = np.ones((T, N))
    n_leaders = int(N * 1.2)
    for _ in range(n_leaders):              # planted leader episodes
        i = rng.integers(0, N)
        t0 = rng.integers(60, T - 60)
        L = rng.integers(40, 120)
        emerg = rng.integers(8, 20)
        mu_e = rng.uniform(0.002, 0.004)
        mu_l = rng.uniform(0.003, 0.009)
        drift[t0:t0 + emerg, i] += mu_e
        drift[t0 + emerg:t0 + emerg + L, i] += mu_l
        down_beta_mult[t0:t0 + emerg + L, i] = 0.45
    for _ in range(int(N * 2.0)):           # planted false starts
        i = rng.integers(0, N)
        t0 = rng.integers(60, T - 40)
        k = rng.integers(5, 12)
        drift[t0:t0 + k, i] += rng.uniform(0.003, 0.007)
        drift[t0 + k:t0 + k + 15, i] -= rng.uniform(0.003, 0.006)
    mb = np.where(mret[:, None] < 0, beta[None, :] * down_beta_mult, beta[None, :])
    ret = mb * mret[:, None] + gamma[None, :] * sret[:, sector] + idio + drift
    ret = np.clip(ret, -0.0999, 0.0999)

    # listing / delisting windows
    start_idx = np.zeros(N, int)
    end_idx = np.full(N, T)
    newl = rng.choice(N, 20, replace=False)
    start_idx[newl] = rng.integers(150, T - 300, 20)
    dl_cand = np.setdiff1d(np.arange(N), np.r_[newl, [ids.index("3653")]])
    dl = rng.choice(dl_cand, 15, replace=False)
    end_idx[dl] = rng.integers(400, T - 30, 15)

    # dividends: reference price = prev close - div on ex-date
    div_frac = np.zeros((T, N))
    for yr in range(dates[0].year, dates[-1].year + 1):
        m = np.where((dates.year == yr) & (dates.month.isin([7, 8])))[0]
        if len(m) == 0:
            continue
        payers = rng.random(N) < 0.7
        exd = rng.choice(m, N)
        div_frac[exd[payers], np.where(payers)[0]] = rng.uniform(0.01, 0.05, payers.sum())

    p0 = np.exp(rng.uniform(np.log(15), np.log(400), N))
    close = np.zeros((T, N))
    ref = np.zeros((T, N))
    prev = p0.copy()
    for t in range(T):
        ref[t] = prev * (1 - div_frac[t])
        close[t] = ref[t] * (1 + ret[t])
        prev = close[t]
    gapf = rng.uniform(0.2, 0.6, (T, N))
    o, h, l, c = _ohlc(ref, ret, gapf, rng, ivol[None, :] + 0.004)
    base_val = np.exp(rng.normal(np.log(9e7), 1.0, N))
    value = base_val[None, :] * np.exp(rng.normal(0, 0.35, (T, N))) * (1 + 10 * np.abs(ret))

    cache = FinMindCache(client=None)
    stock_type = np.where(np.arange(N) < int(N * 0.6), "twse", "tpex")
    rows_info = []
    disp_rows = []
    for i, sid in enumerate(ids):
        sl = slice(start_idx[i], end_idx[i])
        df = pd.DataFrame({
            "date": dates[sl], "stock_id": sid,
            "Trading_Volume": (value[sl, i] / c[sl, i]).round(0),
            "Trading_money": value[sl, i].round(0),
            "open": o[sl, i].round(2), "max": h[sl, i].round(2), "min": l[sl, i].round(2),
            "close": c[sl, i].round(2), "spread": (c[sl, i] - ref[sl, i]).round(2),
            "Trading_turnover": (value[sl, i] / 2e5).round(0),
        })
        zero = rng.random(len(df)) < 0.001
        df.loc[zero, ["Trading_Volume", "Trading_money", "open", "max", "min", "close", "spread",
                      "Trading_turnover"]] = 0
        cache.write("TaiwanStockPrice", sid, df, requested_start=config.DATA_START, checked_until=end)
        if i not in dl:
            name = "健策(合成)" if sid == "3653" else f"合成{sid}"
            rows_info.append({"industry_category": SECTORS[sector[i]], "stock_id": sid, "stock_name": name,
                              "type": stock_type[i], "date": end})
        # disposition after big 6-day run-ups
        r6 = pd.Series(ret[sl, i]).rolling(6).sum().to_numpy()
        hits = np.where(r6 > 0.25)[0]
        last = -999
        for k in hits:
            if k - last < 15 or k + 12 >= len(r6):
                continue
            last = k
            ds = dates[sl][k]
            ps, pe = dates[sl][min(k + 1, len(r6) - 1)], dates[sl][min(k + 10, len(r6) - 1)]
            meas = "第一次處置：每五分鐘撮合一次" if rng.random() < 0.7 else "第二次處置：每二十分鐘撮合一次，預收款券"
            disp_rows.append({"date": ds, "stock_id": sid, "stock_name": f"合成{sid}", "disposition_cnt": 1,
                              "condition": "連續六個營業日累積漲幅過大", "measure": meas,
                              "period_start": ps.date().isoformat(), "period_end": pe.date().isoformat()})

    cache.write("TaiwanStockInfo", "_all", pd.DataFrame(rows_info))
    cache.write("TaiwanStockTradingDate", "_all", pd.DataFrame({"date": dates}))
    cache.write("TaiwanStockDelisting", "_all", pd.DataFrame(
        {"date": dates[end_idx[dl]], "stock_id": [ids[i] for i in dl], "stock_name": [f"合成{ids[i]}" for i in dl]}))
    dp = pd.DataFrame(disp_rows)
    for yr, g in dp.groupby(pd.to_datetime(dp["date"]).dt.year):
        cache.write("TaiwanStockDispositionSecuritiesPeriod", f"Y{yr}", g.reset_index(drop=True))

    # index: TAIEX / TPEx OHLC
    for key, scale, lvl in (("IDX_TAIEX", 1.0, 17000.0), ("IDX_TPEx", 1.15, 200.0)):
        r = np.clip(mret * scale + rng.normal(0, 0.001, T), -0.095, 0.095)
        lev = lvl * np.cumprod(1 + r)
        refi = np.r_[lvl, lev[:-1]]
        oi, hi_, li, ci = _ohlc(refi, r, rng.uniform(0.2, 0.6, T), rng, np.full(T, 0.006))
        sid = key.split("_")[1]
        cache.write("TaiwanStockPrice", key, pd.DataFrame({
            "date": dates, "stock_id": sid, "Trading_Volume": 5e9, "Trading_money": 3e11,
            "open": oi.round(2), "max": hi_.round(2), "min": li.round(2), "close": ci.round(2),
            "spread": (ci - refi).round(2), "Trading_turnover": 2e6}))
    cache.manifest["universe_ids"] = ids
    cache.manifest["delisted_ids"] = [ids[i] for i in dl]
    cache.manifest["capabilities"] = {"SYNTHETIC": {"label": "SYNTHETIC", "dataset": "ALL", "status": "SYNTHETIC",
                                                    "note": "離線合成資料，僅供程式測試"}}
    cache.manifest["source"] = "synthetic"
    cache.mark_refresh()
    print(f"synthetic cache written: {N} stocks x {T} days -> {config.RAW_DIR}")


if __name__ == "__main__":
    generate()
