"""Cached / incremental FinMind download orchestrator.

    python -m data.download --probe            # only test datasets (tiny requests)
    python -m data.download                    # incremental update (default)
    python -m data.download --case-intraday    # also fetch 3653 minute data for the case study

All HTTP goes through data.finmind_client. Nothing here prints the token.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from data.finmind_client import (FinMindCache, FinMindClient, FinMindError,  # noqa: E402
                                 FinMindNetworkError, FinMindPermissionError, probe_capabilities)

INDEX_IDS = ["TAIEX", "TPEx"]
PROXY_ETFS = ["0050", "006201"]      # tradable fallbacks if index OHLC is unavailable
NON_EQUITY_INDUSTRIES = {"ETF", "ETN", "受益證券", "存託憑證", "大盤", "Index", "所有證券", "上櫃指數股票型基金(ETF)",
                         "指數投資證券(ETN)", "上櫃ETN", "創新版股票", "創新板股票"}


def is_common_stock_id(sid: str) -> bool:
    """TW common stocks: 4-digit numeric id not starting with 0 (00xx = ETF)."""
    return bool(re.fullmatch(r"[1-9]\d{3}", str(sid)))


def common_stock_ids(info: pd.DataFrame) -> list[str]:
    if info.empty:
        return []
    df = info.copy()
    df = df[df["stock_id"].map(is_common_stock_id)]
    if "type" in df.columns:
        df = df[df["type"].isin(["twse", "tpex"])]          # 興櫃 (emerging) excluded: no price limits
    df = df[~df["industry_category"].isin(NON_EQUITY_INDUSTRIES)]
    return sorted(df["stock_id"].unique())


def trading_dates(cache: FinMindCache, start: str, end: str) -> list[str]:
    cal = cache.snapshot("TaiwanStockTradingDate")
    if cal.empty:
        return [d.date().isoformat() for d in pd.bdate_range(start, end)]
    d = pd.to_datetime(cal["date"])
    d = d[(d >= start) & (d <= end)]
    return sorted(x.date().isoformat() for x in d)


def run_probe(client: FinMindClient, cache: FinMindCache) -> list[dict]:
    caps = probe_capabilities(client)
    cache.manifest["capabilities"] = {r["label"]: r for r in caps}
    cache.manifest["capabilities_checked_utc"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    cache.save_manifest()
    return caps


def _cap_ok(cache: FinMindCache, label: str) -> bool | None:
    rec = cache.manifest.get("capabilities", {}).get(label)
    if rec is None:
        return None
    return rec.get("status") == "OK"


def download(full_start: str = config.DATA_START, end: str | None = None, case_intraday: bool = False,
             max_stocks: int | None = None, log=print) -> dict:
    end = end or date.today().isoformat()
    client = FinMindClient()
    cache = FinMindCache(client)
    summary: dict = {"end": end}

    if not cache.manifest.get("capabilities"):
        log("[probe] 檢查 FinMind datasets 可用性 ...")
        run_probe(client, cache)

    # 1. reference tables -------------------------------------------------------------
    info = cache.snapshot("TaiwanStockInfo")
    cache.snapshot("TaiwanStockTradingDate")
    try:
        delist = cache.snapshot("TaiwanStockDelisting")
    except FinMindError as e:
        log(f"[warn] TaiwanStockDelisting 無法取得: {e}")
        delist = pd.DataFrame()
    ids = common_stock_ids(info)
    delisted_ids: list[str] = []
    if not delist.empty:
        dl = delist.copy()
        dl["date"] = pd.to_datetime(dl["date"])
        dl = dl[(dl["date"] >= full_start) & dl["stock_id"].map(is_common_stock_id)]
        delisted_ids = sorted(set(dl["stock_id"]) - set(ids))
    # datalist (all ids that ever had price data) - best-effort survivorship coverage
    try:
        all_ids = [s for s in client.datalist("TaiwanStockPrice") if is_common_stock_id(s)]
        info_ids = set(info["stock_id"]) if not info.empty else set()
        extra = sorted(set(all_ids) - info_ids - set(delisted_ids))
        cache.manifest["datalist_extra_ids_count"] = len(extra)   # audit only (mostly pre-2021 delistings)
    except FinMindError:
        extra = []
    universe_ids = sorted(set(ids) | set(delisted_ids))
    if max_stocks:
        universe_ids = universe_ids[:max_stocks]
    summary.update(n_listed_common=len(ids), n_delisted=len(delisted_ids), n_extra=len(extra),
                   n_download_ids=len(universe_ids))
    cache.manifest["universe_ids"] = universe_ids
    cache.manifest["delisted_ids"] = delisted_ids
    cache.save_manifest()

    # 2. daily prices -------------------------------------------------------------------
    #    history: one request per stock (small payloads); afterwards new trading days are
    #    appended with one all-market request per day when the account allows it.
    bulk_ok = _cap_ok(cache, "個股日線 (全市場單日 bulk)")
    have = {sid for sid in universe_ids if cache.path("TaiwanStockPrice", sid).exists()}
    missing = [sid for sid in universe_ids if sid not in have]
    summary["price_mode"] = "per_stock_history+bulk_incremental" if bulk_ok else "per_stock"
    log(f"[price] {len(missing)} ids need history; {len(have)} cached")
    n_fail = 0
    for i, sid in enumerate(missing):
        try:
            cache.series("TaiwanStockPrice", sid, full_start, end)
        except FinMindNetworkError:
            raise
        except FinMindError as e:
            n_fail += 1
            log(f"[warn] {sid}: {e}")
        if (i + 1) % 100 == 0:
            cache.save_manifest()
            log(f"[price] history {i + 1}/{len(missing)} requests={client.stats.requests}")
    cache.save_manifest()
    if have:
        if bulk_ok:
            summary["bulk_days"] = bulk_incremental(cache, sorted(have), end, log=log)
        else:
            for i, sid in enumerate(sorted(have)):
                try:
                    cache.series("TaiwanStockPrice", sid, full_start, end)
                except FinMindError as e:
                    n_fail += 1
                    log(f"[warn] {sid}: {e}")
    summary["price_failures"] = n_fail
    cache.save_manifest()

    # 2b. FinMind adjusted prices (primary daily-return source; `spread` is 0 on ex-dividend days)
    adj_ok = _cap_ok(cache, "還原股價") is not False
    if adj_ok:
        have_a = {sid for sid in universe_ids if cache.path("TaiwanStockPriceAdj", sid).exists()}
        miss_a = [sid for sid in universe_ids if sid not in have_a]
        log(f"[adj] {len(miss_a)} ids need TaiwanStockPriceAdj history")
        for i, sid in enumerate(miss_a):
            try:
                cache.series("TaiwanStockPriceAdj", sid, full_start, end)
            except FinMindNetworkError:
                raise
            except FinMindError as e:
                log(f"[warn] adj {sid}: {e}")
            if (i + 1) % 200 == 0:
                cache.save_manifest()
                log(f"[adj] history {i + 1}/{len(miss_a)}")
        if have_a and bulk_ok:
            summary["adj_bulk_days"] = bulk_incremental(cache, sorted(have_a), end, log=log,
                                                        dataset="TaiwanStockPriceAdj")
        cache.save_manifest()

    # 3. market indices + proxies ------------------------------------------------------
    for sid in INDEX_IDS + PROXY_ETFS:
        try:
            cache.series("TaiwanStockPrice", sid, full_start, end, key=f"IDX_{sid}")
        except FinMindError as e:
            log(f"[warn] index {sid}: {e}")
    for sid in INDEX_IDS:
        try:
            cache.series("TaiwanStockTotalReturnIndex", sid, full_start, end)
        except FinMindError as e:
            log(f"[warn] TotalReturnIndex {sid}: {e}")

    # 3b. month-end market value snapshots (turnover %) ---------------------------------
    try:
        summary["market_value_months"] = market_value_snapshots(cache, full_start, end, log=log)
    except FinMindError as e:
        log(f"[warn] market value: {e}")

    # 4. event tables (range queries without data_id, chunked by year) ------------------
    for ds in ("TaiwanStockDispositionSecuritiesPeriod", "TaiwanStockDividendResult",
               "TaiwanStockCapitalReductionReferencePrice", "TaiwanStockSuspended"):
        for yr in range(int(full_start[:4]), int(end[:4]) + 1):
            s = max(f"{yr}-01-01", full_start)
            e_ = min(f"{yr}-12-31", end)
            key = f"Y{yr}"
            ent = cache.entry(ds, key)
            if ent and ent.get("checked_until", "") >= e_ and yr < int(end[:4]):
                continue
            try:
                df = client.fetch(ds, start_date=s, end_date=e_)
                cache.write(ds, key, df, checked_until=e_)
            except FinMindPermissionError as e:
                log(f"[limit] {ds}: {e}")
                break
            except FinMindError as e:
                log(f"[warn] {ds} {yr}: {e}")
                break
        cache.save_manifest()

    # 5. optional: case-study intraday data ------------------------------------------------
    if case_intraday:
        summary["intraday"] = download_case_intraday(client, cache, log=log)

    cache.mark_refresh()
    summary["requests"] = client.stats.requests
    summary["rows"] = client.stats.rows
    return summary


def bulk_incremental(cache: FinMindCache, ids: list[str], end: str, log=print,
                     dataset: str = "TaiwanStockPrice") -> int:
    """Append new trading days to per-stock caches using one all-market request per day."""
    maxd = []
    for sid in ids:
        ent = cache.entry(dataset, sid) or {}
        if ent.get("max_date"):
            maxd.append(ent["max_date"])
    if not maxd:
        return 0
    # the latest date most stocks already have (delisted names stop earlier)
    last = pd.Series(maxd).value_counts().index[0]
    days = [d for d in trading_dates(cache, last, end) if d > last]
    if not days:
        return 0
    frames = []
    for d in days:
        df = cache.client.fetch(dataset, start_date=d, end_date=d)
        if not df.empty:
            frames.append(df[df["stock_id"].isin(set(ids))])
        log(f"[{dataset}] bulk day {d}: {len(df)} rows")
    if not frames:
        return 0
    new = pd.concat(frames, ignore_index=True)
    for sid, g in new.groupby("stock_id"):
        old = cache.read(dataset, sid)
        merged = pd.concat([old, g], ignore_index=True).drop_duplicates(["date", "stock_id"], keep="last")
        ent = cache.entry(dataset, sid) or {}
        cache.write(dataset, sid, merged.sort_values("date").reset_index(drop=True),
                    requested_start=ent.get("requested_start", config.DATA_START), checked_until=end,
                    max_date=str(pd.to_datetime(merged["date"]).max().date()))
    for sid in ids:   # mark checked even if no new rows (suspended / delisted)
        ent = cache.entry(dataset, sid)
        if ent:
            ent["checked_until"] = end
    cache.save_manifest()
    return len(days)


def market_value_snapshots(cache: FinMindCache, start: str, end: str, log=print) -> int:
    """Month-end market value for every stock (one all-market request per month) -> turnover %."""
    cal = trading_dates(cache, start, end)
    if not cal:
        return 0
    s = pd.Series(pd.to_datetime(cal))
    month_ends = s.groupby(s.dt.to_period("M")).max().dt.date.astype(str).tolist()
    n = 0
    for d in month_ends:
        key = f"ME_{d}"
        if cache.path("TaiwanStockMarketValue", key).exists():
            continue
        try:
            df = cache.client.fetch("TaiwanStockMarketValue", start_date=d, end_date=d)
        except FinMindError as e:
            log(f"[warn] MarketValue {d}: {e}")
            break
        if not df.empty:
            df = df[df["stock_id"].map(is_common_stock_id)]
            cache.write("TaiwanStockMarketValue", key, df)
            n += 1
    cache.save_manifest()
    return n


def download_case_intraday(client: FinMindClient, cache: FinMindCache, symbol: str = config.CASE_SYMBOL,
                           period: tuple[str, str] = config.CASE_PERIOD, peers: list[str] | None = None,
                           log=print) -> dict:
    """Minute bars for the case-study stock (+ optional peers) and 5-second index data."""
    dates = trading_dates(cache, period[0], min(period[1], date.today().isoformat()))
    res = {"kbar_days": 0, "index5s_days": 0, "sector5s_days": 0}
    for ds, did, kp, cnt in (("TaiwanStockKBar", symbol, f"{symbol}", "kbar_days"),
                             ("TaiwanVariousIndicators5Seconds", None, "TAIEX5S", "index5s_days"),
                             ("TaiwanStockEvery5SecondsIndex", None, "SECTOR5S", "sector5s_days")):
        for sid in ([did] + (peers or [] if ds == "TaiwanStockKBar" else [])):
            for d in dates:
                key = f"{kp if sid == did else sid}_{d}"
                if cache.path(ds, key).exists():
                    res[cnt] += 1
                    continue
                try:
                    df = client.fetch(ds, data_id=sid, start_date=d, end_date=d, validate=False)
                except FinMindPermissionError as e:
                    log(f"[limit] {ds}: {e}")
                    break
                except FinMindError as e:
                    log(f"[warn] {ds} {d}: {e}")
                    continue
                if not df.empty:
                    cache.write(ds, key, df)
                    res[cnt] += 1
            cache.save_manifest()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只檢查 dataset 可用性")
    ap.add_argument("--start", default=config.DATA_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--case-intraday", action="store_true")
    ap.add_argument("--max-stocks", type=int, default=None, help="smoke test: 只下載前 N 檔")
    a = ap.parse_args()
    if config.IS_SYNTHETIC:
        raise SystemExit("EL_DATA_SOURCE=synthetic：請改用 python -m data.synthetic 產生測試資料")
    if a.probe:
        client = FinMindClient()
        cache = FinMindCache(client)
        caps = run_probe(client, cache)
        for r in caps:
            print(f"{r['status']:16s} {r['dataset']:42s} {r['data_id']:7s} rows={r.get('rows', 0):6} {r['label']}")
        return
    s = download(a.start, a.end, a.case_intraday, a.max_stocks)
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
