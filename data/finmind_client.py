"""Unified FinMind access layer (Part 2.3).

Every research / pipeline script MUST go through this module. It is the only
place that talks to the FinMind HTTP API and it owns:

* .env authentication (FINMIND_TOKEN, never hard-coded, never printed)
* request / retry / timeout / exponential back-off
* rate-limit (HTTP 402 "upper limit") handling
* incremental download into a local parquet cache (data/cache/raw)
* response validation (status, schema, dtypes, duplicates)
* token scrubbing in every error message and log line

Data flow:  FinMind API -> raw local cache (this module) -> engine.panel (clean PIT)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

API_DATA = "https://api.finmindtrade.com/api/v4/data"
API_DATALIST = "https://api.finmindtrade.com/api/v4/datalist"
API_USER_INFO = "https://api.web.finmindtrade.com/v2/user_info"

# Expected minimum schema per dataset (used for validation; extra columns allowed)
SCHEMAS: dict[str, list[str]] = {
    "TaiwanStockInfo": ["industry_category", "stock_id", "stock_name", "type"],
    "TaiwanStockTradingDate": ["date"],
    "TaiwanStockPrice": ["date", "stock_id", "Trading_Volume", "Trading_money", "open", "max",
                         "min", "close", "spread", "Trading_turnover"],
    "TaiwanStockPriceAdj": ["date", "stock_id", "open", "max", "min", "close"],
    "TaiwanStockTotalReturnIndex": ["date", "stock_id", "price"],
    "TaiwanStockDelisting": ["date", "stock_id"],
    "TaiwanStockDispositionSecuritiesPeriod": ["date", "stock_id"],
    "TaiwanStockDividendResult": ["date", "stock_id"],
    "TaiwanStockCapitalReductionReferencePrice": ["date", "stock_id"],
    "TaiwanStockMarketValue": ["date", "stock_id", "market_value"],
    "TaiwanStockKBar": ["date", "minute", "stock_id", "open", "high", "low", "close", "volume"],
    "TaiwanVariousIndicators5Seconds": ["date"],
    "TaiwanStockEvery5SecondsIndex": ["date"],
    "TaiwanStockSuspended": ["stock_id", "date"],
}


# ---------------------------------------------------------------------------
# Secrets handling
# ---------------------------------------------------------------------------
def load_token(required: bool = True) -> str | None:
    """Read FINMIND_TOKEN from the project-root .env (Part 2.1)."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 python-dotenv，請執行：pip install python-dotenv") from exc
    load_dotenv(ROOT / ".env")
    token = os.getenv("FINMIND_TOKEN")
    if token:
        token = token.strip().strip('"').strip("'")
    if required and not token:
        raise RuntimeError("找不到 FINMIND_TOKEN，請確認專案根目錄的 .env 檔案。")
    return token or None


class _TokenScrubber(logging.Filter):
    """Logging filter: redact the token from every record (Part 2.2)."""

    def __init__(self, token: str | None):
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        if self.token:
            msg = record.getMessage()
            if self.token in msg:
                record.msg = msg.replace(self.token, "***REDACTED***")
                record.args = ()
        return True


def scrub(text: str, token: str | None) -> str:
    if not token or not text:
        return text
    text = text.replace(token, "***REDACTED***")
    # also redact partial leaks (e.g. truncated URLs in exceptions)
    for n in (64, 32, 16):
        if len(token) > n:
            text = text.replace(token[:n], "***REDACTED***")
    return text


def get_logger(token: str | None = None) -> logging.Logger:
    log = logging.getLogger("finmind")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
        fh = logging.FileHandler(config.LOG_DIR / "finmind.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    for h in log.handlers:
        if not any(isinstance(f, _TokenScrubber) for f in h.filters):
            h.addFilter(_TokenScrubber(token))
    return log


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class FinMindError(RuntimeError):
    pass


class FinMindNetworkError(FinMindError):
    """Host unreachable / proxy denial / timeout after retries."""


class FinMindPermissionError(FinMindError):
    """Dataset or query mode not available for this account level."""


class FinMindRateLimitError(FinMindError):
    pass


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
@dataclass
class RequestStats:
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    rows: int = 0


class FinMindClient:
    """Thin, safe HTTP client. Token is only ever sent in the Authorization header."""

    def __init__(self, token: str | None = None, timeout: float = 60.0, max_retries: int = 5,
                 min_interval: float = 0.15, max_rate_wait: float = 3900.0):
        self.token = token if token is not None else load_token(required=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.max_rate_wait = max_rate_wait
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}",
                                     "User-Agent": "emerging-leader-research/1.0"})
        self.log = get_logger(self.token)
        self.stats = RequestStats()
        self._last_call = 0.0

    # -- low level ---------------------------------------------------------
    def _sleep_pacing(self) -> None:
        dt = time.time() - self._last_call
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_call = time.time()

    def _get(self, url: str, params: dict) -> dict:
        backoff = 2.0
        waited_rate = 0.0
        attempt = 0
        while True:
            attempt += 1
            self._sleep_pacing()
            self.stats.requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.ProxyError as exc:
                # Proxy / egress policy denial is not transient -> fail fast with a clear message
                msg = scrub(str(exc), self.token)
                if "403" in msg or attempt > 2:
                    raise FinMindNetworkError(
                        f"網路被拒絕 (proxy/egress) host={requests.utils.urlparse(url).hostname} "
                        f"detail={msg[:240]}") from None
                time.sleep(backoff)
                backoff *= 2
                continue
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt > self.max_retries:
                    raise FinMindNetworkError(
                        f"連線失敗 host={requests.utils.urlparse(url).hostname} "
                        f"type={type(exc).__name__} detail={scrub(str(exc), self.token)[:240]}") from None
                self.stats.retries += 1
                self.log.warning("network error %s, retry %d in %.0fs", type(exc).__name__, attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            status = resp.status_code
            try:
                payload = resp.json()
            except ValueError:
                payload = {"msg": resp.text[:300], "status": status}
            msg = scrub(str(payload.get("msg", "")), self.token) if isinstance(payload, dict) else ""

            if status == 200 and isinstance(payload, dict) and payload.get("status", 200) == 200:
                return payload
            if status == 402 or "upper limit" in msg.lower():
                self.stats.rate_limited += 1
                if waited_rate >= self.max_rate_wait:
                    raise FinMindRateLimitError(f"FinMind 請求次數上限，已等待 {waited_rate:.0f}s: {msg}")
                wait = 120.0
                self.log.warning("FinMind rate limit reached (%s); sleeping %.0fs", msg[:80], wait)
                time.sleep(wait)
                waited_rate += wait
                continue
            if status in (500, 502, 503, 504) and attempt <= self.max_retries:
                self.stats.retries += 1
                self.log.warning("server %s, retry %d in %.0fs", status, attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            low = msg.lower()
            if status in (400, 401, 403) and any(k in low for k in ("level", "sponsor", "backer", "permission",
                                                                    "權限", "升級", "register")):
                raise FinMindPermissionError(f"權限不足 status={status} msg={msg[:200]}")
            if status == 401 or "token" in low and "invalid" in low:
                raise FinMindError(f"Token 驗證失敗 status={status} msg={msg[:200]}")
            raise FinMindError(f"FinMind 錯誤 status={status} msg={msg[:200]}")

    # -- public API --------------------------------------------------------
    def user_info(self) -> dict:
        """Account info (request quota). Token travels in header + required query param."""
        payload = self._get(API_USER_INFO, {"token": self.token})
        return {k: v for k, v in payload.items() if "token" not in k.lower()}

    def datalist(self, dataset: str) -> list[str]:
        payload = self._get(API_DATALIST, {"dataset": dataset})
        data = payload.get("data", [])
        return [str(x) for x in data] if isinstance(data, list) else []

    def fetch(self, dataset: str, data_id: str | None = None, start_date: str | None = None,
              end_date: str | None = None, validate: bool = True) -> pd.DataFrame:
        params: dict = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        payload = self._get(API_DATA, params)
        data = payload.get("data", [])
        df = pd.DataFrame(data)
        self.stats.rows += len(df)
        if validate:
            df = validate_frame(dataset, df)
        return df


def validate_frame(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    """Schema / dtype / duplicate validation of a FinMind response."""
    if df.empty:
        return df
    need = SCHEMAS.get(dataset, [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise FinMindError(f"{dataset} 缺少欄位 {missing}; got {list(df.columns)[:20]}")
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str).str.strip()
    if "date" in df.columns and dataset not in ("TaiwanVariousIndicators5Seconds",):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    num_cols = [c for c in ("Trading_Volume", "Trading_money", "open", "max", "min", "close", "spread",
                            "Trading_turnover", "price", "market_value", "high", "low", "volume")
                if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keys = [k for k in ("date", "stock_id", "minute") if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Incremental cache
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FinMindCache:
    """Parquet cache keyed by (dataset, key). Incremental updates only fetch new dates."""

    def __init__(self, client: FinMindClient | None = None, raw_dir: Path = config.RAW_DIR,
                 manifest_path: Path = config.MANIFEST_PATH):
        self.client = client
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    # -- manifest ------------------------------------------------------------
    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"datasets": {}, "capabilities": {}, "last_refresh_utc": None, "source": config.DATA_SOURCE}

    def save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def entry(self, dataset: str, key: str) -> dict | None:
        return self.manifest["datasets"].get(dataset, {}).get(key)

    def _set_entry(self, dataset: str, key: str, **kw) -> None:
        self.manifest["datasets"].setdefault(dataset, {})[key] = {**(self.entry(dataset, key) or {}), **kw}

    # -- paths -----------------------------------------------------------------
    def path(self, dataset: str, key: str) -> Path:
        d = self.raw_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.parquet"

    def read(self, dataset: str, key: str) -> pd.DataFrame:
        p = self.path(dataset, key)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def read_all(self, dataset: str, keys: Iterable[str] | None = None) -> pd.DataFrame:
        d = self.raw_dir / dataset
        if not d.exists():
            return pd.DataFrame()
        files = sorted(d.glob("*.parquet")) if keys is None else [self.path(dataset, k) for k in keys]
        frames = [pd.read_parquet(f) for f in files if f.exists()]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def write(self, dataset: str, key: str, df: pd.DataFrame, **meta) -> None:
        p = self.path(dataset, key)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        info = {"rows": int(len(df)), "updated_utc": _utcnow(), **meta}
        if "date" in df.columns and len(df):
            info.setdefault("min_date", str(pd.to_datetime(df["date"]).min().date()))
            info.setdefault("max_date", str(pd.to_datetime(df["date"]).max().date()))
        self._set_entry(dataset, key, **info)

    # -- fetch helpers -------------------------------------------------------------
    def _require_client(self) -> FinMindClient:
        if self.client is None:
            raise FinMindError("FinMindCache 未設定 client（離線模式只能讀取本地快取）")
        return self.client

    def snapshot(self, dataset: str, key: str = "_all", max_age_hours: float = 20.0, **params) -> pd.DataFrame:
        """Whole-table datasets (e.g. TaiwanStockInfo). Re-download only if stale."""
        ent = self.entry(dataset, key)
        if ent and ent.get("updated_utc"):
            age = (datetime.now(timezone.utc) - datetime.strptime(ent["updated_utc"], "%Y-%m-%dT%H:%M:%SZ")
                   .replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if age < max_age_hours and self.path(dataset, key).exists():
                return self.read(dataset, key)
        df = self._require_client().fetch(dataset, **params)
        if not df.empty:
            self.write(dataset, key, df)
        return df if not df.empty else self.read(dataset, key)

    def series(self, dataset: str, data_id: str, start: str, end: str | None = None,
               overlap_days: int = 7, key: str | None = None) -> pd.DataFrame:
        """Incremental per-id time series: only fetches (cached_max_date - overlap, end]."""
        key = key or data_id
        end = end or date.today().isoformat()
        old = self.read(dataset, key)
        ent = self.entry(dataset, key) or {}
        fetch_start = start
        if not old.empty and ent.get("max_date"):
            cached_min = ent.get("requested_start", ent.get("min_date"))
            if cached_min and cached_min <= start:
                fetch_start = (pd.Timestamp(ent["max_date"]) - timedelta(days=overlap_days)).date().isoformat()
            if ent.get("checked_until") and ent["checked_until"] >= end:
                return old
        new = self._require_client().fetch(dataset, data_id=data_id, start_date=fetch_start, end_date=end)
        if old.empty:
            merged = new
        elif new.empty:
            merged = old
        else:
            keys = [k for k in ("date", "stock_id", "minute") if k in new.columns]
            merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=keys, keep="last")
            merged = merged.sort_values(keys).reset_index(drop=True)
        self.write(dataset, key, merged, requested_start=min(start, ent.get("requested_start", start)),
                   checked_until=end)
        return merged

    def by_dates(self, dataset: str, dates: list[str], key_prefix: str = "bulk") -> int:
        """Bulk mode: one request per trading date without data_id (sponsor/backer level).

        Stored in yearly partitions <dataset>/<prefix>_<YYYY>.parquet. Returns #dates fetched.
        """
        client = self._require_client()
        done = set(self.manifest.get("bulk_dates", {}).get(dataset, []))
        todo = [d for d in dates if d not in done]
        by_year: dict[str, list[pd.DataFrame]] = {}
        n = 0
        for i, d in enumerate(todo):
            df = client.fetch(dataset, start_date=d, end_date=d)
            by_year.setdefault(d[:4], []).append(df)
            done.add(d)
            n += 1
            if len(by_year.get(d[:4], [])) >= 20 or i == len(todo) - 1:
                for yr, frames in by_year.items():
                    frames = [f for f in frames if not f.empty]
                    if frames:
                        key = f"{key_prefix}_{yr}"
                        old = self.read(dataset, key)
                        merged = pd.concat([old] + frames, ignore_index=True) if not old.empty else pd.concat(frames)
                        merged = merged.drop_duplicates(subset=["date", "stock_id"], keep="last")
                        self.write(dataset, key, merged.sort_values(["date", "stock_id"]).reset_index(drop=True))
                by_year = {}
                self.manifest.setdefault("bulk_dates", {})[dataset] = sorted(done)
                self.save_manifest()
                client.log.info("%s bulk %d/%d dates (last %s)", dataset, i + 1, len(todo), d)
        return n

    def mark_refresh(self) -> None:
        self.manifest["last_refresh_utc"] = _utcnow()
        self.save_manifest()


# ---------------------------------------------------------------------------
# Capability probing (Part 2.5: do not assume a dataset exists)
# ---------------------------------------------------------------------------
PROBES = [
    # (label, dataset, data_id, days_back, note)
    ("股票基本資訊", "TaiwanStockInfo", None, None, "上市/上櫃/興櫃清單與產業別"),
    ("交易日曆", "TaiwanStockTradingDate", None, None, ""),
    ("個股日線 (單檔)", "TaiwanStockPrice", "2330", 10, "未還原，含 spread(漲跌價差)"),
    ("個股日線 (全市場單日 bulk)", "TaiwanStockPrice", None, 3, "需 backer/sponsor 等級"),
    ("還原股價", "TaiwanStockPriceAdj", "2330", 10, ""),
    ("加權指數日線 (TAIEX)", "TaiwanStockPrice", "TAIEX", 10, ""),
    ("櫃買指數日線 (TPEx)", "TaiwanStockPrice", "TPEx", 10, ""),
    ("報酬指數", "TaiwanStockTotalReturnIndex", "TAIEX", 10, ""),
    ("下市櫃", "TaiwanStockDelisting", None, None, "survivorship"),
    ("處置股票", "TaiwanStockDispositionSecuritiesPeriod", None, 60, ""),
    ("除權息結果", "TaiwanStockDividendResult", None, 60, ""),
    ("減資參考價", "TaiwanStockCapitalReductionReferencePrice", None, 365, ""),
    ("市值", "TaiwanStockMarketValue", "2330", 10, ""),
    ("暫停交易", "TaiwanStockSuspended", None, 60, ""),
    ("分K (KBar)", "TaiwanStockKBar", "2330", 1, "需 sponsor 等級"),
    ("加權指數 5 秒", "TaiwanVariousIndicators5Seconds", None, 1, ""),
    ("類股指數 5 秒", "TaiwanStockEvery5SecondsIndex", None, 1, ""),
]


def probe_capabilities(client: FinMindClient, ref_date: str | None = None) -> list[dict]:
    """Try a tiny request per dataset and record what this account can actually use."""
    ref = pd.Timestamp(ref_date or date.today().isoformat())
    out = []
    for label, ds, did, back, note in PROBES:
        params: dict = {}
        if back is not None:
            # walk back to find a trading day for single-day probes
            params["start_date"] = (ref - timedelta(days=max(back, 7))).date().isoformat()
            params["end_date"] = ref.date().isoformat()
            if back <= 3:
                params["start_date"] = (ref - timedelta(days=6)).date().isoformat()
        rec = {"label": label, "dataset": ds, "data_id": did or "", "note": note}
        try:
            if ds in ("TaiwanStockKBar", "TaiwanVariousIndicators5Seconds", "TaiwanStockEvery5SecondsIndex") \
                    or (ds == "TaiwanStockPrice" and did is None):
                # single-day datasets: try the most recent weekdays
                df = pd.DataFrame()
                for k in range(0, 8):
                    d = (ref - timedelta(days=k))
                    if d.weekday() >= 5:
                        continue
                    df = client.fetch(ds, data_id=did, start_date=d.date().isoformat(),
                                      end_date=d.date().isoformat(), validate=False)
                    if not df.empty:
                        break
            else:
                df = client.fetch(ds, data_id=did, validate=False, **params)
            rec.update(status="OK" if not df.empty else "EMPTY", rows=len(df),
                       columns=",".join(map(str, df.columns[:15])))
        except FinMindPermissionError as e:
            rec.update(status="NO_PERMISSION", rows=0, columns="", error=str(e)[:200])
        except FinMindNetworkError as e:
            rec.update(status="NETWORK_BLOCKED", rows=0, columns="", error=str(e)[:200])
        except FinMindError as e:
            rec.update(status="ERROR", rows=0, columns="", error=str(e)[:200])
        out.append(rec)
    return out


def connectivity_check() -> dict:
    """Quick end-to-end check used by the CLI and the dashboard (never prints the token)."""
    token = load_token(required=False)
    res = {"token_loaded": bool(token), "token_hint": ("****" + token[-4:]) if token else None}
    if not token:
        res["status"] = "NO_TOKEN"
        return res
    client = FinMindClient(token=token, max_retries=1)
    try:
        df = client.fetch("TaiwanStockPrice", data_id="2330",
                          start_date=(date.today() - timedelta(days=10)).isoformat(),
                          end_date=date.today().isoformat())
        res.update(status="OK", rows=len(df), last_date=str(df["date"].max().date()) if len(df) else None)
    except FinMindError as e:
        res.update(status=type(e).__name__, error=str(e)[:300])
    try:
        res["user_info"] = client.user_info()
    except FinMindError as e:
        res["user_info_error"] = str(e)[:200]
    return res


if __name__ == "__main__":
    print(json.dumps(connectivity_check(), ensure_ascii=False, indent=1, default=str))
