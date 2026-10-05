"""Dashboard data access — reads ONLY local files (CSV / parquet / manifest). No API calls, no token."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = os.getenv("EL_DATA_SOURCE", "finmind").lower()
OUT = ROOT / ("outputs_synthetic" if SOURCE == "synthetic" else "outputs")
SIG = OUT / "signals"
CACHE = ROOT / "data" / ("cache_synthetic" if SOURCE == "synthetic" else "cache")
DOCS = ROOT if SOURCE != "synthetic" else OUT

STATE_ZH = {"WATCH": "觀察", "EMERGING": "Emerging", "PROBE_READY": "可試單", "PROBED": "已試單",
            "WAIT_CONFIRM": "等待確認", "CONFIRMED": "趨勢確認", "ADD_READY": "可加碼", "FULL": "正式持股",
            "FAILED": "試單失敗", "EXIT": "出場"}
STATE_ORDER = ["正式持股", "可加碼", "趨勢確認", "等待確認", "已試單", "可試單", "Emerging", "觀察", "試單失敗", "出場"]


def available() -> bool:
    return (SIG / "meta.json").exists()


@st.cache_data(show_spinner=False)
def meta() -> dict:
    m = json.loads((SIG / "meta.json").read_text(encoding="utf-8")) if (SIG / "meta.json").exists() else {}
    man = CACHE / "manifest.json"
    if man.exists():
        mm = json.loads(man.read_text(encoding="utf-8"))
        m["finmind_last_refresh_utc"] = mm.get("last_refresh_utc", m.get("finmind_last_refresh_utc"))
        m["capabilities"] = mm.get("capabilities", {})
    return m


@st.cache_data(show_spinner=False)
def dates() -> pd.DatetimeIndex:
    mk = pd.read_parquet(SIG / "market.parquet", columns=["date"])
    return pd.DatetimeIndex(mk["date"])


@st.cache_data(show_spinner=False)
def market() -> pd.DataFrame:
    return pd.read_parquet(SIG / "market.parquet").set_index("date")


@st.cache_data(show_spinner=False)
def stock_info() -> pd.DataFrame:
    return pd.read_parquet(SIG / "stock_info.parquet")


@st.cache_data(show_spinner=False, max_entries=20)
def signals_on(d: pd.Timestamp) -> pd.DataFrame:
    return pd.read_parquet(SIG / "signals_daily.parquet", filters=[("date", "==", pd.Timestamp(d))])


@st.cache_data(show_spinner=False, max_entries=30)
def signals_stock(sid: str) -> pd.DataFrame:
    return pd.read_parquet(SIG / "signals_daily.parquet", filters=[("stock_id", "==", sid)]).set_index("date")


@st.cache_data(show_spinner=False, max_entries=30)
def prices_stock(sid: str) -> pd.DataFrame:
    return pd.read_parquet(SIG / "prices.parquet", filters=[("stock_id", "==", sid)]).set_index("date")


@st.cache_data(show_spinner=False, max_entries=30)
def sector_stock(sid: str) -> pd.Series:
    df = pd.read_parquet(SIG / "sector_level.parquet", filters=[("stock_id", "==", sid)])
    return df.set_index("date")["sec_level"]


@st.cache_data(show_spinner=False)
def states() -> pd.DataFrame:
    df = pd.read_parquet(SIG / "states_daily.parquet")
    df["state_zh"] = df["state"].map(STATE_ZH)
    return df


@st.cache_data(show_spinner=False)
def events() -> pd.DataFrame:
    return pd.read_parquet(SIG / "campaign_events.parquet")


@st.cache_data(show_spinner=False)
def campaigns() -> pd.DataFrame:
    return pd.read_parquet(SIG / "campaigns_live.parquet")


@st.cache_data(show_spinner=False)
def portfolio() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eq = pd.read_parquet(SIG / "portfolio_live_equity.parquet").set_index("date")
    pos = pd.read_parquet(SIG / "portfolio_live_positions.parquet")
    tr = pd.read_parquet(SIG / "portfolio_live_trades.parquet")
    return eq, pos, tr


@st.cache_data(show_spinner=False)
def disposition() -> pd.DataFrame:
    f = SIG / "disposition.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    f = OUT / name
    return pd.read_csv(f, encoding="utf-8-sig") if f.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def case() -> tuple[pd.DataFrame, pd.DataFrame]:
    tl = SIG / "case_timeline.parquet"
    ip = SIG / "case_intraday.parquet"
    return (pd.read_parquet(tl) if tl.exists() else pd.DataFrame(),
            pd.read_parquet(ip) if ip.exists() else pd.DataFrame())


def doc(name: str) -> str:
    f = DOCS / name
    return f.read_text(encoding="utf-8") if f.exists() else f"_{name} 尚未產生_"


def frozen_config() -> dict:
    return meta().get("config", {})


@st.cache_data(show_spinner=False)
def names() -> dict:
    try:
        si = stock_info()
        col = "name" if "name" in si.columns else si.columns[1]
        key = "stock_id" if "stock_id" in si.columns else si.index.name
        return (si.set_index(key)[col] if key in si.columns else si[col]).astype(str).to_dict()
    except Exception:   # noqa: BLE001
        return {}
