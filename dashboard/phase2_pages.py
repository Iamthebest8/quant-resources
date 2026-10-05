"""Phase-2 dashboard pages: Momentum Long / Weinstein Long / Weinstein Short / High R/R Radar / Intraday Replay /
Phase-2 verdicts. Reads ONLY local files written by pipeline.phase2* (no API, no token)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

P2 = D.OUT / "phase2"
C_UP, C_DN, C_MUTED, C_BLUE, C_ORG = "#e34948", "#008300", "#8a8986", "#2a78d6", "#eb6834"
STAGE_COL = {1: "rgba(138,137,134,0.18)", 2: "rgba(227,73,72,0.15)", 3: "rgba(235,104,52,0.18)",
             4: "rgba(0,131,0,0.15)"}
STAGE_ZH = {0: "—", 1: "Stage 1 打底", 2: "Stage 2 上升", 3: "Stage 3 做頭", 4: "Stage 4 下跌"}
WIN_ZH = {"PRE_2020_2022": "Pre 2020–2022", "DISCOVERY": "Discovery 2023–2024", "STRICT_OOS": "Strict OOS 2025–2026",
          "EXTENDED": "Extended 2024–2026", "FULL_2023_2026": "Full 2023–2026"}


def _csv(name: str, root: bool = True) -> pd.DataFrame:
    path = (D.DOCS if root else P2) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"stock_id": str})


@st.cache_data(show_spinner=False)
def _weekly() -> pd.DataFrame:
    f = P2 / "dash_weekly.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _pkl(name: str):
    f = P2 / name
    return pd.read_pickle(f) if f.exists() else None


def _missing(what: str):
    st.info(f"尚未產生 {what}。請執行 `python -m pipeline.phase2 all`（見 README_DASHBOARD.md）。")


def _fmt(df: pd.DataFrame, pct_cols=(), num_cols=()) -> pd.DataFrame:
    df = df.copy()
    for c in pct_cols:
        if c in df:
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")
    for c in num_cols:
        if c in df:
            df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x:,.2f}")
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return df


METRIC_COLS = ["n", "win_rate", "pf", "payoff", "ev", "avg_r", "ge20", "ge30", "ge40", "top5pct_share",
               "pnl_per_100_slot_days", "stop_dist_pct_med", "stop_dist_atr_med", "median_hold"]


def _results_table(df: pd.DataFrame, keys: list[str], windows=("DISCOVERY", "STRICT_OOS")):
    if df.empty:
        return _missing("結果檔")
    sub = df[df["window"].isin(windows)].copy()
    sub["window"] = sub["window"].map(WIN_ZH).fillna(sub["window"])
    cols = keys + ["window"] + [c for c in METRIC_COLS if c in sub]
    st.dataframe(_fmt(sub[cols], pct_cols=("win_rate", "ev", "ge20", "ge30", "ge40", "top5pct_share",
                                           "stop_dist_pct_med"),
                      num_cols=("pf", "payoff", "avg_r", "pnl_per_100_slot_days", "stop_dist_atr_med")),
                 hide_index=True, use_container_width=True)


def weekly_chart(sid: str, name: str = ""):
    W = _weekly()
    if W.empty:
        return _missing("週線資料 dash_weekly.parquet")
    w = W[W["stock_id"] == sid].sort_values("date")
    if w.empty:
        st.warning("沒有這檔股票的週線資料")
        return
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=(f"{sid} {name} 週線 + 30 週 MA（Stage 色塊）", "成交量（週日均）",
                                        "Mansfield RS（RESEARCH 定義：RS/SMA52−1）"))
    # stage shading
    s = w["stage"].fillna(0).astype(int).to_numpy()
    d = w["date"].to_numpy()
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if s[start] in STAGE_COL:
                fig.add_vrect(x0=d[start], x1=d[min(i, len(s) - 1)], fillcolor=STAGE_COL[s[start]], line_width=0,
                              row=1, col=1)
            start = i
    fig.add_trace(go.Candlestick(x=w["date"], open=w["close"].shift(1), high=w["high"], low=w["low"],
                                 close=w["close"], increasing_line_color=C_UP, decreasing_line_color=C_DN,
                                 name="週K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=w["date"], y=w["ma30w"], name="30 週 MA", line=dict(color=C_BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w["date"], y=w["ma10w"], name="10 週 MA", line=dict(color=C_ORG, width=1, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=w["date"], y=w["vol"], name="量", marker_color=C_MUTED), row=2, col=1)
    fig.add_trace(go.Scatter(x=w["date"], y=w["mrs"], name="mRS", line=dict(color=C_BLUE)), row=3, col=1)
    fig.add_hline(y=0, line_color=C_MUTED, row=3, col=1)
    fig.update_layout(height=640, xaxis_rangeslider_visible=False, showlegend=True, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    last = w.iloc[-1]
    st.caption(f"最新週（{pd.Timestamp(last['date']).date()}）：{STAGE_ZH.get(int(last['stage']), '—')}；"
               "色塊：灰=Stage1、紅=Stage2、橘=Stage3、綠=Stage4（台股紅漲綠跌）。價格為還原價。")


def page_momentum():
    st.header("🚀 Momentum Long（凍結 MOMENTUM_LONG_BASELINE = V1）")
    st.markdown("Phase 1 的 Emerging Leader 策略原封不動凍結為 **MOMENTUM_LONG_BASELINE**（hash `eba0cff4a970`）。"
                "Phase 2 只重現、不修改。即時訊號、試單狀態請看「🏠 Emerging Leader Radar」。")
    ref = _csv("MOMENTUM_LONG_RESULTS.csv")
    if ref.empty:
        return _missing("MOMENTUM_LONG_RESULTS.csv")
    st.dataframe(_fmt(ref), hide_index=True, use_container_width=True)


def page_weinstein_long():
    st.header("📗 Weinstein Long（W1 突破 / W2 續漲 / W3 回測）")
    st.caption("BOOK-DERIVED 規則附原書頁碼；數字門檻為 RESEARCH-DERIVED（見 WEINSTEIN_QUANT_RULES.md）。")
    mk = _csv("dash_market_stage.csv", root=False)
    if not mk.empty:
        mk["date"] = pd.to_datetime(mk["date"])
        last = mk.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("TAIEX 週線 Stage", STAGE_ZH.get(int(last["mkt_stage"]), "—"))
        c2.metric("全市場 Stage 1+2 比例（breadth）", f"{last['breadth12'] * 100:.0f}%")
        c3.metric("資料週", str(last["date"].date()))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=mk["date"], y=mk["taiex"], name="TAIEX", line=dict(color=C_BLUE)))
        fig.add_trace(go.Scatter(x=mk["date"], y=mk["breadth12"], name="Stage1+2 比例", line=dict(color=C_ORG)),
                      secondary_y=True)
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    tab1, tab2, tab3, tab4 = st.tabs(["下週掛單 / 觀察", "個股週線", "回測結果", "成功 / 失敗案例"])
    with tab1:
        cur = _csv("dash_weinstein_current.csv", root=False)
        if cur.empty:
            _missing("目前 setup")
        else:
            longs = cur[cur["side"] > 0]
            st.markdown("**W1 / W2：下週 buy-stop-limit 掛單**（觸發價、限價、停損為未還原實際價格）")
            show = ["engine", "variant_ok", "stock_id", "name", "sector", "stage", "ep_len", "trigger_raw", "limit_raw",
                    "stop_raw", "stop_pct", "oh15", "virgin", "mrs", "rs_slope13", "grp_stage", "mkt_stage"]
            st.dataframe(_fmt(longs[[c for c in show if c in longs]], pct_cols=("stop_pct", "mrs", "rs_slope13"),
                              num_cols=("trigger_raw", "limit_raw", "stop_raw")), hide_index=True,
                         use_container_width=True)
    with tab2:
        W = _weekly()
        if not W.empty:
            ids = sorted(W["stock_id"].unique())
            names = D.names() if hasattr(D, "names") else {}
            sid = st.selectbox("股票", ids, index=ids.index("3653") if "3653" in ids else 0,
                               format_func=lambda s: f"{s} {names.get(s, '')}")
            weekly_chart(sid, names.get(sid, ""))
        else:
            _missing("週線資料")
    with tab3:
        res = _csv("WEINSTEIN_LONG_RESULTS.csv")
        if not res.empty:
            win = st.multiselect("期間", list(WIN_ZH), default=["DISCOVERY", "STRICT_OOS"],
                                 format_func=lambda k: WIN_ZH[k])
            _results_table(res, ["engine", "variant", "exit"], tuple(win))
        else:
            _missing("WEINSTEIN_LONG_RESULTS.csv")
    with tab4:
        ex = _csv("dash_examples.csv", root=False)
        if ex.empty:
            _missing("案例")
        else:
            eng = st.selectbox("引擎", sorted(ex["engine"].unique()))
            sub = ex[ex["engine"] == eng]
            st.dataframe(_fmt(sub, pct_cols=("ret", "mfe", "mae", "stop_dist_pct")), hide_index=True,
                         use_container_width=True)


def page_weinstein_short():
    st.header("📕 Weinstein Short（S1 跌破 / S2 量縮反彈）")
    st.warning("原書 Ch.7「Selling Short」不在提供的 PDF。停損、回補規則全部為 RESEARCH-DERIVED；"
               "台股平盤下限制、券源、停券強制回補可能讓理論報酬無法實現。")
    cur = _csv("dash_weinstein_current.csv", root=False)
    if not cur.empty:
        sh = cur[cur["side"] < 0]
        st.markdown("**S1：下週 sell-stop 掛單**")
        st.dataframe(_fmt(sh, pct_cols=("stop_pct", "mrs", "rs_slope13"), num_cols=("trigger_raw", "limit_raw",
                                                                                     "stop_raw")),
                     hide_index=True, use_container_width=True)
    res = _csv("WEINSTEIN_SHORT_RESULTS.csv")
    if not res.empty:
        _results_table(res, ["engine", "variant", "exit", "executability"])
    with st.expander("SHORT_EXECUTABILITY_AUDIT.md"):
        st.markdown(D.doc("SHORT_EXECUTABILITY_AUDIT.md"))


def page_highrr():
    st.header("🎯 High R/R Radar")
    st.caption("每週收盤後更新；Stage 1/2 個股的結構停損、上方空間、上方套牢量、RS、壓縮、量能、延伸度與 "
               "HIGH_RR_SCORE_V1（Discovery 凍結）。不是即時行情。")
    df = _csv("HIGH_RR_CANDIDATES.csv")
    if df.empty:
        return _missing("HIGH_RR_CANDIDATES.csv")
    c1, c2, c3, c4 = st.columns(4)
    stage = c1.multiselect("Stage", [1, 2], default=[2])
    max_atr = c2.slider("停損距離上限（ATR）", 0.5, 6.0, 3.0, 0.5)
    min_rr = c3.slider("PRE_TRADE_RR 下限", 0.0, 10.0, 2.0, 0.5)
    lt = c4.multiselect("領導類型", ["SECTOR_CONFIRMED", "INDEPENDENT", "NOT_LEADER"],
                        default=["SECTOR_CONFIRMED", "INDEPENDENT"])
    sub = df[df["stage"].isin(stage) & (df["stop_dist_atr"] <= max_atr) & (df["pre_trade_rr"] >= min_rr)
             & df["leader_type"].isin(lt)]
    st.markdown(f"符合條件：**{len(sub)}** 檔（資料日 {df['date'].iloc[0]}）")
    cols = ["stock_id", "name", "sector", "stage", "stage_age_weeks", "hrr_score", "close", "structural_stop",
            "stop_dist_pct", "stop_dist_atr", "room_to_resistance_pct", "blue_sky", "pre_trade_rr", "supply15",
            "rs120_pct", "rs_leads120_10d", "mrs", "leader_type", "atr5_20", "vol10_60", "ext_ma60_atr"]
    st.dataframe(_fmt(sub[[c for c in cols if c in sub]], pct_cols=("stop_dist_pct", "room_to_resistance_pct",
                                                                    "supply15", "rs120_pct", "mrs"),
                      num_cols=("hrr_score", "close", "structural_stop", "stop_dist_atr", "pre_trade_rr", "atr5_20",
                                "vol10_60", "ext_ma60_atr")), hide_index=True, use_container_width=True)
    with st.expander("HIGH_RR_FEATURE_RESEARCH.md"):
        st.markdown(D.doc("HIGH_RR_FEATURE_RESEARCH.md"))


def page_intraday():
    st.header("⏱ Intraday Replay（1 秒 K）")
    st.caption("1 秒引擎只決定「何時」進場，不決定「買哪檔」。OHLCV 由逐筆成交聚合，沒有委託簿，不假裝有 order flow。"
               "標記為各政策的實際進場秒（決策秒 t，於 t+1 秒成交）。")
    pe = _csv("intraday_policy_entries.csv", root=False)
    if pe.empty:
        return _missing("intraday_policy_entries.csv")
    strat = st.selectbox("策略", sorted(pe["strategy"].unique()))
    sub = pe[pe["strategy"] == strat]
    split = st.radio("樣本", ["TEST", "TRAIN"], horizontal=True)
    sub = sub[sub["split"] == split]
    cands = sub["cand_id"].drop_duplicates().tolist()
    if not cands:
        return st.info("無候選日")
    cid = st.selectbox("候選日", cands)
    rows = sub[sub["cand_id"] == cid]
    r0 = rows.iloc[0]
    try:
        from data.ticks import bars_1s, sec_to_time
        b = bars_1s(str(r0["stock_id"]), str(r0["date"]))
    except Exception as e:     # noqa: BLE001
        st.error(f"讀取 tick 失敗：{type(e).__name__}")
        return
    if b is None:
        return st.warning("本機沒有這一天的 tick 快取")
    b = b.copy()
    b["t"] = pd.to_datetime(str(r0["date"])) + pd.to_timedelta(9, "h") + pd.to_timedelta(b.index, "s")
    tr = b[b["has_trade"]]
    cum_v = b["volume"].cumsum()
    vwap = (b["volume"] * b["close"]).cumsum() / cum_v.replace(0, np.nan)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=b["t"], y=b["close"], name="1 秒價格", line=dict(color=C_BLUE, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=b["t"], y=vwap, name="VWAP", line=dict(color=C_ORG, width=1, dash="dot")), row=1, col=1)
    fig.add_hline(y=float(r0["trigger"]), line_color=C_UP, line_dash="dash", annotation_text="觸發價", row=1, col=1)
    fig.add_hline(y=float(r0["stop"]), line_color=C_DN, line_dash="dash", annotation_text="日線結構停損", row=1, col=1)
    sym = {"OPEN": "circle", "A_BREAKOUT_IMMEDIATE": "triangle-up", "LEARNED": "star", "B_BREAKOUT_HOLD_60S": "square",
           "C_BREAKOUT_RETEST_HOLD": "diamond", "E_BREAKOUT_VWAP_RECLAIM": "x"}
    for r in rows.itertuples():
        t = pd.to_datetime(str(r0["date"])) + pd.to_timedelta(9, "h") + pd.to_timedelta(int(r.sec) + 1, "s")
        fig.add_trace(go.Scatter(x=[t], y=[r.entry_price], mode="markers+text", text=[f"{r.policy} {r.entry_time}"],
                                 textposition="top center", marker=dict(size=12, symbol=sym.get(r.policy, "circle")),
                                 name=r.policy), row=1, col=1)
    fig.add_trace(go.Bar(x=tr["t"], y=tr["volume"], name="量（張）", marker_color=C_MUTED), row=2, col=1)
    fig.update_layout(height=620, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(_fmt(rows[["policy", "entry_time", "entry_price", "fwd_15m", "fwd_60m", "fwd_eod", "mfe_eod",
                            "mae_eod", "utility", "stop_dist_daily_atr", "stop_dist_micro_atr"]],
                      pct_cols=("fwd_15m", "fwd_60m", "fwd_eod", "mfe_eod", "mae_eod", "utility"),
                      num_cols=("entry_price", "stop_dist_daily_atr", "stop_dist_micro_atr")),
                 hide_index=True, use_container_width=True)
    res = _csv("INTRADAY_TRIGGER_RESULTS.csv")
    if not res.empty:
        with st.expander("各政策彙總（INTRADAY_TRIGGER_RESULTS.csv）"):
            st.dataframe(res[res["strategy"] == strat], hide_index=True, use_container_width=True)


def page_verdicts2():
    st.header("🧾 Phase 2 判決（ACCEPT / WATCH / REJECT）")
    v = _csv("PHASE2_VERDICTS.csv")
    if v.empty:
        return _missing("PHASE2_VERDICTS.csv")
    st.dataframe(v, hide_index=True, use_container_width=True)
    for doc in ("STRATEGY_DECISION_LOG.md", "WEINSTEIN_PDF_AUDIT.md"):
        with st.expander(doc):
            st.markdown(D.doc(doc))
