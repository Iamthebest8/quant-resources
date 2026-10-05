"""Emerging Leader 中文 Dashboard (Streamlit).

    streamlit run dashboard/app.py

只讀本地檔案（outputs/*.csv、outputs/signals/*.parquet、data/cache/manifest.json）。
不呼叫 FinMind、不讀 Token；「更新資料」按鈕會在子程序執行 pipeline.refresh。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

st.set_page_config(page_title="Emerging Leader Radar", page_icon="📡", layout="wide")

C_STOCK, C_MKT, C_SEC = "#2a78d6", "#eb6834", "#1baf7a"
C_UP, C_DN, C_MUTED = "#e34948", "#008300", "#8a8986"   # 台股慣例：紅漲綠跌
PERIODS = {"Discovery 2023–2024": "DISCOVERY", "Extended Validation 2024–2026": "EXTENDED",
           "Strict OOS 2025–2026": "STRICT_OOS", "Year-by-Year": "YEARLY"}


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{d}f}%"


def fnum(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"


# ---------------------------------------------------------------------------
# sidebar: data banner + historical replay date
# ---------------------------------------------------------------------------
if not D.available():
    st.error("找不到 Dashboard 資料（outputs/signals/meta.json）。請先執行：`python run_all.py`（見 README_DASHBOARD.md）。")
    st.stop()

meta = D.meta()
cfg = D.frozen_config()
all_dates = D.dates()
sig_dates = all_dates[all_dates >= pd.Timestamp("2023-01-01")]

with st.sidebar:
    st.title("📡 Emerging Leader")
    if meta.get("data_source") == "synthetic":
        st.error("⚠️ SYNTHETIC 測試資料 — 非真實市場")
    st.markdown(f"**最新資料日期：{meta.get('last_data_date')}**  \n"
                f"FinMind 最後更新：{meta.get('finmind_last_refresh_utc', '—')} (UTC)  \n"
                f"訊號計算：{meta.get('signals_built_utc', '—')} (UTC)  \n"
                f"策略版本：{meta.get('version')} · `{meta.get('config_hash')}`")
    st.caption("資料為日線收盤後更新，**不是即時行情**。")
    page = st.radio("頁面", ["🏠 Emerging Leader Radar", "📈 個股分析", "💼 投資組合", "🔬 研究結果",
                             "🔎 3653 健策 Case Study", "🔄 資料狀態 / 更新資料"])
    st.divider()
    st.subheader("Historical Replay")
    asof = st.select_slider("觀察日期（只顯示當時已知資料）", options=list(sig_dates),
                            value=sig_dates[-1], format_func=lambda d: d.strftime("%Y-%m-%d"))
    asof = pd.Timestamp(asof)
    if asof < sig_dates[-1]:
        st.warning(f"回放模式：{asof.date()}（之後的資料全部隱藏）")


def state_table(d: pd.Timestamp) -> pd.DataFrame:
    sig = D.signals_on(d)
    if sig.empty:
        return sig
    info = D.stock_info().set_index("stock_id")
    stt = D.states()
    s_d = stt[stt["date"] == d].set_index("stock_id")
    df = sig.set_index("stock_id")
    df["名稱"] = info["name"].reindex(df.index)
    df["產業"] = info["sector"].reindex(df.index)
    wq, dq = cfg.get("watch_q", 0.8), cfg.get("disc_q", 0.9)
    base = np.where(df["disc_pct"] >= dq, "Emerging", np.where(df["disc_pct"] >= wq, "觀察", ""))
    df["狀態"] = s_d["state_zh"].reindex(df.index).fillna(pd.Series(base, index=df.index))
    df.loc[(df["狀態"] == "Emerging") & df["probe_signal"].astype(bool), "狀態"] = "可試單"
    df.loc[(df["狀態"].isin(["", "觀察"])) & df["probe_signal"].astype(bool), "狀態"] = "可試單"
    conds = ["c_universe", "c_no_disposition", "c_probe_score", "c_trigger", "c_regime", "c_stop_ok"]
    names = {"c_universe": "流動性", "c_no_disposition": "非處置", "c_probe_score": "分數", "c_trigger": "觸發",
             "c_regime": "大盤", "c_stop_ok": "停損距"}

    def probe_trig(r):
        miss = [names[c] for c in conds if not bool(r[c])]
        return "✔ 全部通過" if not miss else f"{6 - len(miss)}/6 缺：" + "、".join(miss)
    df["Probe Trigger"] = df.apply(probe_trig, axis=1)
    df["Probe Status"] = s_d["state_zh"].reindex(df.index).fillna("未持有")

    def conf_trig(sid):
        if sid not in s_d.index:
            return "—"
        r = s_d.loc[sid]
        ks = [k for k in ("conf_price", "conf_rs", "conf_persist", "conf_trend") if k in r.index and pd.notna(r[k])]
        if not ks or r["state"] not in ("PROBED", "WAIT_CONFIRM"):
            return "已確認" if r["state"] in ("CONFIRMED", "ADD_READY", "FULL") else "—"
        return " ".join(f"{k[5:]}{'✔' if r[k] else '✘'}" for k in ks)
    df["Confirmation Trigger"] = [conf_trig(s) for s in df.index]

    def add_trig(sid):
        if sid not in s_d.index:
            return "—"
        r = s_d.loc[sid]
        if r["state"] == "ADD_READY":
            return "明日開盤加碼"
        if r["state"] in ("PROBED", "WAIT_CONFIRM"):
            return "需先 Confirmation"
        if r["state"] == "FULL":
            return "已滿倉"
        if r["state"] == "CONFIRMED":
            return "等待第二段確認（新高+趨勢）" if cfg.get("add_arch") == "C" else "—"
        return "—"
    df["Add Trigger"] = [add_trig(s) for s in df.index]
    df["Price Structure"] = [
        ("趨勢✔" if r.trend_ok > 0 else "趨勢✘") + (" HH" if r.hh_flag > 0 else "") + (" HL" if r.hl_flag > 0 else "")
        + f" 距20日高{r.dist_h20 * 100:+.1f}%" for r in df.itertuples()]
    df["處置狀態"] = np.where(df["disp"].astype(bool), "處置中(" + df["disp_minutes"].astype(int).astype(str) + "分)",
                          np.where(df["c_no_disposition"].astype(bool), "—", "將處置"))
    return df.reset_index()


# ---------------------------------------------------------------------------
# PAGE: Radar
# ---------------------------------------------------------------------------
def page_radar():
    st.header(f"🏠 Emerging Leader Radar · {asof.date()}")
    mk = D.market()
    if asof in mk.index:
        r = mk.loc[asof]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("加權指數", fnum(r["close"], 2), pct(r["m_ret"], 2))
        c2.metric("大盤 Regime", {"MARKET_UP": "上漲", "MARKET_SIDEWAYS": "盤整", "MARKET_DOWN": "下跌"}.get(
            r["regime"], r["regime"]))
        c3.metric("20 日報酬", pct(r["m_ret20"]))
        c4.metric("趨勢效率 ER20", fnum(r["m_er20"]))
        c5.metric("開→收", pct(r["m_oc"], 2))
    df = state_table(asof)
    if df.empty:
        st.info("該日無訊號資料。")
        return
    show = df[df["狀態"] != ""].copy()
    counts = show["狀態"].value_counts()
    st.caption(" · ".join(f"{k} {counts.get(k, 0)}" for k in D.STATE_ORDER))
    sel = st.multiselect("狀態篩選", D.STATE_ORDER, default=[s for s in D.STATE_ORDER if s not in ("觀察",)])
    show = show[show["狀態"].isin(sel)]
    show["_o"] = show["狀態"].map({s: i for i, s in enumerate(D.STATE_ORDER)})
    show = show.sort_values(["_o", "disc_pct"], ascending=[True, False])
    tbl = pd.DataFrame({
        "股票代號": show["stock_id"], "名稱": show["名稱"], "產業": show["產業"], "狀態": show["狀態"],
        "Discovery Score": show["disc_pct"].round(3), "Probe Status": show["Probe Status"],
        "市場相對強度(10D超額)": (show["cex_10"] * 100).round(2), "產業相對強度(10D)": (show["srs_10"] * 100).round(2),
        "RS20%": (show["rs_20"] * 100).round(1), "RS40%": (show["rs_40"] * 100).round(1),
        "RS60%": (show["rs_60"] * 100).round(1), "RS acceleration": (show["rs_accel"] * 1e3).round(2),
        "Price Structure": show["Price Structure"], "ATR%": (show["atr_pct"] * 100).round(2),
        "Turnover 百分位": show["turnover_pct"].round(2), "20日均成交值(億)": (show["val20"] / 1e8).round(2),
        "處置狀態": show["處置狀態"], "Probe Trigger": show["Probe Trigger"],
        "Confirmation Trigger": show["Confirmation Trigger"], "Add Trigger": show["Add Trigger"],
        "收盤": show["close"].round(2)})
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=600)
    st.caption("RS acceleration ×1000；市場/產業相對強度為 10 日累積超額報酬（%）。狀態由凍結 V1 狀態機在當日收盤後計算（PIT）。")


# ---------------------------------------------------------------------------
# PAGE: stock
# ---------------------------------------------------------------------------
def rel_chart(px, mkt, sec, start, title):
    px, mkt, sec = px[px.index >= start], mkt[mkt.index >= start], sec[sec.index >= start]
    fig = go.Figure()
    for s, name, col in ((px["close"], "個股", C_STOCK), (mkt["adj_close"] if "adj_close" in mkt else mkt["close"],
                                                          "大盤 TAIEX", C_MKT), (sec, "產業(等權,不含本股)", C_SEC)):
        s = s.dropna()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=name, line=dict(color=col, width=2),
                                     hovertemplate="%{x|%Y-%m-%d}<br>" + name + " %{y:.1f}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color=C_MUTED, width=1, dash="dot"))
    fig.update_layout(title=title, height=360, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.02, x=0), yaxis_title="起點 = 100")
    return fig


def page_stock():
    info = D.stock_info()
    opts = (info["stock_id"] + " " + info["name"].fillna("")).tolist()
    default = opts.index("3653 健策") if "3653 健策" in opts else 0
    choice = st.selectbox("選擇股票", opts, index=default)
    sid = choice.split()[0]
    px = D.prices_stock(sid)
    px = px[px.index <= asof]
    sg = D.signals_stock(sid)
    sg = sg[sg.index <= asof]
    if px.empty:
        st.warning("無價格資料")
        return
    st.header(f"📈 {choice} · {asof.date()}")
    win = st.radio("視窗", [60, 120, 250, 500], index=1, horizontal=True, format_func=lambda x: f"{x} 日")
    px_w = px.iloc[-win:]
    mk = D.market()
    mk = mk[mk.index <= asof]
    sec = D.sector_stock(sid)
    sec = sec[sec.index <= asof]
    ev = D.events()
    ev = ev[(ev["stock_id"] == sid) & (ev["date"] <= asof)]
    stt = D.states()
    stt = stt[(stt["stock_id"] == sid) & (stt["date"] <= asof)]

    # --- price chart with events -------------------------------------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=px_w.index, open=px_w["open"], high=px_w["high"], low=px_w["low"],
                                 close=px_w["close"], name="還原K線", increasing_line_color=C_UP,
                                 decreasing_line_color=C_DN), row=1, col=1)
    for n, col in ((10, "#9085e9"), (20, "#eda100"), (60, "#52514e")):
        ma = px["close"].rolling(n).mean().reindex(px_w.index)
        fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"MA{n}", line=dict(width=1.5, color=col)), row=1, col=1)
    sym = {"PROBE": ("triangle-up", "Probe"), "CONFIRMED": ("star", "確認"), "ADD1": ("diamond", "加碼"),
           "ADD2": ("diamond", "加碼2"), "FULL": ("square", "Full"), "FAILED_PROBE": ("x", "試單失敗"),
           "EXIT": ("x", "出場")}
    evw = ev[ev["date"] >= px_w.index[0]]
    for e, (mk_, lab) in sym.items():
        g = evw[evw["event"] == e]
        if len(g):
            fig.add_trace(go.Scatter(x=g["date"], y=g["price"], mode="markers+text", name=lab, text=[lab] * len(g),
                                     textposition="top center", marker=dict(symbol=mk_, size=12, color="#0b0b0b",
                                                                            line=dict(width=2, color="white")),
                                     hovertemplate="%{x|%Y-%m-%d} " + lab + " @ %{y:.2f}<extra></extra>"), row=1, col=1)
    if len(stt) and "stop" in stt:
        st_w = stt[stt["date"] >= px_w.index[0]].dropna(subset=["stop"])
        if len(st_w):
            fig.add_trace(go.Scatter(x=st_w["date"], y=st_w["stop"], mode="markers", name="停損",
                                     marker=dict(symbol="line-ew", size=10, line=dict(width=2, color=C_DN))),
                          row=1, col=1)
    val = px_w["value"] / 1e8
    fig.add_trace(go.Bar(x=val.index, y=val, name="成交值(億)", marker_color=C_MUTED), row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.04, x=0))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(rel_chart(px, mk, sec, px_w.index[0], "相對走勢：個股 / 大盤 / 產業 = 100"),
                        use_container_width=True)
    with c2:
        f2 = go.Figure()
        sgw = sg[sg.index >= px_w.index[0]]
        for col, name, c in (("rs_20", "RS20", C_STOCK), ("rs_60", "RS60", C_MKT)):
            f2.add_trace(go.Scatter(x=sgw.index, y=sgw[col] * 100, name=name, line=dict(color=c, width=2)))
        f2.add_trace(go.Scatter(x=sgw.index, y=sgw["disc_pct"] * 100, name="Discovery 百分位", line=dict(
            color=C_SEC, width=2, dash="dot")))
        f2.add_hline(y=cfg.get("disc_q", 0.9) * 100, line=dict(color=C_MUTED, dash="dash"),
                     annotation_text="Emerging 門檻")
        f2.update_layout(title="RS（相對大盤 %）與 Discovery 百分位", height=360, hovermode="x unified",
                         margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(f2, use_container_width=True)

    # --- position building / trigger panel -----------------------------------------------------
    st.subheader("Position Building")
    if len(ev):
        last_camp = ev["camp_id"].iloc[-1]
        ce = ev[ev["camp_id"] == last_camp]
        steps = [("觀察 / Probe 訊號", ce[ce["event"].str.contains("SIGNAL")]),
                 (f"Probe {cfg.get('add_arch', '')}", ce[ce["event"] == "PROBE"]), ("確認", ce[ce["event"] == "CONFIRMED"]),
                 ("Add", ce[ce["event"].isin(["ADD1", "ADD2"])]), ("Full", ce[ce["event"] == "FULL"]),
                 ("Exit / 失敗", ce[ce["event"].isin(["EXIT", "FAILED_PROBE"])])]
        cols = st.columns(len(steps))
        for col, (lab, g) in zip(cols, steps):
            if len(g):
                r = g.iloc[0]
                col.success(f"**{lab}**  \n{pd.Timestamp(r['date']).date()}  \n@ {r['price']:,.2f}  \nsize {r['size']:.2f}  \n{r['reason']}")
            else:
                col.info(f"**{lab}**  \n尚未發生")
    else:
        st.info("此股票在觀察日前沒有任何 Probe 紀錄。")

    st.subheader("Trigger Panel（觀察日收盤）")
    if asof in sg.index:
        r = sg.loc[asof]
        labels = {"c_universe": "流動性/價格/上市天數", "c_no_disposition": "非處置股", "c_probe_score":
                  f"Discovery 百分位 ≥ {cfg.get('probe_q', 0.9):.0%}（現 {r['disc_pct']:.2f}）",
                  "c_trigger": f"觸發 `{cfg.get('trigger')}`", "c_regime": f"大盤條件 `{cfg.get('regime_filter')}`",
                  "c_stop_ok": f"停損距離 ≤ {cfg.get('max_stop_dist', 0.08):.0%}（現 {r['stop_dist'] * 100:.1f}%）"}
        a, b, c_, d_ = st.columns(4)
        with a:
            st.markdown("**Probe**")
            for k, lab in labels.items():
                st.markdown(("✅ " if bool(r[k]) else "❌ ") + lab)
        srow = stt[stt["date"] == asof]
        with b:
            st.markdown("**Confirmation**")
            if len(srow) and srow.iloc[0]["state"] in ("PROBED", "WAIT_CONFIRM"):
                s0 = srow.iloc[0]
                for k, lab in (("conf_price", "價格突破 + 試單後新高"), ("conf_rs", "RS line 40D 新高 + RS 上升"),
                               ("conf_persist", "持續勝大盤/產業"), ("conf_trend", "MA20 上升 + HH/HL")):
                    if k in s0 and pd.notna(s0[k]):
                        st.markdown(("✅ " if s0[k] else "❌ ") + lab)
                st.caption(f"凍結規則：`{cfg.get('confirm')}`")
            else:
                st.markdown("未在試單狀態")
        with c_:
            st.markdown("**Add**")
            if len(srow):
                s0 = srow.iloc[0]
                st.markdown({"ADD_READY": "✅ 明日開盤加碼", "FULL": "已滿倉", "CONFIRMED": "已確認，等待加碼條件"}.get(
                    s0["state"], "需先 Confirmation"))
            else:
                st.markdown("—")
        with d_:
            st.markdown("**Exit 風險點**")
            if len(srow) and pd.notna(srow.iloc[0].get("stop", np.nan)):
                s0 = srow.iloc[0]
                st.markdown(f"停損價 {s0['stop']:,.2f}（距收盤 {fnum(s0.get('dist_stop', np.nan) * 100, 1)}%）")
                st.markdown(f"距 MA20 {fnum(s0.get('dist_ma20', np.nan) * 100, 1)}% · 距 MA10 "
                            f"{fnum(s0.get('dist_ma10', np.nan) * 100, 1)}%")
                st.markdown(f"未實現 {pct(s0.get('unrealized', np.nan))} · Exit 規則 `{cfg.get('exit')}`")
            else:
                st.markdown(f"Probe Low（近3日最低）{fnum(r['low3'])}")
    st.subheader("State Timeline")
    if len(stt):
        t = stt.sort_values("date", ascending=False)[["date", "state_zh", "close", "stop", "size", "unrealized",
                                                      "detail"]].head(60)
        st.dataframe(t, hide_index=True, use_container_width=True)
    ip = D.case()[1]
    if len(ip) and sid == "3653":
        st.caption("盤中資料請見「3653 健策 Case Study」頁。")


# ---------------------------------------------------------------------------
# PAGE: portfolio
# ---------------------------------------------------------------------------
def page_portfolio():
    st.header(f"💼 10-slot 投資組合（凍結 {meta.get('version')}，回放至 {asof.date()}）")
    eq, pos, tr = D.portfolio()
    eq = eq[eq.index <= asof]
    if eq.empty:
        st.info("無資料")
        return
    last = eq.iloc[-1]
    c = st.columns(6)
    c[0].metric("權益", f"{last['equity'] / 1e6:,.2f}M", pct(last["equity"] / eq["equity"].iloc[0] - 1))
    c[1].metric("現金", pct(last["cash"] / last["equity"]))
    c[2].metric("Probe 資金", pct(last["probe_capital"] / last["equity"]))
    c[3].metric("Slot 占用", f"{last['slots_used']:.2f} / 10")
    c[4].metric("Probe 部位數", int(last["n_probe"]))
    c[5].metric("確認 / Full", f"{int(last['n_confirmed'])} / {int(last['n_full'])}")
    mk = D.market()["close"].reindex(eq.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"] / eq["equity"].iloc[0] * 100, name="策略權益",
                             line=dict(color=C_STOCK, width=2)))
    fig.add_trace(go.Scatter(x=mk.index, y=mk / mk.iloc[0] * 100, name="TAIEX", line=dict(color=C_MKT, width=2)))
    fig.add_vline(x=pd.Timestamp("2025-01-01"), line=dict(color=C_MUTED, dash="dash"))
    fig.add_annotation(x=pd.Timestamp("2025-01-01"), y=1, yref="paper", text="Strict OOS →", showarrow=False,
                       xanchor="left")
    fig.update_layout(title="權益曲線 vs TAIEX（起點=100，價格指數不含股利）", height=380, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=eq.index, y=eq["occupancy"] * 100, name="Slot 占用 %", line=dict(color=C_STOCK, width=2)))
    f2.add_trace(go.Scatter(x=eq.index, y=eq["probe_capital"] / eq["equity"] * 100, name="Probe 資金 %",
                            line=dict(color=C_SEC, width=2)))
    f2.update_layout(height=260, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                     legend=dict(orientation="h", y=1.05, x=0), yaxis_title="%")
    st.plotly_chart(f2, use_container_width=True)
    info = D.stock_info().set_index("stock_id")
    p = pos[pos["date"] == eq.index[-1]].copy()
    p["名稱"] = p["stock_id"].map(info["name"])
    p["未實現"] = p["close"] / p["avg_cost"] - 1
    zh = {"PROBE": "Probe", "CONFIRMED": "Confirmed", "ADD": "Add 中", "FULL": "Full"}
    for stt, lab in (("PROBE", "Probe Positions"), ("CONFIRMED", "Confirmed Positions"), ("ADD", "Add 中"),
                     ("FULL", "Full Positions")):
        g = p[p["state"] == stt]
        st.markdown(f"**{lab}（{len(g)}）**")
        if len(g):
            st.dataframe(g[["stock_id", "名稱", "probe_date", "size_slots", "avg_cost", "close", "未實現", "stop"]],
                         hide_index=True, use_container_width=True)
    ft = tr[(tr["final_state"] == "FAILED_PROBE") & (tr["exit_date"] <= asof)].sort_values("exit_date",
                                                                                         ascending=False).head(20)
    st.markdown(f"**最近 Failed Probes（{len(ft)}）**")
    if len(ft):
        ft = ft.assign(名稱=ft["stock_id"].map(info["name"]))
        st.dataframe(ft[["stock_id", "名稱", "probe_date", "probe_price", "exit_date", "exit_price", "exit_reason",
                         "ret_on_invested", "pnl"]], hide_index=True, use_container_width=True)
    _ = zh


# ---------------------------------------------------------------------------
# PAGE: research
# ---------------------------------------------------------------------------
def page_research():
    st.header("🔬 研究結果（凍結 V1；全期間結果，不受回放日期影響）")
    v = D.csv("FINAL_VERDICTS.csv")
    if len(v):
        st.subheader("最終裁決")
        st.dataframe(v[["component", "verdict", "evidence"]], hide_index=True, use_container_width=True)
    per = st.radio("期間", list(PERIODS), horizontal=True)
    key = PERIODS[per]
    ts = D.csv("TRADE_LEVEL_SUMMARY.csv")
    ps = D.csv("PORTFOLIO_SUMMARY.csv")
    cap = D.csv("LEADER_CAPTURE_RATE.csv")
    if key == "YEARLY":
        y = D.csv("YEARLY_VALIDATION.csv")
        st.dataframe(y, hide_index=True, use_container_width=True)
    else:
        t = ts[ts["window"] == key]
        p = ps[ps["window"] == key]
        if len(t) and len(p):
            t, p = t.iloc[0], p.iloc[0]
            cc = st.columns(8)
            cc[0].metric("CAGR", pct(p["cagr"]))
            cc[1].metric("MDD", pct(p["mdd"]))
            cc[2].metric("Sharpe", fnum(p["sharpe"]))
            cc[3].metric("PF", fnum(t["pf"]))
            cc[4].metric("Payoff", fnum(t["payoff"]))
            cc[5].metric("False Probe", pct(t["false_probe_rate"]))
            cc[6].metric("PnL/100 slot-days", fnum(t["pnl_per_100_slot_days"], 3))
            cc[7].metric("TAIEX 同期", pct(p["taiex_return"]))
            cc2 = st.columns(6)
            cc2[0].metric("Probes", int(t["n_probes"]))
            cc2[1].metric("確認率", pct(t["confirmation_rate"]))
            cc2[2].metric("平均失敗損失", pct(t["avg_failed_loss_ret"]))
            cc2[3].metric("≥20% / ≥40%", f"{int(t['ge20'])} / {int(t['ge40'])}")
            cc2[4].metric("Top 5 占比", pct(t["top5_trades_share"]))
            cc2[5].metric("Right tail p99", pct(t["ret_p99"]))
        cp = cap[(cap["period"] == ("ALL" if key == "EXTENDED" else key))]
        if len(cp):
            st.subheader("Leader Capture Rate")
            st.dataframe(cp, hide_index=True, use_container_width=True)
    st.subheader("Cost Sensitivity（Profit Factor）")
    cs = D.csv("PROBE_COST_ROBUSTNESS.csv")
    if len(cs):
        pn = st.selectbox("成本期間", sorted(cs["period"].unique()), index=list(sorted(cs["period"].unique())).index(
            "STRICT_OOS") if "STRICT_OOS" in cs["period"].unique() else 0)
        g = cs[cs["period"] == pn].pivot(index="round_trip_cost", columns="slippage_bps_per_side", values="pf")
        z = g.to_numpy()
        fig = go.Figure(go.Heatmap(z=z, x=[f"{c} bps" for c in g.columns], y=[f"{r:.2%}" for r in g.index],
                                   colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]], zmid=1.0,
                                   text=np.round(z, 2), texttemplate="%{text}", hovertemplate="成本 %{y} 滑價 %{x}<br>PF %{z:.2f}<extra></extra>",
                                   colorbar=dict(title="PF")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="單邊滑價",
                          yaxis_title="來回成本")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cs[cs["period"] == pn], hide_index=True, use_container_width=True)
    st.subheader("Right Tail")
    st.dataframe(D.csv("PROBE_RIGHT_TAIL.csv"), hide_index=True, use_container_width=True)
    st.subheader("Matched-control Placebo")
    st.dataframe(D.csv("PLACEBO_MATCHED_CONTROLS.csv"), hide_index=True, use_container_width=True)
    with st.expander("Expanding validation（robustness only）"):
        st.dataframe(D.csv("EXPANDING_VALIDATION.csv"), hide_index=True, use_container_width=True)
    with st.expander("Strategy decision log"):
        st.dataframe(D.csv("STRATEGY_DECISION_LOG.csv"), hide_index=True, use_container_width=True)
    with st.expander("EMERGING_LEADER_STRATEGY.md"):
        st.markdown(D.doc("EMERGING_LEADER_STRATEGY.md"))


# ---------------------------------------------------------------------------
# PAGE: 3653 case
# ---------------------------------------------------------------------------
def page_case():
    st.header("🔎 3653 健策 Case Study（2026-07-01 ～ 2026-09-30）")
    st.caption("只作 case study / regression test，策略規則未因健策調整。")
    tl, ip = D.case()
    if tl.empty:
        st.info("尚未產生 case 資料")
        return
    tl = tl[tl["date"] <= asof]
    month = st.radio("月份", ["全部", "7 月", "8 月", "9 月"], horizontal=True)
    v = tl[tl["in_case_period"]] if month == "全部" else tl[tl["date"].dt.month == int(month[0])]
    if v.empty:
        st.info("回放日期之前沒有此月份資料")
        return
    fig = go.Figure()
    for col, name, c in (("stock_norm", "3653 健策", C_STOCK), ("market_norm", "大盤", C_MKT),
                         ("sector_norm", "產業(等權)", C_SEC)):
        s = v[col] / v[col].iloc[0] * 100
        fig.add_trace(go.Scatter(x=v["date"], y=s, name=name, line=dict(color=c, width=2)))
    evd = v[v["events"].fillna("") != ""]
    if len(evd):
        fig.add_trace(go.Scatter(x=evd["date"], y=evd["stock_norm"] / v["stock_norm"].iloc[0] * 100,
                                 mode="markers+text", text=evd["events"].str.split("@").str[0], name="事件",
                                 textposition="top center", marker=dict(size=11, symbol="diamond", color="#0b0b0b")))
    fig.add_hline(y=100, line=dict(color=C_MUTED, dash="dot"))
    fig.update_layout(title="3653 / 大盤 / 產業 = 100", height=420, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    cols = [c for c in ("date", "raw_close", "regime", "disc_pct", "watch", "emerging", "c_probe_score", "c_trigger",
                        "c_regime", "c_stop_ok", "probe_signal", "campaign_state", "position_size", "stop",
                        "portfolio_state", "events", "pct_time_above_vwap", "close_vs_vwap") if c in v.columns]
    st.dataframe(v[cols], hide_index=True, use_container_width=True, height=420)
    if len(ip):
        st.subheader("盤中：個股 / 大盤 / 產業 / VWAP（開盤 = 100）")
        days = sorted(d for d in ip["date"].unique() if pd.Timestamp(d) <= asof)
        if days:
            dsel = st.select_slider("交易日", options=days, value=days[-1], format_func=lambda d: pd.Timestamp(d).strftime("%m-%d"))
            g = ip[ip["date"] == dsel]
            f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
            for col, name, c, dash in (("stock", "3653", C_STOCK, None), ("vwap", "VWAP", C_STOCK, "dot"),
                                       ("market", "大盤", C_MKT, None), ("sector", "產業指數", C_SEC, None)):
                if col in g:
                    f2.add_trace(go.Scatter(x=g["ts"], y=g[col], name=name, line=dict(color=c, width=2, dash=dash)),
                                 row=1, col=1)
            if "rel_spread_vs_market" in g:
                f2.add_trace(go.Scatter(x=g["ts"], y=g["rel_spread_vs_market"], name="相對大盤價差", fill="tozeroy",
                                        line=dict(color=C_MUTED, width=1)), row=2, col=1)
            f2.update_layout(height=480, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                             legend=dict(orientation="h", y=1.04, x=0))
            st.plotly_chart(f2, use_container_width=True)
    with st.expander("3653_CASE_STUDY.md"):
        st.markdown(D.doc("3653_CASE_STUDY.md"))


# ---------------------------------------------------------------------------
# PAGE: data status / refresh
# ---------------------------------------------------------------------------
def page_data():
    st.header("🔄 資料狀態 / 更新資料")
    st.markdown(f"- 資料來源：**{meta.get('data_source')}**  \n- 最新資料日期：**{meta.get('last_data_date')}**  \n"
                f"- FinMind 最後更新（UTC）：**{meta.get('finmind_last_refresh_utc')}**  \n"
                f"- 大盤資料：{meta.get('market_source')}")
    caps = meta.get("capabilities", {})
    if caps:
        st.dataframe(pd.DataFrame(caps.values())[["label", "dataset", "status", "rows"]], hide_index=True,
                     use_container_width=True)
    st.subheader("更新資料")
    st.markdown("流程：讀取 `.env` → FinMind 增量下載（只抓快取之後的新交易日）→ 更新本地快取 → 用**凍結 V1** 重算最新訊號 → 刷新 Dashboard。"
                "  \n完整重跑研究（含 OOS 報表）請用命令列 `python run_all.py`。")
    if meta.get("data_source") == "synthetic":
        st.warning("SYNTHETIC 模式不提供更新。")
    elif st.button("🔄 更新資料（增量）", type="primary"):
        root = Path(__file__).resolve().parents[1]
        with st.status("更新中…", expanded=True) as status:
            proc = subprocess.Popen([sys.executable, "-m", "pipeline.refresh"], cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            box = st.empty()
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                box.code("\n".join(lines[-25:]))
            proc.wait()
            if proc.returncode == 0:
                status.update(label="更新完成", state="complete")
                st.cache_data.clear()
            else:
                status.update(label=f"更新失敗（exit {proc.returncode}）", state="error")
    with st.expander("FINMIND_DATA_AUDIT.md", expanded=False):
        st.markdown(D.doc("FINMIND_DATA_AUDIT.md"))


{"🏠 Emerging Leader Radar": page_radar, "📈 個股分析": page_stock, "💼 投資組合": page_portfolio,
 "🔬 研究結果": page_research, "🔎 3653 健策 Case Study": page_case, "🔄 資料狀態 / 更新資料": page_data}[page]()
