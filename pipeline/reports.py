"""Markdown report writers (data audit, strategy, state machine, 3653 case, final answers)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from data.finmind_client import FinMindCache
from pipeline.verdicts import CRITERIA

OUT = config.OUT_DIR
DOC = config.DOC_DIR


def pct(x, d=1):
    try:
        return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x) * 100:.{d}f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(x, d=2):
    try:
        return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def md_table(df: pd.DataFrame, cols=None, fmt: dict | None = None, max_rows=40) -> str:
    if df is None or len(df) == 0:
        return "_(無資料)_\n"
    df = df[cols] if cols else df
    df = df.head(max_rows)
    fmt = fmt or {}
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            if f == "pct":
                cells.append(pct(v))
            elif f == "num":
                cells.append(num(v))
            elif f == "int":
                cells.append("n/a" if pd.isna(v) else f"{int(v)}")
            elif isinstance(v, (pd.Timestamp,)):
                cells.append(str(v.date()))
            elif isinstance(v, float):
                cells.append(num(v, 3))
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def banner() -> str:
    if config.IS_SYNTHETIC:
        return ("> ⚠️ **SYNTHETIC DATA** — 本文件由離線合成資料產生，只用於驗證程式流程，"
                "**不代表任何真實市場結果**。\n\n")
    return ""


# ---------------------------------------------------------------------------
def data_audit(p, F) -> str:
    cache = FinMindCache(client=None)
    man = cache.manifest
    caps = man.get("capabilities", {})
    lines = [f"# FINMIND DATA AUDIT\n\n{banner()}產生時間：{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  \n"
             f"FinMind 本地快取最後更新：**{man.get('last_refresh_utc')}**  \n"
             f"研究資料最新交易日：**{p.dates[-1].date()}**（非即時資料）\n"]
    lines.append("\n## 1. 連線與權限\n")
    lines.append("- 認證：`.env` 的 `FINMIND_TOKEN`，只放在 HTTP `Authorization: Bearer` header；程式、CSV、"
                 "Dashboard、log 均不含 Token（log 有自動遮罩 filter）。\n"
                 "- `api.finmindtrade.com`（資料 API）：可連線。\n"
                 "- `api.web.finmindtrade.com`（帳戶額度 user_info）：本環境 egress policy 仍封鎖 → 無法讀取帳戶等級與剩餘額度，"
                 "改以實際 dataset 探測判斷權限。\n")
    lines.append("\n## 2. Dataset 探測結果（實際呼叫，非假設）\n")
    cap_df = pd.DataFrame(caps.values())
    if len(cap_df):
        lines.append(md_table(cap_df, [c for c in ("label", "dataset", "data_id", "status", "rows", "note")
                                       if c in cap_df.columns]))
    lines.append("\n## 3. 使用的 datasets 與覆蓋\n")
    ds_rows = []
    for ds, ents in man.get("datasets", {}).items():
        mins = [e.get("min_date") for e in ents.values() if e.get("min_date")]
        maxs = [e.get("max_date") for e in ents.values() if e.get("max_date")]
        ds_rows.append({"dataset": ds, "files": len(ents), "rows": sum(e.get("rows", 0) for e in ents.values()),
                        "min_date": min(mins) if mins else "", "max_date": max(maxs) if maxs else "",
                        "last_update_utc": max((e.get("updated_utc", "") for e in ents.values()), default="")})
    lines.append(md_table(pd.DataFrame(ds_rows)))
    ids = p.ids
    stype = p.stock_type.reindex(ids)
    n_del = int(len(set(man.get("delisted_ids", [])) & set(ids)))
    valid = p.raw_c.notna()
    listed_span = valid.cumsum().gt(0) & valid[::-1].cumsum()[::-1].gt(0)
    miss = float(1 - valid[listed_span].sum().sum() / max(listed_span.sum().sum(), 1))
    lines.append(f"\n## 4. 股票池覆蓋\n\n- 期間：{p.dates[0].date()} ～ {p.dates[-1].date()}，{len(p.dates)} 個交易日\n"
                 f"- 普通股代號（4 碼、非 00 開頭、上市+上櫃；興櫃排除）：**{len(ids)}** 檔\n"
                 f"  - 上市 twse：{int((stype == 'twse').sum())}；上櫃 tpex：{int((stype == 'tpex').sum())}；"
                 f"類型未知（多為已下市）：{int(stype.isna().sum())}\n"
                 f"- 2021-06 之後下市櫃且納入回測的股票：**{n_del}** 檔（來源 TaiwanStockDelisting）\n"
                 f"- 每日可交易 universe（價格≥{config.MIN_PRICE}、20日均成交值≥{config.MIN_AVG_VALUE_20D / 1e6:.0f}M、"
                 f"上市≥{config.MIN_LISTED_DAYS}日）：平均 **{p.universe.sum(axis=1)[p.dates >= config.RESEARCH_START].mean():.0f}** 檔/日\n"
                 f"- 上市期間內缺漏（停牌/無成交）比例：{pct(miss, 2)}\n")
    lines.append("\n## 5. 還原 / 未還原\n\n"
                 "- 原始 `TaiwanStockPrice` 為**未還原**價格。\n"
                 "- 本研究用 `spread`（=收盤−參考價，交易所公告之參考價已反映除權息/減資）計算每日總報酬 "
                 "r = close/(close−spread)−1，再累乘得到還原價；比值型特徵與 PIT 序列完全一致，不會因事後還原而前視。\n"
                 f"- spread 正負號一致性檢查：{p.meta.get('spread_sign_agree', float('nan')):.3f}；"
                 f"被中和的除權息/減資跳空：{p.meta.get('corp_action_gaps')} 次。\n"
                 "- `TaiwanStockPriceAdj` 可用，作為交叉檢查（見下）。\n")
    adj_chk = p.meta.get("adj_check")
    if adj_chk is not None and len(adj_chk):
        lines.append(md_table(adj_chk))
    sec = p.sector.reindex(ids)
    lines.append(f"\n## 6. 指數覆蓋\n\n- 大盤：{p.meta.get('market_source')}，{p.market['close'].first_valid_index().date()} ～ "
                 f"{p.market['close'].last_valid_index().date()}\n"
                 f"- 櫃買指數：{'有' if len(p.market2) else '無'}\n")
    lines.append(f"\n## 7. 產業覆蓋\n\n- 來源：TaiwanStockInfo.industry_category（目前快照）\n"
                 f"- 產業數：{sec.nunique()}；UNKNOWN（多為已下市股票，info 中已無資料）：{int((sec == 'UNKNOWN').sum())} 檔\n"
                 f"- 籠統分類「電子工業」：{int((sec == '電子工業').sum())} 檔（例如 2330 被歸在此類），"
                 "降低 sector-relative 的精細度。\n"
                 "- 產業成分 < 4 檔時 sector return 以大盤替代。\n")
    dt = p.meta.get("disposition_table")
    n_disp = 0 if dt is None else len(dt)
    lines.append(f"\n## 8. 處置股票覆蓋\n\n- 來源：TaiwanStockDispositionSecuritiesPeriod，普通股處置事件 {n_disp} 筆"
                 + (f"（{dt['announce_date'].min().date()} ～ {dt['announce_date'].max().date()}）" if n_disp else "") +
                 "\n- `disposition_status`：是否處置、period_start/end、撮合間隔（由 measure 文字解析；未載明時第一次=5分、第二次=20分）、"
                 "限制（預收款券/人工管制）。\n"
                 "- 處置期間及隔日的量能特徵設為 **NOT_COMPARABLE**（NaN），不當作籌碼訊號；處置中或隔日將處置的股票不新開 Probe。\n")
    kb = len([k for k in man.get("datasets", {}).get("TaiwanStockKBar", {})])
    i5 = len(man.get("datasets", {}).get("TaiwanVariousIndicators5Seconds", {}))
    s5 = len(man.get("datasets", {}).get("TaiwanStockEvery5SecondsIndex", {}))
    lines.append(f"\n## 9. 盤中 / 分鐘資料\n\n- TaiwanStockKBar（1 分 K）可用，但一次只能抓「一檔 × 一天」→ 全市場 × 3.7 年約 200 萬次請求，"
                 f"不可行。只下載 {config.CASE_SYMBOL} 個案期間：{kb} 天。\n"
                 f"- 加權指數 5 秒：{i5} 天；類股指數 5 秒：{s5} 天（個案用）。\n"
                 "- 全市場研究的盤中特徵改用日線 proxy：open→close、收盤位置 CLV、low→close recovery、相對大盤版本 → "
                 "**INTRADAY_LIMITATION**。\n")
    lines.append("\n## 10. Survivorship bias\n\n"
                 f"- 已納入 2021-06 之後下市櫃的普通股 {n_del} 檔（價格資料至下市日）。\n"
                 "- 回測中持股遇停止交易 → 以最後收盤價出場（標記 DELISTED/SUSPENDED）。\n"
                 "- 殘留風險：下市股票在 TaiwanStockInfo 中已無產業別（→ UNKNOWN）；2021-06 之前下市者不影響 2023+ 研究。\n"
                 f"- datalist 中不在 info/delisting 名單的普通股代號：{man.get('datalist_extra_ids_count', 'n/a')} 個（多為更早下市，未下載）。\n"
                 "- 判定：**SURVIVORSHIP_BIAS_RISK = LOW（已處理下市股；產業別有殘留缺口）**\n")
    lines.append("\n## 11. PIT 風險\n\n"
                 "- 產業別為目前快照，歷史改類無法重建 → PIT_RISK（中低）。\n"
                 "- 處置資料以公告日 `date` 為已知時點；處置期間從公告後開始 → PIT 正確。\n"
                 "- 市值（turnover 分母）用月底快照向後填補 → PIT 正確。\n"
                 "- 所有特徵皆為 trailing window / 當日橫斷面排名；訊號 t 日收盤產生、t+1 開盤成交。"
                 "`tests/test_pit.py` 以截斷資料重算比對。\n"
                 "- 研究標籤（未來報酬）只在 research/ 使用，strategy/ 不 import。\n")
    lines.append("\n## 12. API 限制\n\n"
                 "- 帳戶額度端點被環境封鎖 → 無法顯示剩餘額度；client 內建 402 rate-limit 等待重試。\n"
                 "- 全市場單日 `TaiwanStockPrice` 含權證（約 4.8 萬列/日），歷史改以逐檔下載，增量更新才用單日全市場。\n"
                 "- KBar / 5 秒資料一次一天。\n"
                 "- 資料非即時：最新為上一個收盤後 FinMind 更新的日線。\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
def strategy_doc(cfg, res) -> str:
    tm, psum = res["tm"], res["psum"]
    frozen = json.loads((OUT / "frozen" / f"{cfg.version}.json").read_text(encoding="utf-8"))
    feats = frozen.get("discovery_features", {})
    v = res["verdicts"]
    L = [f"# EMERGING LEADER PROBE STRATEGY — {cfg.version}\n\n{banner()}"
         f"凍結時間：{frozen['frozen_utc']}（Discovery 2023–2024 選定，**在任何 2025–2026 結果計算前凍結**）  \n"
         f"Config hash：`{frozen['hash']}`\n\n"
         "> 核心：DISCOVER EARLY · RISK SMALL · ADD ONLY WHEN RIGHT · LET WINNERS MATTER\n\n"]
    L.append("## 1. 資料流程\n\n```\nFinMind API (data/finmind_client.py)\n    ↓\nRaw Local Cache (data/cache/raw/*.parquet, manifest.json)\n"
             "    ↓\nClean / PIT Dataset (engine/panel.py：spread 還原、universe、產業、處置)\n    ↓\nResearch Engine "
             "(engine/features.py, research/*)\n    ↓\nStrategy Engine (strategy/engine.py 狀態機)\n    ↓\n"
             "Portfolio Simulator (strategy/portfolio.py 10-slot)\n    ↓\n中文 Dashboard (dashboard/app.py)\n```\n\n")
    L.append("## 2. 凍結規則（V1）\n\n")
    L.append(f"**Discovery score** = 下列特徵橫斷面百分位的等權平均（每個 family 取 Discovery 期間 vol-controlled lift 最高者，"
             f"lift < 1.10 的 family 不納入）：\n\n")
    for fam, d in feats.items():
        L.append(f"- {fam}: `{d['feature']}`（Discovery lift {d['lift']}）\n")
    L.append(f"\n| 區塊 | 凍結值 |\n|---|---|\n"
             f"| 觀察 / Emerging | score ≥ {cfg.watch_q:.0%} / ≥ {cfg.disc_q:.0%} 百分位 |\n"
             f"| Probe 分數門檻 | ≥ {cfg.probe_q:.0%} 百分位 |\n"
             f"| Probe 觸發 | `{cfg.trigger}` |\n| 大盤條件 | `{cfg.regime_filter}` |\n"
             f"| 停損距離 | ≤ {cfg.max_stop_dist:.0%}（Probe Low = 近 3 日最低） |\n"
             f"| 追價限制 | 開盤 > 訊號收盤 +{cfg.max_gap:.0%} 不進；漲停鎖死不進 |\n"
             f"| Failed Probe | `{cfg.fail}`" + (f"（時間停損 {cfg.time_stop} 日）" if "F4" in cfg.fail else "") +
             f"，最長試單 {cfg.probe_max_life} 日 |\n"
             f"| Confirmation | `{cfg.confirm}`（試單後至少 {cfg.min_days_confirm} 個收盤） |\n"
             f"| Add 架構 | `{cfg.add_arch}`（Probe {cfg.probe_size:.2f} → {' → '.join(f'{x:.2f}' for x in cfg.add_targets)}） |\n"
             f"| Exit | `{cfg.exit}` |\n| 同時 Probe 上限 | {cfg.max_probes} |\n"
             f"| 資金架構 | {cfg.capital_arch}" + (f"（保留 {cfg.probe_reserve} slot 給 Probe）" if cfg.capital_arch == 'B' else "（10 slots 共用）") + " |\n"
             f"| 排名 | Part 23 固定等權：relative persistence、sector RS、RS acceleration、price progress/ATR、relative high formation、capital velocity |\n")
    L.append("\n## 3. 主要結果（基準成本 0.45% 來回 + 25bps 單邊滑價）\n\n")
    rows = []
    for k in ("DISCOVERY", "EXTENDED", "STRICT_OOS"):
        m = tm[k]
        pm = psum[psum["window"] == k].iloc[0]
        rows.append({"期間": k, "Probes": m.get("n_probes"), "PF": num(m.get("pf")), "Payoff": num(m.get("payoff")),
                     "勝率": pct(m.get("win_rate")), "EV/probe(slot)": num(m.get("ev_slots_per_probe"), 4),
                     "PnL/100 slot-days": num(m.get("pnl_per_100_slot_days"), 3), "失敗率": pct(m.get("false_probe_rate")),
                     "確認率": pct(m.get("confirmation_rate")), "≥20%": m.get("ge20"), "組合CAGR": pct(pm["cagr"]),
                     "MDD": pct(pm["mdd"]), "Sharpe": num(pm["sharpe"]), "TAIEX": pct(pm["taiex_return"])})
    L.append(md_table(pd.DataFrame(rows)))
    L.append("\n## 4. 最終裁決（依 pipeline/verdicts.py 預先登錄之標準）\n\n| 元件 | 裁決 | 證據 |\n|---|---|---|\n")
    for k, (vv, why) in v.items():
        L.append(f"| {k} | **{vv}** | {why} |\n")
    L.append("\n## 5. 預先登錄的判定標準\n\n")
    for k, c in CRITERIA.items():
        L.append(f"- **{k}** — ACCEPT: {c['ACCEPT']}；WATCH: {c['WATCH']}；其餘 REJECT\n")
    L.append("\n## 6. 研究版本紀律\n\n- V1 由 Discovery（2023–2024，標籤 purge 40 日、campaign 於 2024-12-31 強制結算）選定後凍結。\n"
             "- 2024 同時屬於 Discovery 與 Extended Validation → 2024 **不是** OOS。\n"
             "- 2025-01-01 ～ 2026-09-03 為 Strict OOS；看到 OOS 後若修改規則必須以 `--refreeze V2` 建立新版本並記錄於 "
             "STRATEGY_DECISION_LOG.csv。\n- 健策 3653 只作 case study / regression test，未參與任何門檻選擇。\n")
    return "".join(L)


def state_machine_doc(cfg) -> str:
    return f"""# EMERGING_LEADER_STATE_MACHINE — {cfg.version}

{banner()}```
STATE 0  DISCOVERY        觀察 (score≥{cfg.watch_q:.0%}) → Emerging (score≥{cfg.disc_q:.0%})   無部位
   │  Probe 條件全部成立（收盤）
   ▼
STATE 1  PROBE            t+1 開盤買進 {cfg.probe_size:.2f} 個正式部位（= {cfg.probe_size * 10:.1f}% 權益）
   │                      停損 = Probe Low（近 3 日最低，盤中觸價出場）
   ├──── 失敗 ──────────► STATE 2A FAILED PROBE  快速小虧出場 ({cfg.fail}{', ' + str(cfg.time_stop) + '日' if 'F4' in cfg.fail else ''}; 最長 {cfg.probe_max_life} 日)
   │
   │  Confirmation `{cfg.confirm}` 成立（收盤，至少 {cfg.min_days_confirm} 個收盤後）
   ▼
STATE 2B CONFIRMED        停損上移至 max(Probe Low, 確認日近 3 日最低)
   │  t+1 開盤加碼
   ▼
STATE 3  ADD              架構 {cfg.add_arch}: {cfg.probe_size:.2f} → {' → '.join(f'{x:.2f}' for x in cfg.add_targets) or '(不加碼)'}
   ▼
STATE 4  FULL POSITION    1.0 個正式部位（10% 權益）
   │  Exit `{cfg.exit}` 或硬停損
   ▼
STATE 5  EXIT             t+1 開盤出場（停損為盤中觸價）
```

## 狀態定義

| 狀態 | Dashboard 標籤 | 進入條件 | 離開條件 |
|---|---|---|---|
| 0 DISCOVERY | 觀察 / Emerging | Discovery score 百分位 ≥ {cfg.watch_q:.0%} / ≥ {cfg.disc_q:.0%} | 分數下降或 Probe 條件成立 |
| 1 PROBE | 可試單 → 已試單 → 等待確認 | 流動性 + 非處置 + score ≥ {cfg.probe_q:.0%} + 觸發 `{cfg.trigger}` + regime `{cfg.regime_filter}` + 停損距離 ≤ {cfg.max_stop_dist:.0%} | 確認 / 失敗 |
| 2A FAILED PROBE | 試單失敗 | F1 跌破 Probe Low；F2 收盤 < MA10 且 < 成本；F3 試單後相對大盤 ≤ {cfg.f3_thresh:.0%}；F4 N 日未上漲；最長 {cfg.probe_max_life} 日（依凍結組合 `{cfg.fail}`） | 冷卻 {cfg.cooldown} 日後可再被發現 |
| 2B CONFIRMED | 趨勢確認 / 可加碼 | price：收盤突破訊號日前 20 日高且創試單後新高；rs：RS line 40 日新高 + RS20 高於訊號日 + RS40 上升；persist：試單後 ≥60% 天數勝大盤、超額 ≥3%、勝產業 | 加碼成交 |
| 3 ADD | 可加碼 | 架構 C 第二段：確認後 ≥3 日、創確認後新高、趨勢成立、獲利 ≥3% | FULL |
| 4 FULL | 正式持股 | 部位達 1.0 | Exit 規則 |
| 5 EXIT | 出場 | HS 硬停損；TF 收盤連 2 日 < MA20；MS 收盤 < MA10 或跌破 10 日 swing low；MP 獲利曾 ≥{cfg.mfe_trigger:.0%} 後回吐一半 | — |

## 成交假設（真實可成交）

- 訊號一律以 t 日收盤資料計算，t+1 **開盤**成交；停損以盤中觸價、跳空時以開盤價成交。
- 開盤漲停鎖死 / 開盤 ≥ 前收 +9.5% 不買；跌停鎖死延後賣出。
- 每筆成交 ≤ 當日成交值的 {config.MAX_PARTICIPATION:.0%}。
- 成本：來回 {config.BASE_COST:.2%}（買賣各半）+ 單邊 {config.BASE_SLIPPAGE_BPS} bps 滑價；敏感度 0.30%–1.00% × 0/25/50 bps。
"""


def case_doc(case: dict, cfg) -> str:
    sym = case.get("symbol")
    if not case.get("found"):
        return f"# {sym} CASE STUDY\n\n{banner()}資料中沒有 {sym}：{case.get('reason')}\n"
    tl = case["timeline"]
    tl.to_csv(OUT / f"{sym}_TIMELINE.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    def fmt_ev(x):
        return "未發生" if not x else f"{x[0].date()} @ {x[1]:,.2f}（size {x[2]:.2f}；{x[3]}）"

    def d(x):
        return "未發生" if x is None else str(pd.Timestamp(x).date())
    fc = case["fail_counts"]
    cp = tl[tl["in_case_period"]]
    L = [f"# 3653 健策 CASE STUDY（{config.CASE_PERIOD[0]} ～ {config.CASE_PERIOD[1]}）\n\n{banner()}"
         f"> 只作 case study / regression test。策略 {cfg.version} 已在 Discovery 期凍結，**未用健策調整任何門檻**。\n\n"
         "## 1. PIT Timeline 摘要\n\n| 問題 | 答案 |\n|---|---|\n"
         f"| 期間內第一次「觀察」 | {d(case['first_watch'])}（含前置期最早：{d(case['first_watch_ctx'])}） |\n"
         f"| 第一次 Discovery（Emerging） | {d(case['first_emerging'])} |\n"
         f"| 第一次 Probe 條件成立 | {d(case['first_probe_signal'])} |\n"
         f"| Probe 成交（日期/價格/size/理由） | {fmt_ev(case['probe'])} |\n"
         f"| 試單期間最接近停損 | {pct(case['closest_stop']) if case['closest_stop'] is not None else 'n/a'}（收盤距停損） |\n"
         f"| Confirmation | {fmt_ev(case['confirm'])} |\n| Add | {fmt_ev(case['add'])} |\n| Add 2 | {fmt_ev(case['add2'])} |\n"
         f"| Full | {fmt_ev(case['full'])} |\n| Failed probe | {fmt_ev(case['failed'])} |\n| Exit | {fmt_ev(case['exit'])} |\n"
         f"| 真正脫離大盤（stock−market > 5 點且之後不再落後） | {d(case['breakaway'])} |\n"
         "\n※ 價格為還原價（spread 還原）；原始收盤價見 3653_TIMELINE.csv `raw_close`。\n"]
    L.append("\n## 2. Probe 條件逐項（期間內不成立天數）\n\n| 條件 | 不成立天數 |\n|---|---|\n")
    from strategy.signals import PROBE_CONDITION_LABELS
    for k, n in fc.items():
        L.append(f"| {PROBE_CONDITION_LABELS.get(k, k)} | {n} / {len(cp)} |\n")
    L.append("\n## 3. 每日狀態（期間內）\n\n")
    cols = [c for c in ("date", "raw_close", "stock_norm", "market_norm", "sector_norm", "regime", "disc_pct",
                        "emerging", "c_trigger", "c_stop_ok", "probe_signal", "campaign_state", "position_size", "stop",
                        "portfolio_state", "events") if c in tl.columns]
    L.append(md_table(cp[cols], max_rows=80))
    isum = case.get("intraday_summary")
    if isum is not None and len(isum):
        L.append("\n## 4. 盤中（1 分 K + 5 秒指數）\n\n")
        L.append(md_table(isum, max_rows=80))
    else:
        L.append("\n## 4. 盤中\n\n無已下載的分鐘資料（執行 `python -m data.download --case-intraday`）。\n")
    return "".join(L)


# ---------------------------------------------------------------------------
def final_report(p, cfg, res) -> str:
    """Data-driven draft answers to the 40 questions (numbers filled from this run)."""
    tm, psum, v = res["tm"], res["psum"].set_index("window"), res["verdicts"]
    mo, md = tm["STRICT_OOS"], tm["DISCOVERY"]
    lifts = res["lifts"]
    yearly = res["yearly"]
    yt = yearly[yearly["level"] == "trade"].set_index("year")
    yp = yearly[yearly["level"] == "portfolio(continuous)"].set_index("year")
    arch = res["arch"]
    costs = res["costs"]
    cap = res["capture"]
    plac = res["placebo"]
    case = res.get("case", {})

    def lift(pn, fam=None, feat=None, lab="leader20"):
        x = lifts[(lifts["period"] == pn)]
        if fam:
            x = x[x["family"] == fam]
        if feat:
            x = x[x["feature"] == feat]
        col = f"lift_{lab}_volctl"
        x = x.dropna(subset=[col]) if col in x else x.iloc[0:0]
        return x
    top_disc = lift("DISCOVERY").sort_values("lift_leader20_volctl", ascending=False)
    top_disc = top_disc[top_disc["family"] != "DISCOVERY_SCORE"].head(6)
    top_oos = lift("STRICT_OOS")
    top_oos = top_oos[top_oos["family"] != "DISCOVERY_SCORE"].set_index("feature")["lift_leader20_volctl"]
    ds_l = lift("DISCOVERY", "DISCOVERY_SCORE")
    os_l = lift("STRICT_OOS", "DISCOVERY_SCORE")
    os30 = lift("STRICT_OOS", "DISCOVERY_SCORE", lab="leader30")

    def fam_lift(fam, pn):
        x = lift(pn, fam)
        return x["lift_leader20_volctl"].max() if len(x) else np.nan
    inc = res["inc"].set_index("period")

    def archm(pn, var, col):
        x = arch[(arch["period"] == pn) & (arch["variant"] == var)]
        return x[col].iloc[0] if len(x) and col in x else np.nan

    def capr(thr, src="trade_level(unconstrained)", per="ALL"):
        x = cap[(cap["source"] == src) & (cap["threshold"] == thr) & (cap["period"] == per)]
        return (x["early_capture_rate"].iloc[0], x["episodes"].iloc[0], x["random_baseline"].iloc[0]) if len(x) \
            else (np.nan, 0, np.nan)
    c20, c30, c40 = capr(">=20%"), capr(">=30%"), capr(">=40%")
    o30 = capr(">=30%", per="STRICT_OOS")
    p30 = capr(">=30%", src="10slot_portfolio")
    stress = costs[(costs["period"] == "STRICT_OOS")].set_index(["round_trip_cost", "slippage_bps_per_side"])
    po = psum.loc["STRICT_OOS"]
    fp = res["fp"]
    fsmd = fp[(fp["section"] == "feature_SMD_false_vs_bigwinner") & (fp["period"] == "ALL")].copy()
    fsmd["abs"] = fsmd["smd"].abs()
    fsmd = fsmd.sort_values("abs", ascending=False).head(5)
    pr = plac.set_index(["period", "group"])
    na = "P25_ONLY" if cfg.probe_size <= 0.25 else "P50_ONLY"

    def q(n, text):
        return f"**{n}.** {text}\n\n"
    L = [f"# FINAL REPORT — EMERGING LEADER PROBE STRATEGY {cfg.version}\n\n{banner()}"
         f"資料最新交易日：{p.dates[-1].date()}；Strict OOS = {config.STRICT_OOS[0]} ～ {config.STRICT_OOS[1]}；"
         f"基準成本 {config.BASE_COST:.2%} 來回 + {config.BASE_SLIPPAGE_BPS}bps 單邊滑價。\n\n"
         "## 最終裁決\n\n| 元件 | 裁決 | 證據 |\n|---|---|---|\n"]
    for k, (vv, why) in v.items():
        L.append(f"| {k} | **{vv}** | {why} |\n")
    L.append("\n## 40 個問題\n\n")
    L.append(q(1, f"能否提早辨認？Discovery score 前 {1 - cfg.disc_q:.0%} 的 vol-controlled leader(+20%/40D) lift："
                  f"Discovery {num(ds_l['lift_leader20_volctl'].iloc[0] if len(ds_l) else np.nan)}、"
                  f"Strict OOS {num(os_l['lift_leader20_volctl'].iloc[0] if len(os_l) else np.nan)}（+30%："
                  f"{num(os30['lift_leader30_volctl'].iloc[0] if len(os30) else np.nan)}）。"))
    L.append(q(2, "Discovery 期 vol-controlled lift 最高的特徵：" + "；".join(
        f"`{r.feature}`({r.family}) {num(r.lift_leader20_volctl)} → OOS {num(top_oos.get(r.feature, np.nan))}"
        for r in top_disc.itertuples())))
    L.append(q(3, "大盤不漲、個股獨強（同 ATR/beta/超額報酬分層比較）：leader20 差異 Discovery "
                  f"{pct(inc.loc['DISCOVERY'].get('leader20_diff_indep_minus_upday', np.nan), 2)}、OOS "
                  f"{pct(inc.loc['STRICT_OOS'].get('leader20_diff_indep_minus_upday', np.nan), 2)}；"
                  f"獨立強勢事件相對同 10D 超額分位 lift：Discovery {num(inc.loc['DISCOVERY'].get('is_event_lift_vs_same_cex'))}、"
                  f"OOS {num(inc.loc['STRICT_OOS'].get('is_event_lift_vs_same_cex'))}。"))
    L.append(q(4, f"Downside Resilience（RESIL family 最佳 lift）：Discovery {num(fam_lift('RESIL', 'DISCOVERY'))}、"
                  f"OOS {num(fam_lift('RESIL', 'STRICT_OOS'))}。"))
    L.append(q(5, f"Upside Participation（UPPART）：Discovery {num(fam_lift('UPPART', 'DISCOVERY'))}、"
                  f"OOS {num(fam_lift('UPPART', 'STRICT_OOS'))}；ASYM：Discovery {num(fam_lift('ASYM', 'DISCOVERY'))}、"
                  f"OOS {num(fam_lift('ASYM', 'STRICT_OOS'))}。"))
    L.append(q(6, f"Sector Relative（SECTOR_REL）：Discovery {num(fam_lift('SECTOR_REL', 'DISCOVERY'))}、"
                  f"OOS {num(fam_lift('SECTOR_REL', 'STRICT_OOS'))}（詳見 SECTOR_RELATIVE_STRENGTH.csv 的 stock-vs-sector × sector-vs-market 分解）。"))
    L.append(q(7, f"凍結的 Probe：score ≥ {cfg.probe_q:.0%}、觸發 `{cfg.trigger}`、regime `{cfg.regime_filter}`、"
                  f"停損距離 ≤ {cfg.max_stop_dist:.0%}；Discovery score 特徵：{', '.join(cfg.disc_features)}。"))
    L.append(q(8, f"P25 vs P50（Strict OOS, PnL/100 slot-days）：A(P25→1.0) {num(archm('STRICT_OOS', 'A', 'pnl_per_100_slot_days'), 3)}、"
                  f"B(P50→1.0) {num(archm('STRICT_OOS', 'B', 'pnl_per_100_slot_days'), 3)}、C {num(archm('STRICT_OOS', 'C', 'pnl_per_100_slot_days'), 3)}；"
                  f"凍結選擇（Discovery 決定）：{cfg.add_arch}。"))
    L.append(q(9, f"每 100 次 Probe 失敗：Discovery {num(md.get('false_probe_rate', np.nan) * 100, 0)} 次、"
                  f"OOS {num(mo.get('false_probe_rate', np.nan) * 100, 0)} 次。"))
    L.append(q(10, f"平均 Failed Probe 損失（以試單資金計）：Discovery {pct(md.get('avg_failed_loss_ret'))}、"
                   f"OOS {pct(mo.get('avg_failed_loss_ret'))}（中位 {pct(mo.get('median_failed_loss_ret'))}、worst 5% "
                   f"{pct(mo.get('failed_worst5pct_ret'))}；以總權益計約 {pct(mo.get('avg_failed_loss_slots', np.nan) / 10, 3)}/筆）。"))
    L.append(q(11, f"Confirm 比例：Discovery {pct(md.get('confirmation_rate'))}、OOS {pct(mo.get('confirmation_rate'))}。"))
    L.append(q(12, "Confirmation 組合比較（OOS，僅報告不再選擇）：" + "；".join(
        f"{c} PnL/100sd={num(archm('STRICT_OOS', c, 'pnl_per_100_slot_days'), 3)} PF={num(archm('STRICT_OOS', c, 'pf'))}"
        for c in ("price", "rs", "persist", "price+rs", "price+persist")) + f"；凍結：`{cfg.confirm}`。"))
    L.append(q(13, f"Probe→Confirmation 平均 {num(mo.get('days_probe_to_confirm_mean'), 1)} 日（中位 "
                   f"{num(mo.get('days_probe_to_confirm_median'), 0)}）；Discovery {num(md.get('days_probe_to_confirm_mean'), 1)} 日。"))
    L.append(q(14, f"Add 後 payoff（OOS）：有加碼 {num(mo.get('payoff'))} vs 只試單({na}) {num(archm('STRICT_OOS', na, 'payoff'))}。"))
    L.append(q(15, f"PF（OOS）：有加碼 {num(mo.get('pf'))} vs 只試單 {num(archm('STRICT_OOS', na, 'pf'))}。"))
    for n_, k in ((16, "ge20"), (17, "ge30"), (18, "ge40")):
        L.append(q(n_, f"{k.replace('ge', '≥')}% Winner（OOS 筆數）：有加碼 {mo.get(k)} vs 只試單 {archm('STRICT_OOS', na, k)}"
                       f"（以 PnL 計，加碼把贏家的權重放大；筆數相同代表加碼不改變誰會贏，只改變贏多少）。"))
    L.append(q(19, f"Leader Capture Rate（全期，早期 Probe 到、價格未超過漲幅一半）：+20% {pct(c20[0])}（{c20[1]} 段，隨機基準 {pct(c20[2])}）、"
                   f"+30% {pct(c30[0])}（隨機 {pct(c30[2])}）、+40% {pct(c40[0])}（隨機 {pct(c40[2])}）；OOS +30% {pct(o30[0])}；"
                   f"10-slot 組合實際持有 +30% {pct(p30[0])}。"))
    L.append(q(20, "False Positive 主要來源（失敗 Probe vs ≥20% 贏家的標準化差異最大者）：" + "；".join(
        f"`{r.item}` 失敗中位 {num(r.false_probe_median, 3)} vs 贏家 {num(r.big_winner_median, 3)} (SMD {num(r.smd)})"
        for r in fsmd.itertuples())))
    s1 = stress.loc[(0.007, 25)] if (0.007, 25) in stress.index else None
    s2 = stress.loc[(0.0045, 50)] if (0.0045, 50) in stress.index else None
    s3 = stress.loc[(0.01, 50)] if (0.01, 50) in stress.index else None
    L.append(q(21, f"成本後（OOS）：0.70%+25bps PF {num(s1['pf'] if s1 is not None else np.nan)}、組合 CAGR "
                   f"{pct(s1['pf_cagr'] if s1 is not None else np.nan)}；1.00%+50bps PF {num(s3['pf'] if s3 is not None else np.nan)}、"
                   f"CAGR {pct(s3['pf_cagr'] if s3 is not None else np.nan)}。"))
    L.append(q(22, f"50bps 滑價（0.45% 成本）OOS：PF {num(s2['pf'] if s2 is not None else np.nan)}、組合 CAGR "
                   f"{pct(s2['pf_cagr'] if s2 is not None else np.nan)}、Sharpe {num(s2['pf_sharpe'] if s2 is not None else np.nan)}。"))
    L.append(q(23, f"10-slot 組合（OOS）：CAGR {pct(po['cagr'])}、MDD {pct(po['mdd'])}、Sharpe {num(po['sharpe'])}、"
                   f"平均占用 {pct(po['avg_occupancy'])}；同期 TAIEX {pct(po['taiex_return'])}（總報酬比較見 PORTFOLIO_SUMMARY.csv）。"))
    L.append(q(24, f"PnL/100 slot-days：trade-level OOS {num(mo.get('pnl_per_100_slot_days'), 3)} slot、"
                   f"組合 OOS {num(po.get('pnl_per_100_slot_days_pct'), 3)}%；只試單 {num(archm('STRICT_OOS', na, 'pnl_per_100_slot_days'), 3)}。"))
    L.append(q(25, f"Top 5 依賴（OOS）：前 5 筆占總 PnL {pct(mo.get('top5_trades_share'))}，去掉前 5 筆 PnL = "
                   f"{num(mo.get('pnl_ex_top5_slots'), 2)} slot；前 5% 交易占 {pct(mo.get('top5pct_share'))}。"))
    for n_, y in ((26, "2023"), (27, "2024"), (28, "2025"), (29, "2026YTD")):
        t_ = yt.loc[y] if y in yt.index else {}
        pp = yp.loc[y] if y in yp.index else {}
        L.append(q(n_, f"{y}：Probes {t_.get('n_probes', 'n/a')}、PF {num(t_.get('pf'))}、Payoff {num(t_.get('payoff'))}、"
                       f"失敗率 {pct(t_.get('false_probe_rate'))}、組合報酬 {pct(pp.get('return'))}（MDD {pct(pp.get('mdd'))}；"
                       f"TAIEX {pct(pp.get('taiex_return'))}）" + ("（Strict OOS）" if y in ("2025", "2026YTD") else
                                                                    "（Discovery，非 OOS）")))
    dd = lambda x: "未發生" if not x else (str(pd.Timestamp(x).date()) if not isinstance(x, tuple) else
                                         f"{x[0].date()} @ {x[1]:,.2f}")
    if case.get("found"):
        nat = bool(case.get("probe"))
        L.append(q(30, f"健策是否被自然發現：{'是' if case.get('first_emerging') else '否'}（Emerging）；"
                       f"{'有' if nat else '沒有'} Probe。"))
        L.append(q(31, f"第一次 Discovery：{dd(case.get('first_emerging'))}（觀察：{dd(case.get('first_watch'))}）"))
        L.append(q(32, f"第一次 Probe：{dd(case.get('probe'))}"))
        L.append(q(33, f"Confirmation：{dd(case.get('confirm'))}"))
        L.append(q(34, f"Add：{dd(case.get('add'))}；Full：{dd(case.get('full'))}"))
        fc = case.get("fail_counts", {})
        L.append(q(35, "若沒抓到的原因（期間內各條件不成立天數）：" + "、".join(f"{k}={v_}" for k, v_ in fc.items())))
    else:
        for n_ in range(30, 36):
            L.append(q(n_, "健策不在資料中。"))
    cs = res["case_success"]
    L.append(q(36, "成功提早抓到的大 Winner（OOS 優先）：" + "；".join(
        f"{r.stock_id} {r.name} {pd.Timestamp(r.probe_date).date()} → {pct(r.ret_on_invested)}"
        for r in cs.sort_values("ret_on_invested", ascending=False).head(8).itertuples())))
    cf = res["case_failed"]
    L.append(q(37, "代表性 False Probe：" + "；".join(
        f"{r.stock_id} {r.name} {pd.Timestamp(r.probe_date).date()} {pct(r.ret_on_invested)}（{r.why_failed}）"
        for r in cf.head(6).itertuples())))
    L.append(q(38, "FinMind 限制：見 FINMIND_DATA_AUDIT.md（帳戶額度端點被環境封鎖、分K 一次一檔一天、產業別為目前快照、"
                   "全市場單日資料含權證、資料非即時）。"))
    L.append(q(39, "Dashboard：`streamlit run dashboard/app.py`（見 README_DASHBOARD.md）。"))
    L.append(q(40, f"是否進 Paper Trading：FULL SYSTEM = **{v['FULL_SYSTEM'][0]}**、PROBE = **{v['PROBE'][0]}**。"))
    L.append("\n### 安慰劑（matched controls：同日、同 ATR/beta/流動性三分位、同 RS20 五分位、優先同產業，套用相同機制）\n\n")
    for pn in ("DISCOVERY", "STRICT_OOS"):
        try:
            a, b = pr.loc[(pn, "V1_PROBES")], pr.loc[(pn, "MATCHED_CONTROLS")]
            L.append(f"- {pn}: V1 EV/probe {num(a['ev_slots_per_probe'], 4)} PF {num(a['pf'])} vs controls "
                     f"{num(b['ev_slots_per_probe'], 4)} PF {num(b['pf'])}；bootstrap p = {num(a['bootstrap_p_probe_gt_control'], 3)}\n")
        except KeyError:
            pass
    return "".join(L)


def write_reports(p, F, cfg, res) -> None:
    (DOC / "FINMIND_DATA_AUDIT.md").write_text(data_audit(p, F), encoding="utf-8")
    (DOC / "EMERGING_LEADER_STRATEGY.md").write_text(strategy_doc(cfg, res), encoding="utf-8")
    (DOC / "EMERGING_LEADER_STATE_MACHINE.md").write_text(state_machine_doc(cfg), encoding="utf-8")
    if "case" in res:
        (DOC / "3653_CASE_STUDY.md").write_text(case_doc(res["case"], cfg), encoding="utf-8")
        ip = res["case"].get("intraday")
        if ip is not None and len(ip):
            ip.to_parquet(OUT / "signals" / "case_intraday.parquet", index=False)
        tl = res["case"].get("timeline")
        if tl is not None:
            tl.to_parquet(OUT / "signals" / "case_timeline.parquet", index=False)
    (DOC / "FINAL_REPORT.md").write_text(final_report(p, cfg, res), encoding="utf-8")
