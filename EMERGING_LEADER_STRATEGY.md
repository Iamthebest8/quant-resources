# EMERGING LEADER PROBE STRATEGY — V1

凍結時間：2026-10-05T16:07:23Z（Discovery 2023–2024 選定，**在任何 2025–2026 結果計算前凍結**）  
Config hash：`eba0cff4a970`

> 核心：DISCOVER EARLY · RISK SMALL · ADD ONLY WHEN RIGHT · LET WINNERS MATTER

## 1. 資料流程

```
FinMind API (data/finmind_client.py)
    ↓
Raw Local Cache (data/cache/raw/*.parquet, manifest.json)
    ↓
Clean / PIT Dataset (engine/panel.py：PriceAdj/spread 總報酬、universe、產業、處置)
    ↓
Research Engine (engine/features.py, research/*)
    ↓
Strategy Engine (strategy/engine.py 狀態機)
    ↓
Portfolio Simulator (strategy/portfolio.py 10-slot)
    ↓
中文 Dashboard (dashboard/app.py)
```

## 2. 凍結規則（V1）

**Discovery score** = 下列特徵橫斷面百分位的等權平均（每個 family 取 Discovery 期間 vol-controlled lift 最高者，lift < 1.10 的 family 不納入）：

- STRUCT: `hl_flag`（Discovery lift 1.2582）
- MKT_REL: `resid_5`（Discovery lift 1.2152）
- PERSIST: `cex_10`（Discovery lift 1.21）
- SECTOR_REL: `srs_5`（Discovery lift 1.1903）
- INDEP: `is_cnt10`（Discovery lift 1.1553）

| 區塊 | 凍結值 |
|---|---|
| 觀察 / Emerging | score ≥ 80% / ≥ 90% 百分位 |
| Probe 分數門檻 | ≥ 95% 百分位 |
| Probe 觸發 | `indep` |
| 大盤條件 | `all` |
| 停損距離 | ≤ 8%（Probe Low = 近 3 日最低） |
| 追價限制 | 開盤 > 訊號收盤 +6% 不進；漲停鎖死不進 |
| Failed Probe | `F1+F2+F3+F4`（時間停損 5 日），最長試單 20 日 |
| Confirmation | `persist`（試單後至少 2 個收盤） |
| Add 架構 | `A`（Probe 0.25 → 1.00） |
| Exit | `MP` |
| 同時 Probe 上限 | 2 |
| 資金架構 | A（10 slots 共用） |
| 排名 | Part 23 固定等權：relative persistence、sector RS、RS acceleration、price progress/ATR、relative high formation、capital velocity |

## 3. 主要結果（基準成本 0.45% 來回 + 25bps 單邊滑價）

| 期間 | Probes | PF | Payoff | 勝率 | EV/probe(slot) | PnL/100 slot-days | 失敗率 | 確認率 | ≥20% | 組合CAGR | MDD | Sharpe | TAIEX |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DISCOVERY | 695 | 1.96 | 10.87 | 15.3% | 0.0128 | 0.259 | 78.0% | 21.9% | 36 | 11.4% | -8.0% | 1.41 | 61.9% |
| EXTENDED | 731 | 1.13 | 6.20 | 15.5% | 0.0020 | 0.049 | 77.6% | 21.8% | 25 | 1.6% | -11.4% | 0.23 | 156.9% |
| STRICT_OOS | 385 | 1.12 | 4.80 | 19.0% | 0.0019 | 0.044 | 76.4% | 22.3% | 14 | -0.3% | -9.2% | 0.01 | 100.8% |

## 4. 最終裁決（依 pipeline/verdicts.py 預先登錄之標準）

| 元件 | 裁決 | 證據 |
|---|---|---|
| DISCOVERY | **REJECT** | OOS lift20=1.18 lift30=1.27 (Discovery lift20=1.26) MAE10 lift=1.11 |
| PROBE | **WATCH** | OOS PF=1.12 EV/probe=0.0019 slots n=385 placebo p=0.488 |
| FAILED_PROBE_EXIT | **ACCEPT** | OOS avg=-2.68% worst5%=-8.49% median hold=3.0 (Discovery avg=-3.52%) |
| CONFIRMATION | **WATCH** | OOS add-leg PF=1.22 P(>=20%|conf)=11.63% vs all=3.64% add premium med=8.64% R/R med=1.00 |
| ADD | **ACCEPT** | OOS PnL with add=0.73 vs probe-only=0.18 slots; eff 0.044 vs 0.026; PF 1.12 vs 1.06; Calmar -0.03 vs -0.19 |
| FULL_SYSTEM | **REJECT** | CAGR>0:✘; Sharpe>=0.8:✘; MDD>=-25%:✔; PF>=1.3:✘; stress CAGR>0:✘; stress PF>=1.1:✘; ex-top5>0:✘; 2025>0:✘; 2026YTD>0:✘; PROBE!=REJECT:✔ |

## 5. 預先登錄的判定標準

- **DISCOVERY** — ACCEPT: OOS vol-controlled lift(leader20) >= 1.5 AND lift(leader30) >= 1.5 AND OOS lift >= 0.7 x Discovery lift AND lift(MAE<=-10%) < lift(leader20)；WATCH: OOS vol-controlled lift(leader20) >= 1.2；其餘 REJECT
- **PROBE** — ACCEPT: OOS trade-level PF >= 1.3 AND EV/probe > 0 AND >= 50 probes AND EV beats matched-control placebo (bootstrap one-sided p < 0.10)；WATCH: OOS PF >= 1.0 AND EV/probe > 0；其餘 REJECT
- **FAILED_PROBE_EXIT** — ACCEPT: OOS avg failed-probe return >= -6% AND worst-5% >= -12% AND median failed holding <= 10D AND OOS avg failed loss not worse than 1.5 x Discovery；WATCH: OOS avg failed-probe return >= -8%；其餘 REJECT
- **CONFIRMATION** — ACCEPT: OOS add-leg PF >= 1.3 AND P(ret>=20% | confirmed) >= 2 x P(ret>=20% | all probes) AND median add premium <= 10% AND median add reward/risk >= 1.5；WATCH: OOS add-leg PF >= 1.0；其餘 REJECT
- **ADD** — ACCEPT: OOS with-add vs probe-only (same probe size): total PnL higher AND PnL/100 slot-days higher AND PF not more than 10% lower AND portfolio Calmar not lower；WATCH: OOS with-add total PnL higher；其餘 REJECT
- **FULL_SYSTEM** — ACCEPT: OOS 10-slot portfolio @0.45%+25bps: CAGR > 0, Sharpe >= 0.8, MDD >= -25%, campaign PF >= 1.3; @0.70%+50bps: CAGR > 0 and PF >= 1.1; PnL ex-top-5 campaigns > 0; 2025 and 2026YTD both > 0; PROBE verdict not REJECT；WATCH: OOS portfolio CAGR > 0 and campaign PF >= 1.1；其餘 REJECT

## 6. 研究版本紀律

- V1 由 Discovery（2023–2024，標籤 purge 40 日、campaign 於 2024-12-31 強制結算）選定後凍結。
- 2024 同時屬於 Discovery 與 Extended Validation → 2024 **不是** OOS。
- 2025-01-01 ～ 2026-09-03 為 Strict OOS；看到 OOS 後若修改規則必須以 `--refreeze V2` 建立新版本並記錄於 STRATEGY_DECISION_LOG.csv。
- 健策 3653 只作 case study / regression test，未參與任何門檻選擇。

## 7. 研究流程揭露

1. 全部程式（特徵、狀態機、選參流程、判定標準）先以**離線合成資料**開發與測試，接上 FinMind 前未看過任何真實結果。
2. 接上真實資料後、**凍結前**做的兩項修改：(a) 發現 FinMind `spread` 在除權息日為 0，改用 `TaiwanStockPriceAdj` 計算每日報酬；(b) Discovery score 最多 5 個 family（避免黑箱）。兩者都在第一次真實 Discovery 選參之前。
3. **凍結後**的重跑只有報表/工程修正（事件價格改列實際成交價、live 模式不強制平倉、文字），設定 hash 不變、OOS 數字不變，每次都記錄在 STRATEGY_DECISION_LOG.csv（rerun_note）。
4. 看過 OOS 後觀察到的改進方向（Exit 讓贏家跑、避免太早、ATR 停損）只列為 V2 假說，**沒有**套用到 V1。
