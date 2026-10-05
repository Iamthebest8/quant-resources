# HIGH R/R FEATURE RESEARCH（高賺賠比特徵研究）

> **方法**（事前登錄見 `WEINSTEIN_QUANT_RULES.md` §L 與 `STRATEGY_DECISION_LOG.md` §A2）
>
> - 事件面板：每週末收盤時，宇宙內週線 Stage 1／2 的股票（排除處置股）。每組約 2–6 萬個 stock-week。
> - 進場：下一交易日開盤。
> - 結構停損：近 20 日低 × 0.99。
> - 結果：40 日停損感知報酬 `ret_s40`、R 倍數 `r_s40`、`ge20`（40 日 ≥ +20%）、`up20_before_stop`（先 +20% 才碰停損）。
> - 十分位切點只用 DISCOVERY（2023–2024）。
> - PRE（2020–2022）是額外的樣本內歷史，STRICT_OOS（2025-01–2026-09）**只看、不挑**。
>
> 數據檔：
>
> - `outputs/phase2/highrr_deciles_all.csv`（所有特徵 × 十分位 × 期間）、`outputs/phase2/highrr_feature_summary.csv`、`outputs/phase2/PRE_TRADE_RR_ANALYSIS.csv`。
> - 各 family 的 CSV：`OVERHEAD_RESISTANCE_ANALYSIS.csv`、`RS_LEADS_PRICE_ANALYSIS.csv`、`DOWNSIDE_RESILIENCE_ANALYSIS.csv`、`SECTOR_LEADERSHIP_ANALYSIS.csv`、`BASE_QUALITY_ANALYSIS.csv`、`COMPRESSION_EXPANSION_ANALYSIS.csv`、`VOLUME_PATTERN_ANALYSIS.csv`、`TREND_EXTENSION_ANALYSIS.csv`。

## 0. 一句話結論

**台股 2020–2026 的「高賺賠比」主要來自右尾**：

- 靠近新高、上方沒有套牢、RS 強、上漲參與度高、量能擴張的 Stage 2 股票，40 日的平均報酬與 P(+20%) 較高。
- 「上方空間大／停損近」這種傳統 PreTradeRR 反而**負相關**。
- 「下跌抗跌」與「波動壓縮」單獨使用**沒有增量**。
- 唯一在三個期間都穩定有效的突破條件是**基底量縮**（C5）。

## 1. PreTradeRR（上方空間 / 結構停損）

| 特徵 | DISCOVERY 十分位單調性 | PRE | STRICT_OOS | 解讀 |
|---|---|---|---|---|
| `rr_h250`（距 250 日高 / 停損距離） | **−1.00** | −0.78 | −0.98 | 三期一致**負向**：D1（RR 0.13）的 `ret_s40` 為 5.0%／8.2%，D10（RR 16）只有 0.4%／1.8%（DISCOVERY／OOS） |
| `room_h250`（距 250 日高的空間） | −1.00 | −0.20 | −0.94 | 空間越大越差；靠近高點最好 |
| `rr_res`（距最近壓力 / 停損距離） | +0.08 | −0.41 | −0.25 | 無訊號 |
| `res_dist_capped`（距最近壓力） | −0.05 | +0.10 | +0.40 | 只有 D10（blue sky，上方無壓力）明顯較好 |
| `stop_dist_atr`（停損距離，以 ATR 計） | +0.96 | +0.25 | +0.92 | 停損**越遠**平均報酬越高（波動大的股票右尾較厚）。不過停損越近，R 倍數並沒有更高 |

**結論**：

- 「上方空間 / 停損近」不能預測 payoff。
- 原因：上方空間大的股票，通常是剛從高點跌下來、上方有大量套牢的股票。它們的空間是「需要先修復」的，不是「很快可以實現」的。
- 這與 Weinstein 原書「越舊的壓力越弱、處女地最好」（p.98–99）一致，與「空間越大越好」的直覺相反。

## 2. Overhead Resistance / Supply（上方壓力與套牢量）

| 指標 | DISCOVERY 單調性 | PRE | OOS |
|---|---|---|---|
| `resdays10`（過去一年收盤落在 +10% 內的天數） | −0.94 | −0.77 | −0.95 |
| `supply15`（+15% 內的成交量占比） | −0.86 | +0.39 | −0.73 |

`blue_sky = 1`（上方沒有已確認的壓力 pivot）：

| 期間 | `ret_s40`（blue sky vs 其他） | 停損率（blue sky vs 其他） |
|---|---|---|
| DISCOVERY | 4.2% vs 1.7% | 19% vs 50% |
| OOS | 7.9% vs 4.1% | 22% vs 55% |
| PRE | 1.8% vs 1.9%（相近） | — |

**結論**：上方壓力與套牢量確實會降低報酬、提高停損率，方向符合原書。`resdays10` 三期一致，是最穩健的壓力指標。

## 3. Relative Strength

- RS 系列在 DISCOVERY 與 OOS 都是正向：`rs60` +0.98／+0.88、`rs120_pct` +0.84／+0.96、`rs_slope13` +0.99／+0.79。
- PRE 期間較弱。2020–2022 是大起大落的市場，`rs20` 與 `rs40` 在 PRE 為負。
- **RS 有增量**，但會因市場狀態而變弱。

**Weekly RS 截面 Spearman IC 多為負**（例如 `rs120_pct` 為 −0.09），但十分位平均報酬為正：

- IC 看的是每週的「排名中位數」關係，平均數則被右尾主導。
- 也就是說，強 RS 股票的**中位數**表現略差，但**大贏家**較多。
- 這是本研究反覆出現的「右尾型 alpha」特徵。

## 4. RS Leads Price（RS 先創新高、價格還沒）

| 期間 | RS_LEADS_PRICE | RS 與價格同時新高 | 無 RS 新高 |
|---|---|---|---|
| PRE | **4.6%** | 1.5% | 1.8% |
| DISCOVERY | 1.2% | **3.8%** | 1.9% |
| OOS | 6.2% | **9.6%** | 4.3% |

**結論**：

- RS 領先價格只在 PRE 最好。
- DISCOVERY 與 OOS 都不如「RS 與價格同時創新高」。
- **沒有穩定增量**。
- 在 Stage 1 突破事件上（C2，§8），PRE 與 DISCOVERY 為負，OOS 為正（n = 13）。

## 5. Downside Resilience vs Upside Participation（`ret_s40`）

| 類型 | PRE | DISCOVERY | OOS |
|---|---|---|---|
| 只有下跌抗跌 | 1.9% | **1.7%**（最差） | **1.5%**（最差） |
| 只有上漲參與 | 1.6% | 2.9% | **11.5%** |
| 兩者皆有（Asymmetric） | 2.7% | 2.4% | 6.9% |
| 都沒有 | 1.7% | 2.0% | 5.8% |

- `dcap60`（下跌捕獲率）的十分位單調性為正：**越不抗跌**反而越好。這與 Phase 1 的發現一致。
- `ucap60` 與 `up60` 在 DISCOVERY 與 OOS 都是 +0.95 以上。

**結論**：Downside Resilience **沒有增量**；Upside Participation **有增量**，而且是最強的單一特徵之一。

## 6. Sector Leadership

| 類型 | PRE ret／P(20%) | DISCOVERY | OOS |
|---|---|---|---|
| INDEPENDENT leader（個股強、族群弱） | **4.1%／21%** | **2.9%／19%** | 5.0%／24% |
| SECTOR_CONFIRMED（個股強、族群強） | 1.4%／16% | 1.8%／17% | **6.0%／26%** |
| 非領導股 | 1.6%／12% | 1.9%／11% | 4.3%／18% |

**結論**：

- 兩種領導股的 P(+20%) 都比非領導股高 40–70%。
- INDEPENDENT leader 在 PRE 與 DISCOVERY 的右尾最厚，SECTOR_CONFIRMED 在 OOS 較好。
- **族群強不是必要條件**。

## 7. Compression → Expansion、Volume、Base、Extension

**Compression**

- 單獨的壓縮（ATR5/20 < 0.8）在三期都**較差**：DISCOVERY 1.8% vs 2.1%，OOS 1.9% vs 5.4%。
- 「壓縮後放量」的樣本很少（46–160 筆）且結果不一致。
- `atr5_20`、`bbw_pct120` 的十分位顯示**波動擴張**（較高值）較好。

**Volume**

- 只有量縮（DRYUP_ONLY）的組合，停損率最高（60–66%），P(+20%) 最低。
- 只有放量擴張（EXPANSION_ONLY）的組合，`up20_before_stop` 最高（31–41%）。
- 在泛用面板中，「量縮後擴張」沒有額外好處。
- 但在 **Stage 1 突破事件**中，**基底量縮（C5）三期都正向**（§8）：「量縮」要搭配「基底 + 突破」才有意義。

**Base**

- `base_weeks`（基底週數）的單調性在三期都接近 0：**基底時間長短沒有用**。
- `base_tight10w`（10 週區間寬度）在 DISCOVERY 為 −0.76（越緊越好），在 OOS 卻為 +0.67。**不穩定**。

**Extension**

- `dist_ma150`、`ext_ma60_atr` 在 DISCOVERY 與 OOS 為正：越延伸平均越好，這是動能效應。
- 「避免追太晚」**沒有**被資料支持。
- `stage2_age` 單調性很低（−0.27／+0.58／+0.23），**沒有明顯的最佳區間**。
- Stage 2 整體優於 Stage 1（三期 `ret_s40`：2.0% vs 1.3%、2.2% vs 1.0%、5.5% vs 2.3%）。**原書「只買 Stage 2」得到確認**。

## 8. 使用者規格 HIGH_RR_STAGE2_SETUP（PART 38，7 條件）

事件基底：所有 Stage 1 → 2 基底突破（n = 237／264／286）。進場為突破日次日開盤，停損為 10 日低，TEXTBOOK_NOVOL 出場。

| 組別 | PRE PF／EV | DISCOVERY | OOS |
|---|---|---|---|
| 全部 Stage 1 突破 | 2.97／14.7% | 1.66／5.4% | 4.27／27.7% |
| 同日隨機股票對照 | 1.97／5.2% | 2.57／8.3% | 1.91／5.2% |
| 突破 vs 對照（bootstrap p） | **0.020** | 0.725 | **< 0.001** |
| **7 條件全滿足** | n = 1 | n = 0 | n = 0 |
| 符合 ≥ 4 個條件 | n = 15，PF 10.2 | n = 2 | n = 13，PF 6.65 |
| C5 基底量縮 = 1 vs 0 | **35.7% vs 8.3%** | **7.0% vs 4.9%** | **35.0% vs 25.2%** |
| C4 壓縮 = 1 vs 0 | 27.4% vs 7.9% | 4.5% vs 6.0% | 27.4% vs 27.8% |
| C3 低上方壓力 = 1 vs 0 | 21.7% vs 10.1% | −6.1% vs 7.0% | 61.2% vs 22.4% |
| C2 RS Leads = 1 vs 0 | −3.8% vs 16.3% | −11.7% vs 5.7% | 76.7% vs 25.3%（n = 13） |
| C6 突破放量 = 1 vs 0 | 13.3% vs 17.0% | 4.4% vs 7.5% | 35.0% vs 15.1% |
| C7 緊停損 = 1 vs 0 | 35.2% vs 13.4% | 2.5% vs 5.6% | 28.4% vs 27.6% |

**結論**：

- **「7 條件全滿足」幾乎不會發生**（7 年只有 1 筆），**無法驗證，也無法交易**。
- 條件越多不代表越好。符合 4 個以上條件的組合在 PRE 與 OOS 很好，但 DISCOVERY 只有 2 筆且都虧損。
- **只有 C5（基底量縮）三期一致正向**。
- 未過濾的 Stage 1 突破在 PRE 與 OOS 顯著優於同日隨機股票，但 DISCOVERY（2023–2024）不成立。

## 9. HIGH_RR_SCORE_V1（凍結分數）

**組成**：選擇規則是 DISCOVERY 單調性加上 PRE 同向，選出以下 5 個特徵：

- `room_h250`（−）
- `rs_slope13`（+）
- `resdays10`（−）
- `srs_20`（+）
- `dist_ma150`（+）

**泛用面板（前三分之一 vs 其他）**：

| 期間 | `ret_s40` | `up20_before_stop` | 停損率 |
|---|---|---|---|
| PRE | 2.7% vs 1.5% | 33% vs 20% | 33% vs 59% |
| DISCOVERY | 3.4% vs 1.3% | — | — |
| OOS | 6.7% vs 3.6% | — | — |

- 三期一致較好，但 payoff 較低（1.8–2.2 vs 2.9–4.0）：分數挑到的是「較少停損、較多 +20%」的股票。

**當作 Weinstein 交易過濾器（事前登錄的判決依據）：REJECT**

- 依據：OOS 前三分之一 EV 為 8.2%，全部為 21.6%。W1 的兩筆超大贏家都落在後三分之二。
- W2：DISCOVERY 前三分之一較差（3.1% vs 5.8%），OOS 相近。
- 動能：DISCOVERY 較差，OOS 較好（PF 0.85 vs 0.62）。

**HIGH_RR_STAGE2_SCORE_V1**（分數前三分之一 + Stage 2 + 停損 ≤ 2.5 ATR）與「同週隨機 Stage 2 股票」相比：

| 期間 | EV（TEXTBOOK_NOVOL） | 對照 EV | p 值 |
|---|---|---|---|
| PRE | 10.2% | 9.1% | 0.30 |
| DISCOVERY | 6.0% | 8.5% | 0.79 |
| OOS | 18.6% | 19.3% | 0.59 |

**沒有增量**。它的高 PF 來自「Stage 2 + 30 週趨勢出場」本身在多頭市場的 beta。

## 10. 失敗案例（`FAILED_HIGH_RR_SETUPS.csv`）

**分數前三分之一但失敗**的案例（停損先觸發且報酬為負，對照組是同期先 +20% 的成功案例；三期方向一致）：

| 特徵 | 失敗組 vs 成功組 |
|---|---|
| 停損距離 ATR | **較近**：3.0–3.2 vs 4.2–4.4。停損太近最容易被洗掉 |
| 延伸度 `ext_ma60_atr` | **較低**：2.7–2.9 vs 3.5–3.7。失敗並不是因為追太高 |
| 突破量 `brk_vol` | **較小**：1.25–1.30 vs 1.59–1.63 |
| 個股相對族群 `srs_20` | **較弱**：0.08–0.11 vs 0.16–0.19 |
| 上方套牢 `resdays10`、`supply15` | **較多** |
| Stage 2 年齡 | **較老**：81–119 vs 69–104 個交易日 |

失敗原因分類：

| 類別 | 筆數 |
|---|---|
| `SECTOR_WEAK` | 1,722 |
| `STOCK_SPECIFIC` | 832 |
| `TOO_EXTENDED` | 246 |
| `MARKET_DOWN` | 131 |
| `STOP_TOO_TIGHT` | 69 |

**使用者規格 setup（≥ 4 條件）的失敗**另做對照：

- 控制變數：beta、ATR%、成交值（流動性）、大盤 Stage、族群 Stage、停損距離。
- 失敗率在大盤 Stage 3 時較高：OOS 為 80%（n = 10），Stage 2 為 40%（n = 5）。
- beta、ATR、流動性差異不大。
- 樣本很小，僅供參考。

## 11. 對「核心哲學」的回答

| 你想找的 | 資料的回答 |
|---|---|
| 剛開始變強 | Stage 1 → 2 突破在 PRE 與 OOS 有顯著 alpha，DISCOVERY 沒有。Stage 2 普遍優於 Stage 1 |
| 上方空間大 | **否**：上方空間大反而差，靠近新高、上方無套牢較好 |
| 下方停損近 | **否**：停損近，R 倍數沒有更高，平均報酬更低 |
| 結構漂亮 | 基底長度無用。基底量縮有用（只在突破事件上）。基底緊縮不穩定 |
| RS 領先 | RS 強有用；「RS 領先價格」不穩定 |
| 波動收縮後擴張 | 收縮本身無用，擴張與高波動較好 |
| 突破有量 | 突破日 2 倍量在 PRE 與 DISCOVERY 為負，OOS 為正。**不穩定** |
| 盤中高品質 entry | 見 `INTRADAY_TRIGGER_RESEARCH.md` |
