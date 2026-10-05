# FINMIND DATA AUDIT

產生時間：2026-10-05 16:40 UTC  
FinMind 本地快取最後更新：**2026-10-05T16:04:12Z**  
研究資料最新交易日：**2026-10-05**（非即時資料）

## 1. 連線與權限
- 認證：`.env` 的 `FINMIND_TOKEN`，只放在 HTTP `Authorization: Bearer` header；程式、CSV、Dashboard、log 均不含 Token（log 有自動遮罩 filter）。
- `api.finmindtrade.com`（資料 API）：可連線。
- `api.web.finmindtrade.com`（帳戶額度 user_info）：本環境 egress policy 仍封鎖 → 無法讀取帳戶等級與剩餘額度，改以實際 dataset 探測判斷權限。

## 2. Dataset 探測結果（實際呼叫，非假設）
| label | dataset | data_id | status | rows | note |
|---|---|---|---|---|---|
| 股票基本資訊 | TaiwanStockInfo |  | OK | 4329 | 上市/上櫃/興櫃清單與產業別 |
| 交易日曆 | TaiwanStockTradingDate |  | OK | 6936 |  |
| 個股日線 (單檔) | TaiwanStockPrice | 2330 | OK | 5 | 未還原，含 spread(漲跌價差) |
| 個股日線 (全市場單日 bulk) | TaiwanStockPrice |  | OK | 48587 | 需 backer/sponsor 等級 |
| 還原股價 | TaiwanStockPriceAdj | 2330 | OK | 5 |  |
| 加權指數日線 (TAIEX) | TaiwanStockPrice | TAIEX | OK | 5 |  |
| 櫃買指數日線 (TPEx) | TaiwanStockPrice | TPEx | OK | 5 |  |
| 報酬指數 | TaiwanStockTotalReturnIndex | TAIEX | OK | 5 |  |
| 下市櫃 | TaiwanStockDelisting |  | OK | 726 | survivorship |
| 處置股票 | TaiwanStockDispositionSecuritiesPeriod |  | OK | 189 |  |
| 除權息結果 | TaiwanStockDividendResult |  | OK | 17 |  |
| 減資參考價 | TaiwanStockCapitalReductionReferencePrice |  | OK | 38 |  |
| 市值 | TaiwanStockMarketValue | 2330 | OK | 5 |  |
| 暫停交易 | TaiwanStockSuspended |  | OK | 3 |  |
| 分K (KBar) | TaiwanStockKBar | 2330 | OK | 265 | 需 sponsor 等級 |
| 加權指數 5 秒 | TaiwanVariousIndicators5Seconds |  | OK | 3241 |  |
| 類股指數 5 秒 | TaiwanStockEvery5SecondsIndex |  | OK | 197701 |  |

## 3. 使用的 datasets 與覆蓋
| dataset | files | rows | min_date | max_date | last_update_utc |
|---|---|---|---|---|---|
| TaiwanStockInfo | 1 | 3665 | 2020-06-03 | 2026-10-05 | 2026-10-05T15:24:00Z |
| TaiwanStockTradingDate | 1 | 6936 | 1999-01-05 | 2026-12-31 | 2026-10-05T15:24:00Z |
| TaiwanStockDelisting | 1 | 726 | 1995-09-23 | 2026-10-01 | 2026-10-05T15:24:00Z |
| TaiwanStockPrice | 2142 | 2489524 | 2021-06-01 | 2026-10-05 | 2026-10-05T15:35:47Z |
| TaiwanStockTotalReturnIndex | 2 | 2602 | 2021-06-01 | 2026-10-05 | 2026-10-05T15:35:47Z |
| TaiwanStockMarketValue | 65 | 136073 | 2021-06-30 | 2026-10-05 | 2026-10-05T15:36:06Z |
| TaiwanStockDispositionSecuritiesPeriod | 6 | 3147 | 2021-06-01 | 2026-10-02 | 2026-10-05T16:04:11Z |
| TaiwanStockDividendResult | 6 | 2 | 2021-06-01 | 2021-06-01 | 2026-10-05T16:04:12Z |
| TaiwanStockCapitalReductionReferencePrice | 6 | 199 | 2021-06-28 | 2026-10-05 | 2026-10-05T16:04:12Z |
| TaiwanStockSuspended | 6 | 5184 | 2021-06-10 | 2026-08-13 | 2026-10-05T16:04:12Z |
| TaiwanStockKBar | 63 | 11805 | 2026-07-01 | 2026-09-30 | 2026-10-05T15:36:31Z |
| TaiwanVariousIndicators5Seconds | 63 | 204183 | 2026-07-01 | 2026-09-30 | 2026-10-05T15:36:47Z |
| TaiwanStockEvery5SecondsIndex | 63 | 12455163 | 2026-07-01 | 2026-09-30 | 2026-10-05T15:42:29Z |
| TaiwanStockPriceAdj | 2138 | 2478214 | 2021-06-01 | 2026-10-05 | 2026-10-05T16:04:11Z |

## 4. 股票池覆蓋

- 期間：2021-06-01 ～ 2026-10-05，1301 個交易日
- 普通股代號（4 碼、非 00 開頭、上市+上櫃；興櫃排除）：**2029** 檔
  - 上市 twse：1096；上櫃 tpex：932；其他/未知：1
- 2021-06 之後下市櫃、且有價格資料納入回測的普通股：**57** 檔（TaiwanStockInfo 保留已下市股票及其產業別；名單來源 TaiwanStockDelisting）
- 每日可交易 universe（價格≥10.0、20日均成交值≥30M、上市≥120日）：平均 **763** 檔/日
- 上市期間內缺漏（停牌/無成交）比例：1.24%

## 5. 還原 / 未還原

- `TaiwanStockPrice` 為**未還原**價格（顯示、漲跌停判斷、價格門檻用）。
- **重要發現**：FinMind `spread` 在除權息日為 0（例：2330 2024-09-12、2603 2023-06-30），若直接用 close/(close−spread) 會把除息日的真實漲跌（最多 ±10%）變成 0。
- 因此每日總報酬的來源順序：① `TaiwanStockPriceAdj` 收盤比（FinMind 還原，除權息日正確）→ ② spread 參考價（spread≠0 時）→ ③ 未還原收盤比（漲跌幅限制內）。還原價由每日報酬累乘重建；比值型特徵與 PIT 序列一致，無前視。
- 報酬來源筆數：PriceAdj 2,447,061、spread 8,498、raw 101；無法判定 669（視為 0）。spread=0 但收盤有變動（除權息日）13,181 筆。

交叉檢查（spread 法 vs FinMind PriceAdj，修正前的獨立比對，15 檔樣本）：

| stock_id | days | corr | mean_abs_diff_bp | days_diff_gt_20bp | max_abs_diff | worst_day | cum_ret_finmind_adj | cum_ret_spread_method |
|---|---|---|---|---|---|---|---|---|
| 2330 | 905 | 0.992 | 2.563 | 13 | 0.048 | 2024-09-12 | 5.052 | 4.224 |
| 2317 | 904 | 0.997 | 1.013 | 5 | 0.038 | 2025-07-31 | 1.941 | 1.891 |
| 2454 | 905 | 0.999 | 0.847 | 4 | 0.026 | 2025-01-02 | 8.946 | 9.741 |
| 3653 | 905 | 0.999 | 0.535 | 2 | 0.041 | 2023-08-23 | 18.989 | 18.340 |
| 2881 | 905 | 0.996 | 0.917 | 6 | 0.024 | 2024-09-09 | 2.485 | 2.694 |
| 1101 | 904 | 0.995 | 0.765 | 5 | 0.038 | 2025-08-14 | -0.162 | -0.136 |
| 2603 | 905 | 0.986 | 2.009 | 4 | 0.100 | 2023-06-30 | 2.493 | 2.294 |
| 3008 | 905 | 0.998 | 1.554 | 7 | 0.030 | 2024-08-15 | 2.689 | 2.785 |
| 6669 | 905 | 0.998 | 1.121 | 4 | 0.049 | 2023-06-14 | 7.907 | 7.527 |
| 3231 | 905 | 1.000 | 0.499 | 3 | 0.023 | 2026-07-08 | 6.307 | 6.548 |
| 8299 | 905 | 0.997 | 1.825 | 6 | 0.066 | 2025-12-16 | 6.117 | 6.438 |
| 5274 | 905 | 0.994 | 2.428 | 5 | 0.100 | 2026-06-26 | 11.809 | 12.303 |
| 6488 | 905 | 0.995 | 2.374 | 6 | 0.071 | 2026-07-16 | 2.000 | 2.197 |
| 1795 | 905 | 0.992 | 1.287 | 4 | 0.099 | 2024-08-05 | -0.230 | -0.136 |
| 2002 | 905 | 1.000 | 0.206 | 2 | 0.011 | 2026-07-24 | -0.327 | -0.338 |

## 6. 指數覆蓋

- 大盤：TaiwanStockPrice:TAIEX (OHLC)，2021-06-01 ～ 2026-10-05
- 櫃買指數：有

## 7. 產業覆蓋

- 來源：TaiwanStockInfo.industry_category（目前快照）
- 產業數：39；UNKNOWN（多為已下市股票，info 中已無資料）：1 檔
- 籠統分類「電子工業」：229 檔（例如 2330 被歸在此類），降低 sector-relative 的精細度。
- 產業成分 < 4 檔時 sector return 以大盤替代。
- 同一檔在 TaiwanStockInfo 有多列（例如興櫃→上櫃、或多重分類）時，V1 取「最具體（非籠統）」的分類；其中 27 檔與目前上市列的分類不同（例：1563 電機機械 vs 汽車工業）。V1 凍結後不再更動，列為 V2 資料改善項目。

## 8. 處置股票覆蓋

- 來源：TaiwanStockDispositionSecuritiesPeriod，普通股處置事件 2376 筆（2021-06-01 ～ 2026-10-02）
- `disposition_status`：是否處置、period_start/end、撮合間隔（由 measure 文字解析；未載明時第一次=5分、第二次=20分）、限制（預收款券/人工管制）。
- 處置期間及隔日的量能特徵設為 **NOT_COMPARABLE**（NaN），不當作籌碼訊號；處置中或隔日將處置的股票不新開 Probe。

## 9. 盤中 / 分鐘資料

- TaiwanStockKBar（1 分 K）可用，但一次只能抓「一檔 × 一天」→ 全市場 × 3.7 年約 200 萬次請求，不可行。只下載 3653 個案期間：63 天。
- 加權指數 5 秒：63 天；類股指數 5 秒：63 天（個案用）。
- 全市場研究的盤中特徵改用日線 proxy：open→close、收盤位置 CLV、low→close recovery、相對大盤版本 → **INTRADAY_LIMITATION**。

## 10. Survivorship bias

- 已納入 2021-06 之後下市櫃的普通股 57 檔（價格資料至下市日，產業別仍可取得）。
- 回測中持股遇停止交易 → 以最後收盤價出場（標記 DELISTED/SUSPENDED）。
- 殘留風險：2021-06 之前下市者不影響 2023+ 研究；下市前最後幾日的流動性/處置限制以日線近似。
- datalist 中不在 info/delisting 名單的普通股代號：n/a 個（多為更早下市，未下載）。
- 判定：**SURVIVORSHIP_BIAS_RISK = LOW（已處理下市股；產業別有殘留缺口）**

## 11. PIT 風險

- 產業別為目前快照，歷史改類無法重建 → PIT_RISK（中低）。
- 處置資料以公告日 `date` 為已知時點；處置期間從公告後開始 → PIT 正確。
- 市值（turnover 分母）用月底快照向後填補 → PIT 正確。
- 所有特徵皆為 trailing window / 當日橫斷面排名；訊號 t 日收盤產生、t+1 開盤成交。`tests/test_pit.py` 以截斷資料重算比對。
- 研究標籤（未來報酬）只在 research/ 使用，strategy/ 不 import。

## 12. API 限制

- 帳戶額度端點被環境封鎖 → 無法顯示剩餘額度；client 內建 402 rate-limit 等待重試。
- 全市場單日 `TaiwanStockPrice` 含權證（約 4.8 萬列/日），歷史改以逐檔下載，增量更新才用單日全市場。
- KBar / 5 秒資料一次一天。
- 資料非即時：最新為上一個收盤後 FinMind 更新的日線。
- 高價股（如 3653 單價數千元）以金額計部位，假設可用盤中零股成交；零股流動性較整股差 → EXECUTION_RISK。
