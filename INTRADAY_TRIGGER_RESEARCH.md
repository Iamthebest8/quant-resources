# INTRADAY TRIGGER RESEARCH（1 秒 K 進場時機研究）

## 1. 定位與資料

**分工**：1 秒引擎**只決定何時進場**。選哪一檔、哪一天是候選日，全部由日線／週線 alpha 決定。

**資料**：

- FinMind `TaiwanStockPriceTick` 逐筆成交，聚合成 09:00:00–13:30:00 的 1 秒 OHLCV，共 16,201 秒。
- 盤後 14:30 的零股與盤後交易列印已剔除。
- 大盤參考：`TaiwanVariousIndicators5Seconds`（TAIEX 5 秒）。
- **沒有委託簿**：沒有 bid/ask、沒有 order flow，也不做假設。`TickType` 不使用。

**覆蓋**：

| 策略 | 候選日 | 來源 | 期間 |
|---|---|---|---|
| Momentum | 1,083 | 凍結 V1 試單日 | 2023-01–2026-09 |
| Weinstein | 925 | W1／W2／W3／S1／S2 TEXTBOOK 成交日 | 2023 以後 |

- 全部 tick 檔都已下載（`outputs/intraday/tick_requests_*.csv`）。
- 資料集：
  - `INTRADAY_ENTRY_EVENT_DATASET.csv` 是樣本：每個策略 4 個候選日，約 16,600 個決策點。
  - 完整資料集約 1,529 個候選日、每 15 秒一筆，存在 `outputs/phase2/intraday_dataset.pkl`（不進 git，可由 `python -m pipeline.phase2 intraday` 重建）。

**樣本切分**（依時間順序，不隨機）：

- TRAIN：2023-01-01–2024-12-31。
- TEST：2025-01-01–2026-09-03。以天數計約 35–40%，與日線的 Strict OOS 相同。

**決策點**：

- 從第一筆成交開始，每 15 秒一個決策點；收盤前 10 分鐘不再進場。
- 決策秒 t 只使用 ≤ t 的資料，於 **t+1 秒**以最後成交價進場，再加滑價。

**標的（研究標籤）**：

- FutureUtility = MFE(到收盤) − |MAE(到收盤)|（λ = 1）。
- 另報：+1／5／15／30／60 分鐘、收盤、次日報酬；日線停損與盤中微停損的觸發率。

## 2. 防止前視（重要）

- **突破型候選日**（W1／W2 突破、S1 跌破）之所以成為候選日，是因為「當天穿越觸發價」。
- 在這些日子裡，「OPEN 開盤就買」等於事先知道當天會突破，是**前視**。
- 因此突破型候選日的**基準是 A_BREAKOUT_IMMEDIATE**（日線 buy-stop 成交），LEARNED 也只能在第一次越過觸發價之後決策。
- 表中突破型候選日的 OPEN 數字只是參考（utility 2.5%），**不可當作可實現策略**。

## 3. 政策（每種只取「第一個」符合條件的秒）

**突破型變體 A–F**：

| 代號 | 規則 |
|---|---|
| A_BREAKOUT_IMMEDIATE | 第一次越過觸發價 |
| B_BREAKOUT_HOLD_60S | 越過後站穩 60 秒 |
| C_BREAKOUT_RETEST_HOLD | 越過後回測至觸發價 ±0.5–1% 再站上 |
| D_BREAKOUT_MICRO_HL | 越過後 1 分鐘級別出現更高低點 |
| E_BREAKOUT_VWAP_RECLAIM | 越過後收復 VWAP |
| F_BREAKOUT_REL_STRENGTH | 越過後 5 分鐘相對 TAIEX 強 0.5% |

**其他政策**：

- 不需要突破的：OPEN、VWAP_RECLAIM_ANY、MICRO_HL_ANY。
- **LEARNED**：
  - 模型：HistGradientBoosting（depth 4，200 棵，leaf ≥ 200），用 38 個只看過去的特徵預測 utility。
  - 門檻：在 TRAIN 的分數分位 {0.6, 0.7, 0.8, 0.9} 中，取「成交率 ≥ 50% 且 TRAIN utility 最高」者。
  - 凍結檔：`outputs/frozen/INTRADAY_TRIGGERS_V1.json`。

## 4. 結果（TEST = 2025-01–2026-09）

### Momentum（380 個 TEST 日；基準為 OPEN）

| 政策 | 成交率 | 中位進場時間 | Utility | MFE／MAE | 日線停損 ATR | 當日收盤報酬 |
|---|---|---|---|---|---|---|
| OPEN | 100% | 09:00:09 | **+0.17%** | 1.07 | 1.51 | −0.05% |
| A 突破即進 | 74% | 09:00:24 | +0.16% | 1.07 | 1.79 | +0.06% |
| E 突破後收復 VWAP | 68% | 09:03:53 | **+0.21%** | 1.10 | 1.78 | +0.13% |
| LEARNED | 87% | 09:02:22 | +0.04% | 1.02 | 1.44 | −0.06% |

- LEARNED 在 TRAIN 有 +0.39%，到 TEST 只剩 +0.04%，**不能泛化**。
- 與 OPEN 比較的 p = 0.70。
- 隔夜試單報酬（相同出場、只換進場價）：LEARNED −7.6% vs OPEN −10.6%，兩者都是試單本身的負報酬。

### Weinstein 突破（W1／W2；TRAIN 234 天、TEST 43 天；基準為 A 突破即進）

| 政策 | 成交率 | 中位進場時間 | Utility | MFE／MAE | 日線停損 ATR | 5 分鐘微停損 ATR／當日觸發 | 當日低點停損 ATR／觸發 |
|---|---|---|---|---|---|---|---|
| A 突破即進 | 100% | 09:06:32 | **−0.86%** | 0.69 | 4.10 | 0.54／33% | 1.18／17% |
| C 回測站穩 | 77% | 09:14:19 | −0.11% | 0.95 | 4.17 | 0.44／55% | 1.22／32% |
| E 收復 VWAP | 74% | 09:12:49 | −0.09% | 0.96 | 4.07 | 0.15／63% | 0.71／29% |
| LEARNED | 77% | 09:15:59 | **+0.02%** | 1.01 | 3.95 | 0.12／60% | 0.78／30% |

- LEARNED 對 A 的 p = 0.13，未達 0.10。
- **多日交易 EV**（精確重新模擬，進場價改為 1 秒成交價，停損與出場完全相同）：LEARNED +26.4% vs 突破即進 +21.3%。
- 但 LEARNED 只成交 77% 的日子，而且 n = 33，樣本很小。

### Weinstein 回測（W3）與放空（S1／S2）

| 引擎 | TRAIN 天數 | 結論 |
|---|---|---|
| W3（TEXTBOOK） | 0 | 原書「量縮 75%」規則在台股幾乎不成立，無法訓練 |
| S1 | 16 | 少於事前登錄的最低 40 天，**不訓練、不判決** |
| S2 | 4 | 同上 |

- S1 的 TEST 期仍有 103 天，政策描述統計見 `INTRADAY_TRIGGER_RESULTS.csv`。

## 5. 1 秒進場能不能縮小停損？（PART 25）

- 突破日的日線結構停損約 **4.1 ATR**。
- 1 秒結構停損（5 分鐘擺盪低點）約 **0.1–0.5 ATR**，當日低點約 0.7–1.2 ATR。**距離確實大幅縮小**。
- 但用這個停損做多日交易（同一批 Weinstein 突破，精確重新模擬）：

| 進場方式 | 停損 | EV | PF | 平均 R |
|---|---|---|---|---|
| 突破即進 | 微停損（0.66 ATR） | −3.3% | 0（n = 24） | −1.3R |
| LEARNED | 微停損（0.19 ATR） | −2.0% | 0（n = 25） | −2.0R |
| 突破即進 | 日線結構停損 | +21.3% | — | +1.4R |

- **結論**：停損距離可以從約 4 ATR 縮到 0.2–0.7 ATR，但台股突破後的正常波動會把它全部打掉。**縮小停損不會提高 R/R，反而摧毀右尾**。

## 6. Feature importance（permutation，TEST）

| 策略 | 最重要的特徵 |
|---|---|
| Momentum | `ret_open`、`ret_prevclose`、`dist_trigger`、`retest_depth`、`rel_open`（相對開盤與前收的位置） |
| Weinstein 突破 | `dist_vwap`、`dist_trigger`、`dist_lod`、`time_above_vwap`（相對 VWAP 與觸發價的位置） |

兩套策略重要的盤中特徵不同，**需要不同的 trigger**（PART 26）。

## 7. 判決（事前登錄的 `ONE_SECOND_EXECUTION`）

| 策略 | 判決 | 依據 |
|---|---|---|
| Momentum | **REJECT** | LEARNED utility < OPEN |
| Weinstein 突破 | **WATCH** | LEARNED > 突破即進，但 p = 0.13、n = 43 |
| 整體 | **REJECT** | 以樣本 ≥ 100 天的 Momentum 為主 |

**建議**：

- 1 秒引擎只保留為「避免最差的立即追價」的觀察工具：E／LEARNED 晚 7–10 分鐘進場、在 VWAP 之上，比立即追價少虧 0.8%。
- 先 paper trading 累積更多 Weinstein 突破日，**不建議**用於縮小停損。
