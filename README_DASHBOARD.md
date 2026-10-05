# Emerging Leader 中文 Dashboard — 啟動說明

Dashboard 只讀本地檔案（`outputs/*.csv`、`outputs/signals/*.parquet`、`data/cache/manifest.json`），
**不會**在啟動時下載資料，也**不會**讀取或顯示 FinMind Token。

## 1. 安裝

```bash
python -m pip install -r requirements.txt
```

## 2. 設定 Token（只需一次）

在專案根目錄建立 `.env`（已被 `.gitignore` 排除，不會進 Git）：

```
FINMIND_TOKEN=你的FinMindToken
```

可參考 `.env.example`。程式只透過 `python-dotenv` 讀取，Token 只放在 HTTP Authorization header，log 會自動遮罩。

## 3. 第一次建立資料與研究結果

```bash
python run_all.py
```

依序執行：

1. `python -m data.download --case-intraday` — FinMind 下載到 `data/cache/raw/`（第一次約 4,500 次小請求：逐檔日線 + 還原股價 + 指數 + 處置 + 市值 + 3653 分 K / 5 秒指數；之後只做增量）
2. `python -m pipeline.run_research` — PIT 資料集 → 特徵 → 事件研究 → Discovery 期選參並**凍結 V1** → 驗證 / Strict OOS / 成本 / 安慰劑 / Leader capture → 報表 → Dashboard 資料

已有快取時可用 `python run_all.py --skip-download`。

> 已凍結的 V1（`outputs/frozen/V1.json`）不會被重跑覆蓋。若看完 OOS 想改規則，必須 `python -m pipeline.run_research --refreeze V2`，並會記錄在 `STRATEGY_DECISION_LOG.csv`。

## 4. 啟動 Dashboard

```bash
streamlit run dashboard/app.py
```

瀏覽器開 <http://localhost:8501>。

| 頁面 | 內容 |
|---|---|
| 🏠 Emerging Leader Radar | 當日所有觀察 / Emerging / 可試單 / 已試單 / 等待確認 / 趨勢確認 / 可加碼 / 正式持股 / 試單失敗 / 出場 股票，含 Discovery Score、市場與產業相對強度、RS20/40/60、RS acceleration、Price Structure、ATR%、Turnover、處置狀態、Probe / Confirmation / Add Trigger |
| 📈 個股分析 | 還原 K 線 + Probe/確認/加碼/出場標記與停損、個股/大盤/產業 = 100 相對走勢、RS 與 Discovery 百分位、Position Building、Trigger Panel（通過/缺少哪些條件）、State Timeline |
| 💼 投資組合 | 10-slot 組合：Probe / Confirmed / Full 部位、Failed Probes、現金、Probe 資金、Slot 占用、權益 vs TAIEX |
| 🔬 研究結果 | Discovery 2023–2024 / Extended 2024–2026 / Strict OOS 2025–2026 / Year-by-Year：CAGR、MDD、PF、Payoff、Leader Capture、False Probe、PnL/Slot-Day、Right Tail、成本敏感度熱圖、安慰劑、決策紀錄 |
| 🔎 3653 健策 Case Study | 7/8/9 月逐日狀態、3653/大盤/產業 = 100、盤中 1 分 K + VWAP + 大盤 5 秒 + 類股指數 5 秒 + 相對價差 |
| 🔄 資料狀態 / 更新資料 | 最新資料日期、FinMind 最後更新時間、dataset 權限、**更新資料**按鈕 |

### Historical Replay

左側「觀察日期」滑桿可選任何歷史交易日。Radar、個股、組合、3653 頁只顯示**當日收盤時已知**的資料與狀態
（特徵皆為 trailing、橫斷面排名只用當日資料；狀態機事件只取 ≤ 該日者）。研究結果頁為全期間統計，不隨回放變動。

### 資料日期

側欄固定顯示「最新資料日期」與「FinMind 最後更新（UTC）」。資料是**日線收盤後**資料，不是即時行情。

## 5. 更新資料（增量）

Dashboard「🔄 資料狀態 / 更新資料」頁按「更新資料」，或命令列：

```bash
python -m pipeline.refresh
```

流程：讀取 `.env` → FinMind 增量下載（只抓快取最後日期之後的新交易日，每天 1 次全市場請求）→ 更新本地快取 →
用**凍結 V1** 重算最新訊號 / 狀態 / 組合 → Dashboard 自動清快取重新讀取。不會重新下載全部歷史，也不會重新選參。

## 6. 離線測試模式（合成資料）

```bash
EL_DATA_SOURCE=synthetic python run_all.py
EL_DATA_SOURCE=synthetic streamlit run dashboard/app.py
```

合成資料只用來測程式流程，Dashboard 會顯示紅色 SYNTHETIC 警示。

## 7. 測試

```bash
python -m pytest -q tests/test_pit.py     # 截斷資料重算比對，驗證無 look-ahead
```

## 8. 常見問題

- **找不到 Dashboard 資料** → 先跑 `python run_all.py`。
- **找不到 FINMIND_TOKEN** → 確認專案根目錄 `.env`。
- **FinMind 請求上限** → client 會自動等待重試（HTTP 402）。
- **雲端環境連不到 FinMind** → 需在環境 Network access 允許 `api.finmindtrade.com`。

## Phase 2（Weinstein／High R/R／1 秒／出場／整合）

### 重跑順序

每一步只讀寫本地快取。需要 FinMind 的步驟會從 `.env` 讀取 token，token 不會顯示在畫面上。

```bash
python -m data.download                 # 增量更新日線與還原價（從 2019-01 起）
python -m pipeline.phase2 state         # 建立 PIT 面板、特徵、幾何特徵、週線 Stage（約 3 分鐘）
python -m pipeline.phase2 momentum      # 凍結 MOMENTUM_LONG_BASELINE 重現
python -m pipeline.phase2 weinstein     # W1／W2／W3／S1／S2：TEXTBOOK、MODERNIZED、出場研究
python -m pipeline.phase2 highrr        # High R/R 特徵研究與 HIGH_RR_SCORE_V1（已凍結就沿用）
python -m pipeline.phase2 userspec      # 使用者規格的 HIGH_RR_STAGE2_SETUP（7 條件）
python -m data.short_data --ids-file outputs/phase2/short_stock_ids.csv   # 融券、借券資料
python -m pipeline.phase2 shortdata     # 放空可執行性（資料層）
python -m pipeline.phase2 portfolio     # 重疊分析、Hybrids、多 alpha 10-slot 投組
python -m pipeline.phase2 intraday      # 下載 tick 並做 1 秒研究；--no-download --reuse 可重用資料集
python -m pipeline.phase2 dash          # High R/R Radar（PART 47）與成功／失敗案例
python -m pipeline.phase2_report        # 十項裁決 → PHASE2_VERDICTS.csv
```

### 凍結檔與 Dashboard

- 凍結檔在 `outputs/frozen/`：
  - `HIGH_RR_SCORE_V1.json`
  - `BOLLINGER_EXIT_V1.json`
  - `INTRADAY_TRIGGERS_V1.json`
  - `MOMENTUM_LONG_BASELINE.json`
- 已存在的凍結檔**不會被覆寫**。
- Dashboard 新增六個分頁：🚀 Momentum Long、📗 Weinstein Long、📕 Weinstein Short、🎯 High R/R Radar、⏱ Intraday Replay（1 秒價格、VWAP、Bollinger、觸發價與停損、各政策的進場秒數）、🧾 Phase 2 判決。
