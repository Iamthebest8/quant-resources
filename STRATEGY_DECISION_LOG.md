# STRATEGY DECISION LOG（Phase 2：Weinstein / High R/R / 1 秒執行 / 出場 / 整合）

**原則**：

- 每個決策都記錄時間、依據的資料窗、版本標籤與 commit。
- 2025–2026（Strict OOS）結果**不能**用來修改規則；若在看到 OOS 之後改動，一律換新版本標籤（例如 `W_TEXTBOOK_V2`）。
- Phase 1 的逐列決策紀錄在 `outputs/STRATEGY_DECISION_LOG.csv`（V1），本文件只記錄 Phase 2。

## A. 事前登錄（看到任何 Phase 2 回測結果之前）

| 時間 (UTC) | Commit | 決策 | 依據 |
|---|---|---|---|
| 2026-10-05 16:07 | — | `MOMENTUM_LONG_BASELINE` = Phase 1 凍結 V1（hash `eba0cff4a970`），不再修改 | Phase 1 |
| 2026-10-05 17:08 | `70ba58d` | Phase 2 判決門檻：`alpha/verdicts2.py`、`outputs/phase2/PREREGISTERED_CRITERIA_PHASE2.json` | 在任何 Weinstein、intraday 結果之前 |
| 2026-10-05 17:13 | — | PDF 稽核完成：兩份 PDF 為同一本書 pp.1–136（Ch.1–4），Ch.5–10 不在 PDF | `WEINSTEIN_PDF_AUDIT.md` |
| 2026-10-05 17:31 | `940ede1` | `W_TEXTBOOK_V1`、`W_MODERNIZED_V1`、`WEINSTEIN_SHORT_V1` 全部常數凍結（`WEINSTEIN_QUANT_RULES.md` §A–§K） | 只用原書（BOOK）加上研究量化（RESEARCH），未看任何回測結果 |
| 2026-10-05 17:37 | `94ab99e` | HIGH R/R 方法事前登錄（§L）：分數只能由 DISCOVERY 十分位自動選出，RISK／PRE_TRADE_RR 不參選 | 方法事前固定 |
| 2026-10-05 17:41 | `77da3d0` | 事前登錄：Bollinger B1／B2 與變體選擇規則、1 秒引擎、Hybrids A–D、多 alpha 投組 | 同上 |
| 2026-10-05 17:40 | — | 修正（尚未看到任何結果）：MODERNIZED 停損文字的矛盾（「結構距離 > 3 ATR 不做」與「2.5 ATR 上限」並存），改為先過濾、後設上限 | 文字一致性 |
| 2026-10-05 17:36 | — | 修正（只看到 synthetic 測試資料）：HIGH R/R 分數排除 RISK family，因為停損距離與停損感知結果有機械關聯 | 方法正確性，非績效考量 |

## B. 執行紀錄

（回測完成後依序附上：Weinstein、High R/R、1 秒、出場、整合、判決）

## A2. 補登錄（2026-10-05 18:20 UTC）：依使用者原始規格（PART 37／38）定義

**背景**

- 對照使用者原始需求後發現，§M 事前登錄的 HYBRID_C、HYBRID_D、HIGH_RR_STAGE2_SETUP 和使用者規格不同。
- 目前已看過的結果：
  - 泛用事件面板的單變數十分位；
  - 原定義版本的 Hybrid C／D 與 HIGH_RR_STAGE2 結果。
- 下列「使用者規格版」**尚未計算任何結果**。
- 條件組合直接取自使用者原文，不是本研究挑選；只有門檻為 RESEARCH，並在此固定。
- 原定義版改名保留並照實報告，不刪除：
  - `HYBRID_C_ALT`（W3 leader pullback）；
  - `HYBRID_D_ALT`（momentum entry + MODERN exit）；
  - `HIGH_RR_STAGE2_SCORE_V1`（分數前三分之一 + Stage 2 + 停損 ≤ 2.5 ATR）。

### `HIGH_RR_STAGE2_SETUP`（使用者規格，PART 38）

**事件基底**

- 所有 Stage 1 基底突破：週線 stage = 1、基底 ≥ 8 週、MA30 slope4 ≥ −1%（書）。
- 觸發價 max(R, MA30) × 1.003，在下一週內被突破，第一次突破日記為 b。
- 不套用 RS、上方壓力、大盤、族群過濾。

**條件**（全部 PIT，門檻皆為 RESEARCH）

| 代號 | 條件 | 判斷時點 |
|---|---|---|
| C2 | RS Leads Price：`rs_leads60_10d` 或 `rs_leads120_10d` = 1 | 週末 t |
| C3 | Low Overhead：`OH15` < 4 | 週末 t |
| C4 | Volatility Compression：ATR5/ATR20 ≤ 0.85，或 BB 寬度 120 日百分位 ≤ 0.25 | 週末 t |
| C5 | Base Volume Contraction：Vol10/Vol60 ≤ 0.85 | 週末 t |
| C6 | Breakout Volume Expansion：b 日成交量 ≥ 2 × 前 5 日均量 | b 日收盤 |
| C7 | Tight Structural Stop：(b 收盤 − 近 10 日低 × 0.995) / ATR20 ≤ 2.0 | b 日收盤 |

**進場與出場**

- 進場：b 的次日開盤（C6 要等 b 收盤才知道，不使用前視）。
- 停損：近 10 日低 × 0.995。
- 出場：TEXTBOOK_NOVOL（Stage 3 賣半、Stage 4 全出），並另報 MODERN_NOVOL。

**報告項目**

- 全部突破、各條件有／無、符合條件數分組、7 條件全滿足。
- 對照組：同一批突破中未滿足條件者。

### `HYBRID_C`（使用者規格）

- 事件：同一批 Stage 1 突破。
- 條件：C2 + C3 + C4 + C6。
- 進場：次日開盤。
- 停損：TEXTBOOK 式，max(近 4 週週低 × 0.99, 參考價 × 0.85)。
- 出場：TEXTBOOK_INVESTOR，但不套用弱量規則（量已由 C6 確認）。

### `HYBRID_D`（使用者規格：Momentum Probe → Weinstein Confirmation → Add）

**Probe 階段**

- 凍結 V1 試單：0.25 slot、原試單價與停損，換算到 phase-2 價格尺度。

**Weinstein 確認**

- 檢查時點：試單後 20 個交易日內的週末。
- 條件：週線 stage = 2，且週收盤 > 前 6 週最高週高價。

**加碼**

- 確認後下一開盤加 0.75，合計 1.0 slot。

**加碼後管理**

- 停損 = max(試單停損, 近 4 週最低週低價 × 0.99)。
- 出場規則：Stage 4 全出、週收盤 < MA10w 出場、MFE ≥ 2R 後保本。

**未確認的處理**

- 停損，或第 20 個交易日後次日開盤出場。
