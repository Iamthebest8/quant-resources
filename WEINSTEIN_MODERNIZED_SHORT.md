# W_MODERNIZED_SHORT（研究版放空，W_MODERNIZED_SHORT_V1）

> 全部是 **RESEARCH-DERIVED**。原書 Ch.7 不在 PDF。參數事前固定（`WEINSTEIN_QUANT_RULES.md` §J）。

## 與 TEXTBOOK_SHORT 的差異

| 面向 | TEXTBOOK_SHORT | MODERNIZED_SHORT | 動機 |
|---|---|---|---|
| 大盤 | TAIEX Stage ∈ {3, 4} | TAIEX 日收盤 < MA150，或 Stage ∈ {3, 4} | 台股 2023–2026 的大盤 Stage 3／4 時間很少，原書條件可能幾乎沒有事件 |
| 停損 | 結構停損，最多 +15% | 結構停損與 +2.5 ATR 取較近者 | 空方右尾風險（軋空） |
| 回補 | Stage 1 回補一半、Stage 2 全部回補 | 週收盤 > 10 週 MA 回補；MFE ≥ 2R 後保本；Stage 2 全部回補 | 空方獲利回吐快，需要更快的趨勢出場 |
| 可執行性 | THEORETICAL／EXECUTABLE 兩欄 | 同左 | — |

## 結果

見 `WEINSTEIN_SHORT_RESULTS.csv`（`variant = MODERNIZED`），以及 `SHORT_EXECUTABILITY_AUDIT.md`。

## 回測結果摘要

| 版本 | 期間 | n | PF | EV |
|---|---|---|---|---|
| MODERNIZED pooled | OOS（理論） | 143 | 0.66 | −1.4% |
| MODERNIZED pooled | OOS（可執行資料版） | 48 | 0.41 | −2.3% |
| MODERNIZED pooled | PRE | 194 | 1.01 | ≈ 0 |

**裁決：REJECT**。
