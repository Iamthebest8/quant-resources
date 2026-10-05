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
