# FINAL REPORT — EMERGING LEADER PROBE STRATEGY V1

資料最新交易日：2026-10-05；Strict OOS = 2025-01-01 ～ 2026-09-03；基準成本 0.45% 來回 + 25bps 單邊滑價。

## 最終裁決

| 元件 | 裁決 | 證據 |
|---|---|---|
| DISCOVERY | **REJECT** | OOS lift20=1.18 lift30=1.27 (Discovery lift20=1.26) MAE10 lift=1.11 |
| PROBE | **WATCH** | OOS PF=1.12 EV/probe=0.0019 slots n=385 placebo p=0.488 |
| FAILED_PROBE_EXIT | **ACCEPT** | OOS avg=-2.68% worst5%=-8.49% median hold=3.0 (Discovery avg=-3.52%) |
| CONFIRMATION | **WATCH** | OOS add-leg PF=1.22 P(>=20%|conf)=11.63% vs all=3.64% add premium med=8.64% R/R med=1.00 |
| ADD | **ACCEPT** | OOS PnL with add=0.73 vs probe-only=0.18 slots; eff 0.044 vs 0.026; PF 1.12 vs 1.06; Calmar -0.03 vs -0.19 |
| FULL_SYSTEM | **REJECT** | CAGR>0:✘; Sharpe>=0.8:✘; MDD>=-25%:✔; PF>=1.3:✘; stress CAGR>0:✘; stress PF>=1.1:✘; ex-top5>0:✘; 2025>0:✘; 2026YTD>0:✘; PROBE!=REJECT:✔ |

## 研究者解讀（V1，凍結 hash `eba0cff4a970`）

### 一句話結論

「正在形成的 Leader」特徵確實帶有**小而持續**的資訊（波動控制後 leader lift 約 1.2×，2025–2026 仍在），
但把它做成「試單 → 確認 → 加碼」系統後，**在 2025–2026 Strict OOS 不成立**：
trade-level PF 1.12、10-slot 組合 −0.3%/年（同期 TAIEX +100%）、與同日同波動/同 RS 的 matched control 無統計差異（p = 0.49）。
依預先登錄標準：**FULL SYSTEM = REJECT**。

### 有效的部分

- **錯了小虧（Failed Probe Exit）成立**：OOS 失敗試單平均 −2.7%（以試單資金計），中位持有 3 日，worst 5% −8.5%；
  以總權益計每筆失敗約 −0.07%。約 76% 的試單會失敗，這和設計的 optionality payoff 一致。
- **只在對的時候加碼有用**：OOS 有加碼的總 PnL 0.73 slot vs 只試單 0.18 slot，payoff 4.8 vs 3.2。
- **個股本身的強勢比產業 β 更有資訊**：stock-vs-sector 前 20% lift 1.19×（Discovery）/1.14×（OOS），sector-vs-market 1.01×/1.10×。
- **Upside participation 有小幅資訊**（約 1.1×，兩段都在）。

### 不成立的部分

- **「大盤盤整、個股獨強」沒有穩定增量**：同樣的單日超額報酬，大盤不漲時的 leader 機率 Discovery 反而低 2.7pp、OOS 高 1.2pp；
  Discovery score 在 MARKET_UP 的 lift（OOS 1.30×）不低於 MARKET_SIDEWAYS（1.22×）。
- **Downside resilience 無效甚至反向**（Discovery 1.0×，OOS 0.92×）：抗跌股多半是低 beta / 防禦股，不是 Leader。
- **Leader capture 接近隨機**：+30% 漲幅段在漲一半之前被試單的比例只有 5%（OOS 2.7%）；
  任何時點碰到的比例約 9%，與「相同數量隨機試單」的約 7% 差距很小。
- **高度依賴少數贏家**：OOS 去掉前 5 筆後 PnL 為負；2026YTD PF 0.57。
- **對滑價敏感**：OOS PF 0 bps 1.32 → 25 bps 1.12 → 50 bps 0.86（大量 1–5 日的小試單，開盤進出）。

### 最重要的意外（只報告，不得據以修改 V1）

1. **Exit 是最關鍵的一環**：OOS 中「只用硬停損、讓贏家自己跑」（HS）PF 5.4、PnL/100 slot-days 0.43，是凍結 MP 的 10 倍；
   Discovery 期 HS 因資金效率低而落選。2025–2026 是台股極強多頭（TAIEX +100%），這個差異很大一部分是 beta，不是 alpha。
2. **真正的大贏家在試單當下已接近 250 日高點、60 日動能已強**（失敗試單距 250 日高 −7%、贏家 −2.9%）。
   「太早」本身就是 false positive 的主要來源；MARKET_DOWN 下的試單失敗率 88%。
3. **健策 3653**：7/21 就以 3,520 元進入 Emerging（PIT），但凍結的 Probe 觸發（獨立強勢日 + 停損距離 ≤ 8%）
   直到 8/31 才成立（≈5,990 元，已漲 70%）；高 ATR 讓停損距離 26 天超標、8/10–8/14 為處置股。
   9/1 試單、隔日被 RS-failure 規則出場（約 −6%），之後股價續漲到 6,900 元。規則未因健策修改。

### 下一步（必須是新版本 V2，且只能用 2026-10 之後的資料做前向驗證）

2023–2026 的資料已全部看過，任何依此修改的規則都**不能**再宣稱 OOS。若要繼續，建議凍結一個 V2 並 paper trading 3–6 個月：
- Exit 改為較寬的結構/ATR trailing（讓贏家跑），而不是 MFE 回吐一半；
- Probe 加入「60 日報酬 > 0、距 250 日高 < 10%」等過濾，避免太早；
- 停損距離改為 ATR 倍數，而不是固定 8%（高價高波動的 leader 會被系統性排除）；
- 放寬 max probes 至 3（OOS 也較好，但同樣只能前向驗證）。


## 40 個問題（數字由本次 pipeline 自動填入）

**1.** 能否提早辨認？Discovery score 前 10% 的 vol-controlled leader(+20%/40D) lift：Discovery 1.26、Strict OOS 1.18（+30%：1.27）。

**2.** Discovery 期 vol-controlled lift 最高的特徵：`hl_flag`(STRUCT) 1.26 → OOS 1.05；`rs_pct_120`(TRAD_RS) 1.23 → OOS 1.17；`progress_atr`(STRUCT) 1.22 → OOS 1.13；`rs_pct_60`(TRAD_RS) 1.22 → OOS 1.13；`resid_5`(MKT_REL) 1.22 → OOS 1.15；`cex_5`(MKT_REL) 1.21 → OOS 1.14

**3.** 大盤不漲、個股獨強（同 ATR/beta/超額報酬分層比較）：leader20 差異 Discovery -2.69%、OOS 1.18%；獨立強勢事件相對同 10D 超額分位 lift：Discovery 1.05、OOS 1.17。

**4.** Downside Resilience（RESIL family 最佳 lift）：Discovery 0.98、OOS 0.74。

**5.** Upside Participation（UPPART）：Discovery 1.14、OOS 1.22；ASYM：Discovery 1.09、OOS 1.06。

**6.** Sector Relative（SECTOR_REL）：Discovery 1.19、OOS 1.12（詳見 SECTOR_RELATIVE_STRENGTH.csv 的 stock-vs-sector × sector-vs-market 分解）。

**7.** 凍結的 Probe：score ≥ 95%、觸發 `indep`、regime `all`、停損距離 ≤ 8%；Discovery score 特徵：hl_flag, resid_5, cex_10, srs_5, is_cnt10。

**8.** P25 vs P50（Strict OOS, PnL/100 slot-days）：A(P25→1.0) 0.044、B(P50→1.0) 0.025、C 0.030；凍結選擇（Discovery 決定）：A。

**9.** 每 100 次 Probe 失敗：Discovery 78 次、OOS 76 次。

**10.** 平均 Failed Probe 損失（以試單資金計）：Discovery -3.5%、OOS -2.7%（中位 -3.3%、worst 5% -8.5%；以總權益計約 -0.067%/筆）。

**11.** Confirm 比例：Discovery 21.9%、OOS 22.3%。

**12.** Confirmation 組合比較（OOS，僅報告不再選擇）：price PnL/100sd=0.085 PF=1.25；rs PnL/100sd=0.041 PF=1.11；persist PnL/100sd=0.044 PF=1.12；price+rs PnL/100sd=0.075 PF=1.21；price+persist PnL/100sd=0.080 PF=1.23；凍結：`persist`。

**13.** Probe→Confirmation 平均 3.2 日（中位 3）；Discovery 3.3 日。

**14.** Add 後 payoff（OOS）：有加碼 4.80 vs 只試單(P25_ONLY) 3.24。

**15.** PF（OOS）：有加碼 1.12 vs 只試單 1.06。

**16.** ≥20% Winner（OOS，以投入資金報酬率計的筆數）：有加碼 14 vs 只試單 20.0；Discovery：有加碼 36 vs 只試單 44.0。加碼後平均成本墊高，報酬率門檻達標筆數不一定增加，但贏家的 PnL（slot）被放大。

**17.** ≥30% Winner（OOS，以投入資金報酬率計的筆數）：有加碼 7 vs 只試單 10.0；Discovery：有加碼 24 vs 只試單 37.0。加碼後平均成本墊高，報酬率門檻達標筆數不一定增加，但贏家的 PnL（slot）被放大。

**18.** ≥40% Winner（OOS，以投入資金報酬率計的筆數）：有加碼 5 vs 只試單 6.0；Discovery：有加碼 12 vs 只試單 20.0。加碼後平均成本墊高，報酬率門檻達標筆數不一定增加，但贏家的 PnL（slot）被放大。

**19.** Leader Capture Rate（2023–2026-09 未來 40 日漲幅段；early = 漲幅一半之前就 Probe）：+20% early 3.1%／任何時點 7.1%（7382 段）；+30% early 5.0%／任何 9.3%；+40% early 6.4%／任何 10.7%。同樣數量的隨機 Probe 碰到該股的機率約 6.7%（應與「任何時點」比較）。OOS +30% early 2.7%；10-slot 組合實際持有 +30% early 1.3%。

**20.** False Positive 主要來源（失敗 Probe vs ≥20% 贏家的標準化差異最大者）：`dist_h250` 失敗中位 -0.070 vs 贏家 -0.029 (SMD -0.60)；`ret_60` 失敗中位 0.174 vs 贏家 0.265 (SMD -0.38)；`up60` 失敗中位 -0.004 vs 贏家 -0.000 (SMD -0.37)；`dr60` 失敗中位 0.007 vs 贏家 0.003 (SMD 0.32)；`is_cnt10` 失敗中位 2.000 vs 贏家 2.000 (SMD 0.27)

**21.** 成本後（OOS）：0.70%+25bps PF 1.05、組合 CAGR -1.0%；1.00%+50bps PF 0.75、CAGR -2.9%。

**22.** 50bps 滑價（0.45% 成本）OOS：PF 0.86、組合 CAGR -1.3%、Sharpe -0.11。

**23.** 10-slot 組合（OOS）：CAGR -0.3%、MDD -9.2%、Sharpe 0.01、平均占用 10.0%；同期 TAIEX 100.8%（總報酬比較見 PORTFOLIO_SUMMARY.csv）。

**24.** PnL/100 slot-days：trade-level OOS 0.044 slot、組合 OOS -0.012%；只試單 0.026。

**25.** Top 5 依賴：OOS 總 PnL 0.73 slot，前 5 筆 = 總 PnL 的 371.4%，去掉前 5 筆 = -1.97 slot（為負 → 高度依賴少數贏家）；Discovery 去掉前 5 筆 = 3.70 slot。

**26.** 2023：Probes 353.0、PF 2.67、Payoff 12.07、失敗率 76.5%、組合報酬 2.3%（MDD -5.4%；TAIEX 26.8%）（Discovery，非 OOS）

**27.** 2024：Probes 342.0、PF 1.19、Payoff 8.96、失敗率 79.5%、組合報酬 20.3%（MDD -8.0%；TAIEX 28.5%）（Discovery，非 OOS）

**28.** 2025：Probes 286.0、PF 1.28、Payoff 5.39、失敗率 75.2%、組合報酬 -2.6%（MDD -8.5%；TAIEX 25.7%）（Strict OOS）

**29.** 2026YTD：Probes 102.0、PF 0.57、Payoff 2.66、失敗率 79.4%、組合報酬 -1.3%（MDD -9.2%；TAIEX 58.3%）（Strict OOS）

**30.** 健策是否被自然發現：是，進入 Emerging；有 Probe；Confirmation 沒有；試單失敗出場：2026-09-02 @ 5,740.61

**31.** 第一次 Discovery：2026-07-21（觀察：2026-07-20）

**32.** 第一次 Probe：2026-09-01 @ 6,090.19

**33.** Confirmation：未發生

**34.** Add：未發生；Full：未發生

**35.** 沒有抓到主升段的原因（期間 63 個交易日中各 Probe 條件不成立天數）：流動性/價格/上市天數 0 日、非處置股 6 日、Discovery 分數達門檻 50 日、觸發條件 (相對新高/獨立強勢) 59 日、大盤 regime 條件 0 日、停損距離合理 26 日。詳見 3653_CASE_STUDY.md。

**36.** 成功提早抓到的大 Winner（OOS 優先）：3231 緯創 2023-03-21 → 212.8%；3078 僑威 2023-03-15 → 93.5%；6944 兆聯實業 2024-01-16 → 77.6%；8054 安國 2023-11-28 → 77.4%；5371 中光電 2025-07-22 → 69.3%；2344 華邦電 2025-09-16 → 65.5%；4747 強生* 2023-05-31 → 61.3%；2609 陽明 2024-04-26 → 59.8%

**37.** 代表性 False Probe：2609 陽明 2024-09-30 -13.2%（跌破 Probe Low（盤中停損））；6609 瀧澤科 2024-09-03 -11.6%（跌破 Probe Low（盤中停損））；2352 佳世達 2023-07-20 -10.1%（跌破 Probe Low（盤中停損））；3038 全台 2023-07-03 -9.5%（試單後相對大盤落後 ≤ -4%（RS 失敗））；2486 一詮 2024-01-02 -4.0%（收盤跌破 MA10 且低於成本（短結構失敗））；3479 安勤 2023-08-14 -4.0%（跌破 Probe Low（盤中停損））

**38.** FinMind 限制：見 FINMIND_DATA_AUDIT.md（帳戶額度端點被環境封鎖、分K 一次一檔一天、產業別為目前快照、全市場單日資料含權證、資料非即時）。

**39.** Dashboard：`streamlit run dashboard/app.py`（見 README_DASHBOARD.md）。

**40.** 是否進 Paper Trading：FULL SYSTEM = **REJECT**、PROBE = **WATCH**。


### 安慰劑（matched controls：同日、同 ATR/beta/流動性三分位、同 RS20 五分位、優先同產業，套用相同機制）

- DISCOVERY: V1 EV/probe 0.0128 PF 1.96 vs controls 0.0022 PF 1.20；bootstrap p = 0.014
- STRICT_OOS: V1 EV/probe 0.0019 PF 1.12 vs controls 0.0017 PF 1.17；bootstrap p = 0.488
