# PHASE 2 FINAL REPORT：Weinstein × High R/R × 1 秒執行 × 出場 × 整合

- 資料：FinMind（2019-01 – 2026-10-02），2,065 檔普通股，PIT 面板。
- 期間：DISCOVERY 2023–2024｜STRICT OOS 2025-01-01–2026-09-03｜PRE 2020–2022（額外的樣本內歷史）。
- 成本：基準 0.45% + 每邊 25 bps；壓力情境 0.30–1.00% × 0／25／50 bps。
- 判決門檻：`alpha/verdicts2.py`，在任何 Phase 2 結果之前登錄（commit `70ba58d`）。
- 所有決策與 bug 修正：`STRATEGY_DECISION_LOG.md`。

## 一、十項最終裁決（PART 53）

| 項目 | 裁決 | 主要依據（STRICT OOS） |
|---|---|---|
| MOMENTUM LONG | **WATCH** | 凍結 V1：PF 1.12、EV 為正，但 Phase 1 FULL_SYSTEM 被 REJECT，10-slot 投組 CAGR ≈ 0.4% |
| WEINSTEIN STAGE 1→2 ENTRY | **WATCH** | TEXTBOOK：n = 9、PF 9.9、payoff 19.8、placebo p = 0.046，但 2026 年 EV 為負、n < 40；DISCOVERY 的 PF 只有 0.64 |
| WEINSTEIN CONTINUATION ENTRY | **WATCH** | TEXTBOOK：n = 34、PF 2.65、payoff 3.0、p = 0.069；2025 與 2026 年 EV 都為正；三個期間 PF 都 > 1.5，但 n < 40 |
| WEINSTEIN PULLBACK ENTRY | **REJECT** | 原書「量縮 75%」規則在台股幾乎不發生（n = 0）；MODERNIZED 版 PF 0.24 |
| HIGH R/R FILTER | **REJECT** | 分數前三分之一的 EV 8.2%，全部 21.6%；payoff 與 P(≥20%) 都較低 |
| 1-SECOND EXECUTION ENGINE | **REJECT** | Momentum：LEARNED utility 0.04% < OPEN 0.17%。Weinstein 突破個別為 WATCH（p = 0.13、n = 43） |
| WEINSTEIN SHORT | **REJECT** | 理論版 PF 0.54、可執行版 0.52–0.59，三個期間全部虧損 |
| BOLLINGER FAIL-TO-HOLD EXIT | **REJECT** | PF 3.73 vs 5.04；≥ 30% 贏家 3 筆 vs 8 筆（MDD 從 −29% 降到 −3%） |
| HYBRID STRATEGY | **WATCH** | B（Weinstein + 動能確認）與 D（動能試單 → Weinstein 確認 → 加碼）都勝過動能，但沒有勝過 Weinstein |
| MULTI-ALPHA PORTFOLIO | **WATCH** | CAGR 41.4%、Sharpe 1.17、壓力 CAGR 32.6%、2025 +18.0%、2026 +50.9%；MDD −29.9% 超過 −25% 門檻；同期 TAIEX +80% |

**沒有任何一項達到 ACCEPT。**

## 二、PART 52：47 題

1. **Stage 1→2 本身有沒有 alpha？**
   - 部分有。未過濾的 Stage 1 突破對同日隨機股票：
     - PRE：EV 14.7% vs 5.2%，p = 0.02；
     - DISCOVERY：5.4% vs 8.3%，p = 0.73，**不成立**；
     - OOS：27.7% vs 5.2%，p < 0.001。
   - 書本完整過濾版 W1 的樣本太少（OOS n = 9），只能 WATCH。
2. **Stage 2 continuation 有沒有 alpha？**
   - 有，而且是最穩定的 Weinstein 進場。PF 1.57／2.38／2.65（PRE／DISC／OOS），DISCOVERY placebo p = 0.016、OOS p = 0.069。
   - OOS 只有 34 筆，所以只能 WATCH。
3. **Pullback entry？**
   - 沒有。原書規則 0 筆；放寬版（量縮 50%）OOS PF 0.24、placebo p = 0.98。
4. **哪一種 payoff 最高？**
   - Stage 1→2。TEXTBOOK OOS payoff 19.8；未過濾突破 payoff 5.1／4.0／7.2。continuation 為 3.9／4.1／3.0。
5. **Momentum 與 Weinstein 捕捉的是不同 alpha 嗎？**
   - 是。Momentum 的進場只有 3% 與 Weinstein 重疊，Weinstein 的進場只有 8% 與 Momentum 重疊。
   - OOS 每日 PnL 相關係數：Momentum 與 W2 為 −0.28，與 W1 為 0.03。
6. **Market → Sector → Stock 有效嗎？**
   - **族群那一層有效，大盤那一層無效**。
   - 族群不在 Stage 3／4 的突破，三期 EV 都較高：PRE 15.8% vs 11.6%、DISC 5.9% vs 2.5%、OOS 30.6% vs 20.6%。
   - TAIEX 在 Stage 3／4 時的突破反而較好：OOS 38.8% vs 22.5%，三期一致。
   - 原因是 TAIEX 被台積電與 AI 權值股主導，指數橫盤時個股突破仍然有效。
7. **族群強是必要條件嗎？**
   - 不是。族群處於 Stage 2 的突破，不比其他族群的突破好（DISC 4.1% vs 7.6%、OOS 20.9% vs 32.0%；只有 PRE 為 16.2% vs 13.6%）。只要避開 Stage 3／4 族群即可。
8. **Independent leader 的右尾是否較厚？**
   - P(≥20%)：Independent 21%／19%／24%，Sector-confirmed 16%／17%／26%，非領導股 12%／11%／18%。
   - 兩種領導股都明顯較厚。Independent 在 PRE 與 DISC 最好，Sector-confirmed 在 OOS 最好，**並非一致較好**。
9. **RS 有增量嗎？**
   - 有。DISC 與 OOS 十分位單調：`rs60` +0.98／+0.88，`rs120_pct` 的 P(≥20%) 頂底差 +12.8pp。PRE 較弱。
   - Rank IC 為負，代表 RS 的效果來自右尾，不是中位數。
10. **RS Leads Price 有增量嗎？**
    - 不穩定。PRE 最好；DISC 與 OOS 都不如「RS 與價格同時創新高」。
11. **Downside Resilience 有增量嗎？**
    - 沒有。「只有抗跌」在 DISC 與 OOS 是最差的一組（`ret_s40` 1.7%／1.5%）。
12. **Upside Participation 有增量嗎？**
    - 有，是最強的特徵之一。`ucap60` 十分位單調性 +0.95／+0.99（DISC／OOS）；OOS「只有上漲參與」的組合 `ret_s40` 為 11.5%。
13. **Overhead Resistance 能改善 R/R 嗎？**
    - 能。上方無壓力（blue sky）的停損率 19–22%，其他為 50–55%；DISC 與 OOS 的報酬也較高。
    - `resdays10`（上方套牢天數）三期一致負向。
14. **Overhead Supply Density 有效嗎？**
    - 大致有效。`supply15` 的單調性：DISC −0.86、OOS −0.73、PRE +0.39，PRE 不成立。
15. **上方空間 / 停損距離能預測 payoff 嗎？**
    - **不能，而且是反向的**。`rr_h250` 在三期都是負的單調性（−0.78／−1.00／−0.98）。上方空間大的股票，通常是剛從高點跌下來、上方有套牢的股票。
16. **結構停損優於固定百分比停損嗎？**
    - 沒有證據。原書式停損（近 4 週低點，最多 15%）多數時候碰到 15% 上限，實際上接近固定百分比。
    - 較寬的停損（≥ 3–4 ATR）在台股明顯優於緊停損。
    - 1 秒結構停損（0.2–0.7 ATR）在多日持有中全部被打掉（PF 0）。
17. **停損距離 ATR 越小，R/R 越高嗎？**
    - 不是。停損越近，`ret_s40` 與「先 +20% 才停損」的比例都越低：DISCOVERY D1 為 10.7%，D10 為 32.8%。
18. **Base Duration 有用嗎？**
    - 沒有。十分位單調性在三期都接近 0。
19. **Base Tightness 有用嗎？**
    - 不穩定。DISCOVERY 越緊越好（−0.76），OOS 相反（+0.67）。
20. **Compression → Expansion 有效嗎？**
    - 沒有。單獨壓縮的組合較差；擴張（高 ATR5/20、高 BB 寬度）反而較好。「壓縮後放量」的樣本只有 46–160 筆，結果不一致。
21. **Base Volume Contraction 有效嗎？**
    - **有，而且是唯一三期一致的突破條件**。突破事件中，量縮 vs 未量縮的 EV：35.7% vs 8.3%、7.0% vs 4.9%、35.0% vs 25.2%。
22. **Breakout Volume Expansion 有效嗎？**
    - 不穩定。突破日 2 倍量：PRE 與 DISC 較差，OOS 較好。
    - 原書週量 2 倍規則讓大多數突破被判定為「弱量」並提早出場。
23. **Trend Extension 能避免追太晚嗎？**
    - 不能。越延伸平均越好；失敗的高分 setup 反而延伸度**較低**（2.7–2.9 vs 3.5–3.7 ATR）。
24. **Stage Age 有沒有最佳區間？**
    - 沒有明顯區間（單調性 −0.27／+0.58／+0.23）。失敗組的 Stage 2 年齡略老。
25. **HIGH_RR_STAGE2_SETUP 成立嗎？**
    - **無法成立**。7 個條件全部滿足的事件，7 年只有 1 筆，而且虧損。
    - 放寬為「符合 ≥ 4 個條件」：PRE n = 16、PF 9.1；DISC n = 2，兩筆都虧損；OOS n = 15、PF 14.8。DISCOVERY 不成立、樣本很小，不能確認。
26. **它的勝率？**
    - 7 條件全滿足：0／1。
    - ≥ 4 條件：50%（PRE）、0%（DISC）、33%（OOS）。
27. **Payoff？**
    - ≥ 4 條件：9.1（PRE）、不適用（DISC）、29.7（OOS）。
28. **PF？**
    - ≥ 4 條件：9.1、0、14.8。
29. **≥ 20% 贏家比例？**
    - ≥ 4 條件：43.8%、0%、33.3%。所有 Stage 1 突破：23.6%、15.5%、26.9%。
30. **≥ 30% 贏家比例？**
    - ≥ 4 條件：37.5%、0%、33.3%。所有突破：19.4%、11.7%、24.1%。
31. **≥ 40% 贏家比例？**
    - ≥ 4 條件：37.5%、0%、33.3%。所有突破：16.9%、11.4%、20.6%。
32. **失敗時平均 MAE？**
    - ≥ 4 條件：−6.5%、−13.9%、−7.2%。所有突破：−10.3%、−10.3%、−12.6%。
33. **1 秒 K 能改善進場嗎？**
    - Momentum 不能：LEARNED 0.04% vs OPEN 0.17%（TEST 380 天）。
    - Weinstein 突破略有改善：−0.86% 改善到 +0.02%，p = 0.13，n = 43，屬於 WATCH。
34. **能降低 MAE 嗎？**
    - 小幅降低。Weinstein 突破的當日 MAE 從 −2.76% 到 −2.00%；Momentum 從 −2.30% 到 −2.15%。
35. **能縮小停損距離嗎？**
    - 能，從約 4.1 ATR 縮到 0.2–0.7 ATR。但多日持有時全部被停損（PF 0、EV −2% 到 −3%），**不可用**。
36. **能提高 MFE/MAE 嗎？**
    - Weinstein 突破：0.69 → 1.01。Momentum：1.07 → 1.02，沒有改善。
37. **兩者需要不同的 intraday trigger 嗎？**
    - 需要。Momentum 的關鍵特徵是相對開盤與前收的位置，最好的進場是開盤或收復 VWAP。
    - Weinstein 的關鍵特徵是相對 VWAP 與觸發價的位置，最好的進場是突破後 7–10 分鐘。
38. **Weinstein Short 有沒有獨立 edge？**
    - 沒有。三期都虧損；OOS 可執行版 PF 0.52–0.59；S1 的訊號中約 75% 因平盤下限制而無法執行。
39. **Bollinger Fail-to-Hold 有效嗎？**
    - 作為風險控制有效：MDD 從 −29% 降到 −3%，PnL／100 slot-days 提高 58%。
    - 作為高賺賠比出場無效：PF 下降、右尾被砍。結論 REJECT。
40. **Full 還是 Partial 比較好？**
    - **Partial（50%）**。OOS PF 6.5–7.0，Full 只有 2.8–3.7；≥ 30% 贏家 8–10 筆，Full 只有 0–3 筆。
41. **Bollinger 出場會傷害大型 winner 嗎？**
    - 會。≥ +50% 的贏家從 3–11 筆降到 0–1 筆。
42. **50 bps 滑價後仍然成立嗎？**
    - W2：1.00% 成本 + 50 bps 時 OOS PF 2.26、EV 6.4%，仍成立。
    - W1：PF 9.1，仍成立。
    - 投組：壓力 CAGR 32.6%，仍成立。
43. **2025 Strict OOS 成立嗎？**
    - Weinstein 為正，但樣本極少：W2 n = 5、EV 22.5%；W1 n = 3，其中兩筆是 3189 景碩（+572%）與 4967 十銓（+136%）。
    - 投組 +18.0%，**低於 TAIEX 的 +25.7%**。
44. **2026 Strict OOS 成立嗎？**
    - W2：n = 29、PF 2.05、EV 4.9%，成立。
    - W1：n = 6、EV −3.8%，不成立。
    - 投組 +50.9%，TAIEX +43.3%。
45. **最後該保留 Momentum、Weinstein 還是 Hybrid？**
    - 主軸：**Weinstein 多方**（W2 continuation 加 Stage 1→2 突破）。
    - Momentum：只當作低相關的分散來源。
    - Hybrid D：列入 paper trading。
    - 淘汰：放空、回測進場、Bollinger 全出、1 秒引擎。
46. **哪一套最符合「高賺賠比」？**
    - **Weinstein Stage 1→2 突破搭配 Stage 3／4 趨勢出場**：payoff 4–7、≥ 40% 贏家 11–21%；原書完整版 OOS payoff 19.8。
    - 但台股的「高賺賠比」是靠**寬停損 + 長持有 + 右尾**，不是靠「停損近」。
47. **值得 paper trading 嗎？**
    - **值得，但只做 paper，不投入實際資金**。標的：
      - W2 TEXTBOOK_V1（已凍結）；
      - W1 TEXTBOOK_V1；
      - 新版本 W1_CORE_V2、HYBRID_D、BOLLINGER_EXIT_V2_HALF，必須從新的 OOS 起點重新驗證。
    - 原因：沒有一項 ACCEPT，投組 MDD 約 −30%，而且沒有勝過 TAIEX。

## 三、第十九步：30 題

1. **兩個 PDF 是否完整構成同一本書？**
   - 是同一本書，前後連續（p.61 → p.62 句子相接）。但只涵蓋全書約 40%，也就是 Ch.1–4（印刷頁 pp.1–136）。
2. **Part 1 涵蓋哪些 Chapter？**
   - 前置頁（目錄 ix–xi）、Ch.1（pp.1–30）、Ch.2（pp.31–57），以及 Ch.3 的開頭（pp.58–61）。
3. **Part 2 涵蓋哪些 Chapter？**
   - Ch.3 的其餘部分（pp.62–95）與 Ch.4（pp.96–136）。
4. **是否有缺頁？**
   - 兩份 PDF 之間沒有缺頁。
   - Ch.4 的最後兩頁（pp.137–138）不在 PDF 中。
   - Ch.5–10 與索引（pp.139–343 以後）也不在 PDF 中。
5. **是否有重疊？**
   - 沒有。
6. **哪些頁有 OCR／掃描品質問題？**
   - 150 頁都沒有文字層，OCR 完全不可用，全部改為影像判讀。
   - 偶數頁有裝訂陰影。
   - Mansfield 圖的資料框與小字模糊。
   - pp.13–14、25–26、63、65、67、103 有前一位讀者的鉛筆註記。
   - Part 2 的第 70 頁（p.131）沒有印頁碼。
   - p.87 以後的族群半色調圖解析度較低。
   - **沒有無法閱讀的頁面**。
7. **Weinstein 完整方法的核心？**
   - 用 30 週 MA 做四階段分析，只在 Stage 2 買進：Stage 1 基底突破壓力，同時 MA 走平轉上並放量。
   - 選股順序是 Market → Group → Stock，先避開上方壓力，再以 RS 確認。
   - 用 buy-stop-limit 下單。
   - 永不持有 Stage 4。
8. **前半本與後半本有哪些重要差異？**
   - 前半（Ch.1–2）：概念、讀圖、四階段定義與測驗。
   - 後半（Ch.3–4，Part 2 為主）：可操作的買進流程，包括 investor 與 trader 的進場、stop-limit 下單、Forest to the Trees、上方壓力、2 倍量、RS 與 zero line、頭肩底、雙底、Don't Commandments。
   - **大多數可以量化的規則都在 Part 2**。
9. **Weinstein Long 原始策略是什麼？**
   - Investor：在 Stage 1→2 突破時買一半，回測突破點且量縮時再買一半。
   - Trader：已確立的 Stage 2 中，價格靠近明顯上升的 MA 整理後再突破時，買整筆。
   - 出場：Stage 3 時 investor 賣一半、trader 全出；Stage 4 全部出場；量不足的突破遇反彈就賣。
   - 完整規則見 `WEINSTEIN_TEXTBOOK_LONG.md`。
10. **Weinstein Short 原始策略是什麼？**
    - Ch.7 不在 PDF。可得的片段是：
      - 跌破 Stage 3 支撐後進入 Stage 4；
      - 最好的放空點是量縮反彈回跌破點；
      - 不在上升 MA 之上放空、不放空 RS 強的股票、不在 Stage 2 族群放空；
      - 空頭市場放空弱族群中最弱的股票。
    - 停損與回補規則全部是研究補足（RESEARCH-DERIVED）。
11. **原書的 Sell Strategy 是什麼？**
    - Ch.6 不在 PDF。可得的片段是：
      - Stage 3 時 trader 出場、investor 賣一半，剩下一半的停損設在新支撐下方（pp.36–37）；
      - 永不持有 Stage 4（pp.39–40）；
      - 量不足的突破遇第一次反彈就賣，跌回突破點下方立即賣（pp.104–105、115–116）；
      - 使用 sell-stop（p.136，細節在 Ch.6，**沒有讀到**）。
12. **Stage 1→2 進場有沒有 alpha？**
    - 部分有（同第 1 題）。
13. **Stage 2 continuation 有沒有 alpha？**
    - 有，三期一致，評為 WATCH（同第 2 題）。
14. **Pullback 進場有沒有 alpha？**
    - 沒有。
15. **Stage 3→4 放空有沒有 alpha？**
    - 沒有。三期 PF 都 < 1。
16. **Relative Strength 有增量嗎？**
    - 有（DISCOVERY 與 OOS），但屬於右尾型效果。
    - 原書的 RS 過濾是 W1 漏斗中刪掉最多候選的條件（約 −78%）。
17. **Volume 有增量嗎？**
    - 基底量縮有效。
    - 突破量 2 倍不穩定。
    - 「只有放量擴張」的組合，「先 +20% 才停損」的比例最高。
18. **Sector／Group 有增量嗎？**
    - 有。避開 Stage 3／4 族群，三期一致較好。
    - 但大盤 Stage 過濾無效。
19. **Overhead Resistance 能提高 R/R 嗎？**
    - 能。上方無壓力與 `resdays10` 都有效。
20. **Structural Stop 有效嗎？**
    - 作為「定義錯誤點」有效，但不是越緊越好。寬停損較好，緊停損會摧毀右尾。
21. **RS Leads Price 有效嗎？**
    - 不穩定。
22. **Compression → Expansion 有效嗎？**
    - 無效。
23. **1 秒 K 能改善 Weinstein 進場嗎？**
    - 略有改善（WATCH），但不顯著，而且不可以用來縮小停損。
24. **1 秒 K 能改善 Weinstein 放空進場嗎？**
    - 無法驗證。TRAIN 只有 16 天（S1）與 4 天（S2），而且放空本身沒有 edge。
25. **Bollinger Fail-to-Hold 是否優於原書出場？**
    - 風險較低（MDD −3%），但 PF 與右尾較差。以高賺賠比為目標時，原書出場較好。
26. **Partial Exit 是否優於 Full Exit？**
    - 是。
27. **Momentum 與 Weinstein 是同一種 alpha 嗎？**
    - 不是。重疊不到 10%，PnL 相關係數為 −0.28 到 0.03。
28. **Hybrid 是否優於單獨策略？**
    - 優於 Momentum，但沒有優於 Weinstein。
    - Hybrid D 的投組 Calmar 1.46，略高於 Weinstein 的 1.41，但交易層 PF 較低，所以 WATCH。
29. **Strict OOS 成立嗎？**
    - Weinstein 多方（W1、W2）方向成立，但樣本太小，未達 ACCEPT。
    - 放空、回測、1 秒、Bollinger、HIGH R/R 過濾都不成立。
    - 投組為正，但 MDD 約 −30%，而且沒有勝過 TAIEX。
30. **哪一套最符合「高賺賠比」？**
    - Weinstein Stage 1→2 突破加 Stage 3／4 趨勢出場（同第 46 題）。

## 四、研究過程中的重要發現與修正（詳見 `STRATEGY_DECISION_LOG.md` §B）

1. **還原價的尺度問題**：面板的還原價是以日報酬從面板起始日向前複利重建，起始日改變，尺度就改變。FinMind 的 `TaiwanStockPriceAdj` 本身是後復權。已改用未還原價換算，Phase 1 紀錄與 Phase 2 面板可以對齊。（更正：先前誤寫為「FinMind 以查詢起日為基準前復權」）
2. **出場比較設計**：已修正為所有出場使用相同的凍結進場。
3. **1 秒 `groupby().first()` 前視**：已修正。
4. **使用者規格版的 Hybrid C、Hybrid D、HIGH_RR_STAGE2_SETUP**：在計算前補登錄。原本的定義版本保留為 `_ALT` 並照實報告。

## 五、檔案索引

**書籍**

- `WEINSTEIN_PDF_AUDIT.md`、`WEINSTEIN_BOOK_MAP.csv`、`WEINSTEIN_CHART_LIBRARY.csv`、`WEINSTEIN_CHAPTER_NOTES.md`、`WEINSTEIN_THEORY_MAP.md`

**規則**

- `WEINSTEIN_QUANT_RULES.md`、`WEINSTEIN_TEXTBOOK_LONG.md`、`WEINSTEIN_TEXTBOOK_SHORT.md`、`WEINSTEIN_MODERNIZED_LONG.md`、`WEINSTEIN_MODERNIZED_SHORT.md`

**結果**

- `WEINSTEIN_LONG_RESULTS.csv`、`WEINSTEIN_SHORT_RESULTS.csv`、`MOMENTUM_LONG_RESULTS.csv`、`WEINSTEIN_EXIT_RESEARCH.csv`

**High R/R**

- `HIGH_RR_FEATURE_RESEARCH.md`、`HIGH_RR_CANDIDATES.csv`、`HIGH_RR_STAGE2_RESULTS.csv`、`HIGH_RR_WEINSTEIN_RESULTS.csv`、`FAILED_HIGH_RR_SETUPS.csv`
- 八個 family 的分析 CSV

**1 秒**

- `INTRADAY_ENTRY_EVENT_DATASET.csv`（樣本；完整版為 `outputs/phase2/intraday_dataset.pkl`）、`INTRADAY_TRIGGER_RESEARCH.md`、`INTRADAY_TRIGGER_RESULTS.csv`

**出場**

- `BOLLINGER_FAIL_TO_HOLD_RESEARCH.md`、`BOLLINGER_EXIT_RESULTS.csv`、`RIGHT_TAIL_EXIT_ANALYSIS.csv`

**整合**

- `ALPHA_OVERLAP_ANALYSIS.csv`、`HYBRID_STRATEGY_RESULTS.csv`、`MULTI_ALPHA_PORTFOLIO.csv`

**稽核、判決、案例**

- `SHORT_EXECUTABILITY_AUDIT.md`、`STRATEGY_DECISION_LOG.md`、`PHASE2_VERDICTS.csv`
- `TRADE_EXAMPLES_PHASE2.csv`：每套策略各 10 筆成功與 10 筆失敗。

**Dashboard**

- 新增分頁：🚀 Momentum Long、📗 Weinstein Long、📕 Weinstein Short、🎯 High R/R Radar、⏱ Intraday Replay、🧾 Phase 2 判決。
