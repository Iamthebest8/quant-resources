# WEINSTEIN PDF AUDIT

Stan Weinstein《Secrets for Profiting in Bull and Bear Markets》（McGraw-Hill／Dow Jones-Irwin，© 1988；封面標示為銷售 60,000 冊後的平裝版）

本專案收到的是**同一本書拆成兩個 PDF**，研究時視為同一本書的 Part 1 + Part 2：

| 檔案 | 角色 | PDF 頁數 | 檔案大小 |
|---|---|---|---|
| `d9c610d2-Weinstein_Claude_Part1_pages_001-075.pdf` | Part 1（整份掃描第 1–75 頁） | 75 | 17.4 MB |
| `ba2b0afd-Weinstein_Claude_Part2_pages_076-150.pdf` | Part 2（整份掃描第 76–150 頁） | 75 | 18.7 MB |

兩個檔案都是**純掃描影像，沒有文字層**（pypdf 每頁抽出 0 字元），所以 OCR 抽字完全不可用。全部 150 頁都以**影像逐頁閱讀**，圖表另以 200–800 dpi 放大判讀。逐頁對照表在 `WEINSTEIN_BOOK_MAP.csv`，圖表清單在 `WEINSTEIN_CHART_LIBRARY.csv`。

## 1. 頁碼對應（以原書頁碼為準）

| 範圍 | PDF 頁 | 原書頁碼 | 換算 |
|---|---|---|---|
| Part 1 前置頁 | 1–14 | 封面、空白、扉頁（有手寫簽名，似 2006）、書名頁、版權頁、獻詞、致謝 vii–viii、目錄 ix–xi | — |
| Part 1 正文 | 15–75 | **1–61** | 原書頁 = PDF 頁 − 14 |
| Part 2 正文 | 1–75（整份第 76–150 頁） | **62–136** | 原書頁 = PDF 頁 + 61（Part 2 第 70 頁沒印頁碼，依序推為 131） |

## 2. 兩份 PDF 是否銜接、重疊或缺頁

1. **Part 1 最後一頁**：PDF 75 是原書 p.61，Chapter 3「The Ideal Time to Buy」的「The Trader's Way」一節，**在句中斷開**（講 MA 走平時延續突破缺乏力道）。
2. **Part 2 第一頁**：PDF 1 是原書 p.62，第一行接續 p.61 未完的句子，也就是 Chart 3-3「Ideal Buy for Trader」所在頁。
3. **重疊頁**：沒有。
4. **兩部分之間的缺頁**：沒有。p.61 → p.62 連續。
5. **範圍內重複章節**：沒有。
6. **掃描順序錯亂**：沒有。原書 1–136 頁全部依序出現，各讀者交叉核對過頁碼換算。
7. **OCR 失敗頁**：全部 150 頁都沒有文字層，等於「OCR 全失敗」，但每頁影像都可清楚閱讀，**沒有無法閱讀的頁面**。

## 3. 掃描品質問題（不影響主要內容）

| 位置 | 問題 |
|---|---|
| 偶數頁 | 左側裝訂陰影、輕微歪斜，沒有遮到文字 |
| Mansfield 週線圖 | 左側基本面資料框、極小的座標數字模糊。圖上價位與日期只能近似讀出，已標 [UNSURE] |
| 原書 pp.13–14, 25–26, 63, 65, 67, 103 等 | 前一位讀者留下的鉛筆勾、「NB!」、括號、資料框內手寫數字。**不是原書內容，研究時已排除** |
| Part 2 PDF 70（p.131） | 沒有印頁碼 |
| Part 1 PDF 3 | 手寫簽名無法辨識 |
| 半色調群組圖（p.87 起） | 解析度較低，小標籤難讀 |

## 4. 章節覆蓋（**重要**）

依 Part 1 的目錄（pp. ix–xi），全書共 10 章加索引（索引始於 p.343）：

| Ch. | 標題 | 原書頁 | 本專案 PDF 是否包含 |
|---|---|---|---|
| 1 | It All Starts Here! | 1–30 | ✅ 完整（Part 1） |
| 2 | One Glance Is Worth a Thousand Earnings Forecasts | 31–57 | ✅ 完整（Part 1） |
| 3 | The Ideal Time to Buy | 58–95 | ✅ 完整（Part 1 pp.58–61 + Part 2 pp.62–95） |
| 4 | Refining the Buying Process | 96–138 | ⚠️ **只到 p.136**。缺 pp.137–138（「Don't Put All Your Eggs in One Basket」的結尾與 Chart 4-35） |
| 5 | Uncovering Exceptional Winners（Triple Confirmation Pattern） | 139–163 | ❌ **不在 PDF** |
| 6 | When to Sell（Sell-Stop、Investor/Trader 賣法、Measuring the Move） | 164–214 | ❌ **不在 PDF** |
| 7 | Selling Short（When to Sell Short、Protecting Your Short with a Buy-Stop …） | 215–267 | ❌ **不在 PDF** |
| 8 | Using the Best Long-Term Indicators to Spot Bull and Bear Markets（A-D line、momentum …） | 268–309 | ❌ **不在 PDF** |
| 9 | Odds, Ends, and Profits（基金、選擇權、期貨） | 310–335 | ❌ **不在 PDF** |
| 10 | Putting It All Together | 336–342 | ❌ **不在 PDF** |
| — | Index | 343– | ❌ |

**結論：兩份 PDF 合起來是同一本書的前 136 頁（約 40%），中間沒有缺口，也沒有重疊，但不是整本書。** 第 5–10 章只能從目錄得知章名與小節名，內容沒有提供。

## 5. 對研究的影響

- **做多（Ch.1–4）有完整原書依據**：Stage 1–4 定義、30 週 MA、突破／跌破、回測、Buy-Stop-Limit 下單、Investor／Trader 進場法、Forest-to-the-Trees（Market → Group → Stock）、Overhead Resistance、Volume 2× 規則、RS 與 zero line、Don't Commandments、頭肩底、雙底、Bigger Base。
- **賣出紀律（Ch.6）沒有提供**。原書可用的出場資訊只有零星段落：p.36–37 Stage 3 先賣一半、剩下一半把停損設在新支撐下緣；p.39–40 永不持有 Stage 4；p.104、115–116 量不足的突破遇反彈就賣；p.136 分散持股與 sell-stop 的概念。因此「TEXTBOOK EXIT」只能用這些段落加上**標明 RESEARCH-DERIVED 的補充**。
- **放空（Ch.7）沒有提供**。原書可用的放空資訊只散在 Ch.1–4：p.14–15 跌破後量縮反彈至跌破點是理想放空點；p.19 放空定義；p.26–27、109–110 MA 上升或 RS 上升時不放空、RS 跌破 0 是重要負面；p.38 註 1 偏好大量跌破；p.80 不在 Stage 2 族群放空；p.88 空頭市場放空弱勢族群中最弱的股票。`W_TEXTBOOK_SHORT` 是把這些段落忠實量化，**停損、回補、目標價全部標 RESEARCH-DERIVED**。
- **大盤指標（Ch.8）沒有提供**。原書可用的只有 p.75–77 的「大盤趨勢優先」，以及「NYSE 處於 Stage 1+2 的股票比例」這一個指標（p.76 註 11）。
- 「研究時不要假裝讀過不存在的內容」：本專案所有標為 BOOK-DERIVED 的規則都附原書頁碼，而且都落在 pp.1–136。

## 6. 讀取流程紀錄

1. 6 位讀者分段平行閱讀，每段 25 頁：Part 1 的 1–25、26–50、51–75，Part 2 的 1–25、26–50、51–75。全部都是**影像閱讀**，圖表放大判讀。
2. 兩段（Part 1 的 51–75、Part 2 的 51–75）第一次被內容過濾器中止，原因是要求逐字引文。之後改為**轉述（paraphrase）**，引文限 12 字以內，**重新完整閱讀**。
3. **兩部分都讀完並完成 Book Map 之後**，才開始建立 Theory Map 與策略。順序是 Part 1 → Part 2 → Audit → Theory Map → Textbook → Quantification。
4. 逐頁筆記保存在研究過程的工作檔；本 repo 收錄整理後的 `WEINSTEIN_CHAPTER_NOTES.md`、`WEINSTEIN_BOOK_MAP.csv`、`WEINSTEIN_CHART_LIBRARY.csv`。
