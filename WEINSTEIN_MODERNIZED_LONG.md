# W_MODERNIZED_LONG（研究版做多，W_MODERNIZED_V1）

> 全部修改都是 **RESEARCH-DERIVED**，不是 Weinstein 原書規則。
> 參數在看到任何 Weinstein 回測結果之前固定（`WEINSTEIN_QUANT_RULES.md` §J），**不做網格搜尋**。
> 目的是檢驗「把原書框架加上現代風險幾何」能否改善 R/R，同時保留 TEXTBOOK 作為對照。

## 與 TEXTBOOK 的差異

| 面向 | TEXTBOOK | MODERNIZED | 動機 |
|---|---|---|---|
| 進場事件 | W1／W2／W3 | 同一批事件 | 可直接比較 |
| RS | Mansfield 型（自身 RS 趨勢） | 再加上 120 日超額報酬橫斷面百分位 ≥ 70 | 領導股文獻；phase 1 發現上漲參與度有幫助 |
| 上方壓力 | OH15 週數 < 4 | 成交量價位分佈 `supply15 ≤ 0.15` | 以「套牢量」取代「週數」，較貼近供給概念 |
| 停損 | 近 4 週低點，最多 15% | 近 10 日低點，最多 2.5 ATR（W3 為 2.0 ATR）；停損距離 > 3 ATR 的 setup 不做 | High R/R：先控制結構性風險 |
| 大盤 | TAIEX Stage ∈ {1, 2} | 或 breadth12 ≥ 50% | 原書 p.76 認為 breadth 是關鍵指標，門檻屬研究假設 |
| 量能確認 | 週量 2 倍（成交後檢查） | 日量 2 倍（p.104 註5 的日線版） | 更早判斷弱量突破 |
| 出場 | Stage 3 賣一半、Stage 4 全出 | 週收盤 < 10 週 MA 全出；MFE ≥ 2R 後保本；Stage 4 全出 | 原書指出 10 週 MA 給 trader 用（p.13），用在出場屬研究 |

## Bollinger 出場

Bollinger Fail-to-Hold 是另一組出場研究，見 `BOLLINGER_FAIL_TO_HOLD_RESEARCH.md`。測試時使用同一批凍結的進場，與 TEXTBOOK、MODERNIZED 出場並列比較。

## 結果

見 `WEINSTEIN_LONG_RESULTS.csv`（`variant = MODERNIZED`），以及本文件末段。

## 回測結果摘要

| 引擎 | DISC n／PF | OOS n／PF |
|---|---|---|
| W1 MODERNIZED | 0 | 1（無法評估） |
| W2 MODERNIZED | 39／1.34 | 4／3.86 |
| W3 MODERNIZED | 63／1.01 | 42／**0.24** |

**結論**：現代化版本**比原書版差**。

- 「領導股 + 緊 ATR 停損」幾乎排除了所有 Stage 1 基底股；緊停損也把右尾砍掉。
- 在台股，原書的寬停損加 Stage 出場較好。
- MA10w 出場（MODERN）在相同進場上：OOS PF 2.57，TEXTBOOK 為 5.04（`WEINSTEIN_EXIT_RESEARCH.csv`）。
