# EMERGING_LEADER_STATE_MACHINE — V1

```
STATE 0  DISCOVERY        觀察 (score≥80%) → Emerging (score≥90%)   無部位
   │  Probe 條件全部成立（收盤）
   ▼
STATE 1  PROBE            t+1 開盤買進 0.25 個正式部位（= 2.5% 權益）
   │                      停損 = Probe Low（近 3 日最低，盤中觸價出場）
   ├──── 失敗 ──────────► STATE 2A FAILED PROBE  快速小虧出場 (F1+F2+F3+F4, 5日; 最長 20 日)
   │
   │  Confirmation `persist` 成立（收盤，至少 2 個收盤後）
   ▼
STATE 2B CONFIRMED        停損上移至 max(Probe Low, 確認日近 3 日最低)
   │  t+1 開盤加碼
   ▼
STATE 3  ADD              架構 A: 0.25 → 1.00
   ▼
STATE 4  FULL POSITION    1.0 個正式部位（10% 權益）
   │  Exit `MP` 或硬停損
   ▼
STATE 5  EXIT             t+1 開盤出場（停損為盤中觸價）
```

## 狀態定義

| 狀態 | Dashboard 標籤 | 進入條件 | 離開條件 |
|---|---|---|---|
| 0 DISCOVERY | 觀察 / Emerging | Discovery score 百分位 ≥ 80% / ≥ 90% | 分數下降或 Probe 條件成立 |
| 1 PROBE | 可試單 → 已試單 → 等待確認 | 流動性 + 非處置 + score ≥ 95% + 觸發 `indep` + regime `all` + 停損距離 ≤ 8% | 確認 / 失敗 |
| 2A FAILED PROBE | 試單失敗 | F1 跌破 Probe Low；F2 收盤 < MA10 且 < 成本；F3 試單後相對大盤 ≤ -4%；F4 N 日未上漲；最長 20 日（依凍結組合 `F1+F2+F3+F4`） | 冷卻 5 日後可再被發現 |
| 2B CONFIRMED | 趨勢確認 / 可加碼 | price：收盤突破訊號日前 20 日高且創試單後新高；rs：RS line 40 日新高 + RS20 高於訊號日 + RS40 上升；persist：試單後 ≥60% 天數勝大盤、超額 ≥3%、勝產業 | 加碼成交 |
| 3 ADD | 可加碼 | 架構 C 第二段：確認後 ≥3 日、創確認後新高、趨勢成立、獲利 ≥3% | FULL |
| 4 FULL | 正式持股 | 部位達 1.0 | Exit 規則 |
| 5 EXIT | 出場 | HS 硬停損；TF 收盤連 2 日 < MA20；MS 收盤 < MA10 或跌破 10 日 swing low；MP 獲利曾 ≥15% 後回吐一半 | — |

## 成交假設（真實可成交）

- 訊號一律以 t 日收盤資料計算，t+1 **開盤**成交；停損以盤中觸價、跳空時以開盤價成交。
- 開盤漲停鎖死 / 開盤 ≥ 前收 +9.5% 不買；跌停鎖死延後賣出。
- 每筆成交 ≤ 當日成交值的 10%。
- 成本：來回 0.45%（買賣各半）+ 單邊 25 bps 滑價；敏感度 0.30%–1.00% × 0/25/50 bps。
