我已經找到 `summary_record()`（[backtest.py:556-625](src/backtest.py:556)）跟 MDD/Sharpe/Sortino 的公式（[utils.py:523-554](src/utils.py:523)），逐項解釋如下：

---

## 開倉/未開倉相關分類

`self.notes_set = ["Non_Open", "Discard_NegExp", "Below_Tax", "Normal_Close", "Stop_Loss", "Exit", "Above"]`，每個 pair 在 `records_df` 裡都有一個 `note` 標籤，分類依據前面已經討論過：
- `Non_Open`：revert 機率 < `open_prob_threshold`，被機率過濾掉
- `Below_Tax`：算出的開倉門檻 < 交易成本，划不來
- `Normal_Close` / `Stop_Loss` / `Exit`（程式碼註解叫 force_close）/ `Above`：實際進場後的三種結局 + 沒被觸發開倉

---

## 逐項定義

| # | 欄位 | 定義（程式碼位置） | 你的數值意義 |
|---|---|---|---|
| 0 | **Total open count** | `len(Normal_Close) + len(Stop_Loss) + len(Force_Close)`（line 577）= 真正有開倉的配對數 | 320萬筆配對最終真的進場交易 |
| 1 | **Win per open** | `win_case / open_count`（line 581），win = `Backtest_Profit > 0` | 開倉後 72.42% 賺錢 |
| 2 | **Lose per open** | `lose_case / open_count`（line 582） | 開倉後 27.58% 賠錢 |
| 3 | **Win per record** | `win_case / 全部records_size`（line 583）⚠️分母不是 open_count，是**所有配對**（包含沒開倉的） | 全部配對裡 62.87% 最終是賺錢的（含未開倉視為非贏） |
| 4 | **Lose per record** | 同上，分母是全部 records（line 584） | 23.95% |
| 5 | **Tie per record** | `Backtest_Profit == 0` 的比例（line 573,585），未開倉的配對 `Backtest_Profit` 預設是 0，所以**這格其實等於「沒真正交易」的比例總和** | 13.18%沒有產生損益（= Non_Open+Below_Tax+Above_Top加總） |
| 6 | **Non Open per record** | 被機率過濾掉的比例（line 587） | 6.82%（DL專屬機制） |
| 7 | **Discard per record** | `Discard_NegExp` 比例（line 588），expected profit 為負被丟棄 | 0%，代表 `do_discard_expect_prof_threshold=False` 沒有啟用 |
| 8 | **Below Tax per record** | 開倉門檻小於交易成本被放棄的比例（line 589） | 5.04% |
| 9 | **Above Top per record** | 進場後 spread 從未真正越過門檻，沒觸發交易（line 590, 對應 trade.py `rec==0`） | 1.32% |
| 10 | **Normal Close per open** | `Normal_Close / open_count`（line 594）= 開倉後真正反轉回均值平倉 | 70.17% |
| 11 | **Stop Loss per open** | `Stop_Loss / open_count`（line 595） | 8.29%觸發停損 |
| 12 | **Force Close per open** | `Force_Close / open_count`（line 596）= 收盤強制平倉 | 21.54% |
| 13 | **Earn amount** | 所有贏的交易 `Backtest_Profit` **加總**（line 598） | 總共賺進 3,728,483 |
| 14 | **Loss amount** | 所有輸的交易加總（line 599） | 總共虧損 -3,343,048 |
| 15 | **Total P&L** | 全部 `Backtest_Profit` 加總（line 600）= Earn + Loss | 淨利 385,436 |
| 16 | **Profit per open** | `Total P&L / open_count`（line 603） | 平均每次開倉賺 0.12 |
| 17 | **True earn amount per win count** | 贏的交易的**平均值**（line 605, mean不是sum） | 平均贏一筆賺 1.60 |
| 18 | **True Lose amount per lose count** | 輸的交易的平均值（line 607） | 平均輸一筆賠 -3.78 |
| 19 | **Sharp ratio(daily)** | `mean(日報酬) / std(日報酬) × √252`（utils.py:546），年化 | 3.05 |
| 20 | **Sortino ratio(daily)** | 同上但分母只用負報酬的均方根（utils.py:549-554），只懲罰下跌波動 | 4.30，比Sharpe高代表上漲波動大、下跌波動相對小 |
| 21 | **MDD** | 累積損益曲線中「峰值 → 後續最低點」的**絕對金額**最大跌幅（utils.py:531-541） | 最慘時從峰值跌掉 48,283 |
| 22 | **MDD_percent** | **關鍵**：`(峰值cumprofit[j] - 谷值cumprofit[i]) / 峰值cumprofit[j]`（utils.py:524-529），是相對於**當時峰值本身**的跌幅百分比，不是相對於本金或平均資金 | 1446% |

---

## 關於 MDD_percent 為何方向會「反過來」

公式分母是**累積損益曲線當時的峰值數值**，不是固定本金或平均資金。如果 DL model 的損益曲線在某個時間點的峰值本身很小（例如剛從接近 0 或剛轉正不久），緊接著發生一筆不大的絕對虧損，分母很小、分子相對大，比例就會被放大到很誇張的數字（甚至可能超過100%，因為分子分母都是「累積損益」而非「本金」，峰值可以很接近0）。

這不是 bug，是公式設計上「相對自身峰值」的特性——但這代表 **MDD_percent 在累積損益曲線早期或低點附近會非常不穩定、容易失真**，拿來跟 baseline 比較風險程度時要謹慎，建議優先看絕對值 `MDD`（DL: -48,283 < baseline: -75,863，DL風險控制其實更好），而不是被 `MDD_percent` 的數字誤導。