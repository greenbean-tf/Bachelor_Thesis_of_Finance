# DL Model 落後 Baseline (1.5σ) 診斷決策樹

背景：美股 backtest 結果中，DL model 的 Total P&L (385,435) 低於 baseline 固定 1.5σ
門檻 (577,465)，與論文中台股市場 GGWP 勝過 baseline 的結論相反。本檔案記錄診斷
此問題的步驟與決策邏輯。

## ✅ 步驟 8：checkpoint 選擇 bug（已發現、已修正、已驗證，效益顯著）

**發現**：直接讀取目前用於所有 backtest 診斷的 checkpoint
（`Train_ckpt/2026-04-24 14-42-37/Epoch=75_ValLoss=9.5018.pth`）內部存的 `history`，
比對同資料夾的其他 checkpoint：

```
目前實際使用：Epoch=75_ValLoss=9.5018.pth   (Val Loss = 9.50，訓練後期已劣化)
資料夾內最佳：BestEpoch=24_ValLoss=5.9219.pth (Val Loss = 5.92，比目前用的低 38%)
```

訓練從 epoch 24 之後再也沒有改善過，一路震盪到 epoch 75 觸發 early stop
（`min_val_count=49`，接近 `early_stop_count=50` 閾值），但 backtest 用的
`model_path` 卻是 epoch 75，不是驗證集表現最好的 epoch 24。

**根本原因**：[main.py:134-143](src/main.py:134) 的 `inference_only` 分支有 bug：

```python
if model_path:
    PT_model.load_model(model_path)      # 載入 Epoch=75（find_latest_checkpoint 找到的最後一個檔案）
elif PT_model.best_model_name...:        # ← if 已經真，這條路徑永遠不會執行
    ...
# 沒有第二次 reload 就直接拿去 inference
```

對照 `mode=="test"`（[main.py:217-239](src/main.py:217)）的正確流程，多了一步
「先載入 model_path、再用 best_model_name 覆蓋回去」：

```python
if model_path:
    PT_model.load_model(model_path)
elif mode == "test":
    if PT_model.best_model_name...:
        PT_model.load_model(PT_model.best_model_name)   # 關鍵：重新載入最佳checkpoint
```

`inference_only` 少了這一步，導致它永遠停在 `find_latest_checkpoint()`
（[hyperparameters.py:117-125](src/hyperparameters.py:117)）依修改時間找到的**最後一個檔案**——
對任何 early-stop 的訓練而言，這必然是「最後一個 epoch」，不是「驗證集最佳 epoch」。

**驗證**：直接用 Python 執行 `find_latest_checkpoint()` 的邏輯，今天確實回傳
`Epoch=75_ValLoss=9.5018.pth`，跟所有已知 `hyperparameters.json` 記錄的 `model_path`
完全吻合——證實這是每次跑 `inference_only` 都會踩到的系統性問題，不是單次意外。

**影響**：**步驟1、2、3、3.5、6、7 的所有診斷，都是基於這個劣化的 epoch-75 checkpoint
跑出來的 inference_cache**。真正收斂良好的 epoch-24 checkpoint（val loss 低 38%）
從未被實際用於任何一次 backtest。前面觀察到的「T_o 系統性偏小」「revert 機率
輸出集中在高信心區間」等現象，有可能不是模型架構或資料的根本缺陷，而是這個
checkpoint 選擇錯誤的下游症狀。

**已修正**：[main.py:134-150](src/main.py:134) 補上跟 `mode=="test"` 對稱的
第二次 reload 邏輯，載入 `model_path` 後會自動偵測並改用 `best_model_name`。

**驗證結果（2026-07-03，`data/Record/2026-07-03 07-10-12/`）**：GPU log 確認
`[Inference] 改用驗證集最佳 checkpoint：...BestEpoch=24_ValLoss=5.9219.pth`，
重跑 `backtest_from_saved`（設定跟原始診斷一致：`must_open=False`,
`do_discard_expect_prof_threshold=False`, `open_prob_threshold=0.65`）：

| 指標 | 舊 checkpoint (Epoch=75, val loss 9.50) | 新 checkpoint (BestEpoch=24, val loss 5.92) | 變化 |
|---|---|---|---|
| Total open count | 3,207,980 | 3,413,272 | +6.4% |
| **Total P&L** | **385,436** | **438,571** | **+53,135 (+13.8%)** |
| Profit per open | 0.1201 | 0.1285 | +7.0% |
| True earn per win | 1.605 | 1.643 | +2.4% |
| Sharpe | 3.050 | 3.105 | 微升 |
| Mean Opened Threshold | ~1.60 | 1.431 | 更接近 baseline 1.5 |
| Below Tax per record | 5.04% | **0.20%** | 大幅下降 |

跟 Baseline（577,465）的差距：從 -192,030 (-33%) 縮小到 **-138,895 (-24%)**，
差距縮小了 28%。**checkpoint 選取確實是實質 bug，不是無關緊要的小事**——單純換成
驗證集最佳權重，Total P&L 就進步了近 14%，`Below Tax per record` 從 5.04%
驟降到 0.20% 更是模型校準明顯變好的直接證據。但差距仍未補平，代表 checkpoint
只是問題的一部分，不是全部。

### 步驟 8.0：用修正後 checkpoint 重新驗證步驟 1+2（regime split + T_o 分布）

用 [analysis/diagnose_regime_split.py](analysis/diagnose_regime_split.py)（已改指向
`2026-07-03 07-10-12`）重新產生分析，結果存於
[analysis/output_fixed_ckpt/](analysis/output_fixed_ckpt/)。

**Regime split（步驟1）**：

| | DL Total P&L | Baseline | 差距 | Profit/open 差距 |
|---|---|---|---|---|
| 全期 | 438,571 | 577,465 | -138,895 | — |
| 前段（升息劇烈期） | 254,878 | 317,574 | -62,695 | -0.0242 |
| 後段（相對穩定期） | 183,692 | 259,891 | -76,199 | **-0.0378** |

**後段差距依然比前段更大**（-76,199 vs -62,695；Profit/open 差距 -0.0378 vs
-0.0242），跟修正前的模式完全一致——**regime shift 假設在修正 checkpoint 後依然
被否證**，不是這次改善的原因，兩次獨立驗證得到相同結論，確定穩健。

**T_o 分布（步驟2）**：

```
median T_o : 1.041 → 1.157   （往 1.5 靠近，但仍明顯偏低）
mean T_o   : 1.401 → 1.431
比例 < 1.5 : 66.23% → 63.93%（略微改善）
```

**T_o 系統性偏小的現象依然存在，只是程度略微減輕**——checkpoint 修正讓校準變好了
一些，但沒有根除這個偏差。

### 步驟 8.1：用修正後 checkpoint 重新驗證步驟 3（機率篩選 must_open=True）

設定：`mode="backtest_from_saved"`, `must_open=True`（沿用同一份修正後 checkpoint
的 inference_cache，不需重跑 GPU；[main.py](src/main.py) 已改成不自動刪除 cache）。

| 指標 | 修正後基準 (0.65) | must_open=True | 變化 |
|---|---|---|---|
| Total open count | 3,413,272 | 3,614,207 | +200,935 |
| **Total P&L** | **438,571** | **455,726** | **+17,155 (+3.9%)** |
| Profit per open | 0.1285 | 0.1261 | 微降 |
| Sharpe | 3.105 | 3.074 | 微降 |
| MDD | -54,876 | -61,188 | 惡化 |

**跟舊 checkpoint 的步驟3結果幾乎一模一樣**（+16,403 / +4.3% vs +17,155 / +3.9%）。
**結論：這個機制（revert 機率篩選 / Predictor1 校準）的行為模式跟 checkpoint 品質
無關，是持續存在、不受 checkpoint 影響的獨立問題**——不是 checkpoint bug 造成的
假象，是模型架構/訓練本身的固有特性。

### 步驟 8.2：用修正後 checkpoint 重新驗證步驟 3.5（Expected Profit 過濾）

設定：`mode="backtest_from_saved"`, `do_discard_expect_prof_threshold=True`,
`expect_prof_threshold=0`（沿用同一份 inference_cache）。

| 指標 | 修正後基準 | 過濾負期望值 | 變化 |
|---|---|---|---|
| Total open count | 3,413,272 | 3,185,613 | -227,659 (-6.7%) |
| **Total P&L** | **438,571** | **435,708** | **-2,863 (-0.65%)** |
| Profit per open | 0.1285 | 0.1368 | **+6.5%** |
| True earn per win | 1.6427 | 1.7251 | +5.0% |
| Sharpe | 3.105 | 3.204 | +3.2% |
| Sortino | 4.338 | 4.497 | +3.7% |
| MDD | -54,876 | -54,787 | 微幅改善 |

**這是跟舊 checkpoint 結果差異最大的一項**：

| | 舊 checkpoint | 新 checkpoint |
|---|---|---|
| 過濾後 Total P&L 變化 | -31,344 (-8.1%) | -2,863 (-0.65%) |
| 傷害程度 | 明顯有害 | 幾乎中性、品質全面提升 |

**傷害程度從 -8.1% 縮小到 -0.65%，縮小了 91%。** 用舊 checkpoint 時，模型算出的
`expected_return` 嚴重失真——它判斷「會虧」的配對，實際平均還能賺 +0.067，過濾
掉等於白白丟掉真實利潤。換成正確 checkpoint 後，模型的判斷幾乎準了——它認為
「會虧」的配對，現實中也幾乎打平（+2,863/227,659 ≈ 每筆只賺 0.0126，接近雜訊
等級）。**結論：步驟3.5當初「過濾機制方向對但誤殺」的結論，主要是被錯誤
checkpoint 放大出來的假象，不是模型架構或訓練資料的根本缺陷**——換掉 checkpoint
後這個問題基本消失，且過濾機制轉為對風險調整後報酬（Sharpe/Sortino/Profit per
open）明顯有益。

### 步驟 8 總結：「模型校準」問題被拆解成兩個獨立來源

| 校準問題 | 對應診斷 | 是否受 checkpoint bug 影響 |
|---|---|---|
| `expected_return`（μ_rtop/β_rtop等分布參數） | 步驟3.5、步驟7 | **主要是** checkpoint bug 造成，已大幅改善 |
| `revert_prob`（Predictor1 機率輸出壓縮） | 步驟3、附帶發現 | **無關**，checkpoint 修正前後效果幾乎相同 |
| T_o 系統性偏小 | 步驟2 | **部分** 受影響，改善但未消除（median 1.04→1.16，仍低於1.5） |
| Regime shift | 步驟1 | **無關**，兩次驗證都否證 |

## 步驟 9：RTop/Top/Close 三個分布的離散度比例（修正原本只看 RTop 的疏漏）

**背景**：步驟7只檢查了 RTop 的 β/μ ≈ 0.53，並推論這是 T_o 偏小的成因。但這個推論
**不完整**——`model_output_summery.csv` 甚至沒有輸出 `Mean Close Sigma` 這個欄位
（[backtest.py:495-543](src/backtest.py:495) 只對 Top、RTop 輸出了 `_sigma` 統計，
Close 的 σ 沒被記錄到摘要裡，只存在原始 `record_*.csv` 的 `close_sigma` 欄位）。

**用修正後 checkpoint 的完整資料重新計算三者的比例**（[analysis/output_fixed_ckpt](analysis/output_fixed_ckpt)
同批 record 檔案，n=3,694,957）：

| 變數 | 分布類型 | mean(μ) | mean(β/σ) | 比例 (β或σ / μ) |
|---|---|---|---|---|
| RTop | Gumbel | 2.825 | 1.516 | 0.537 |
| Top | Gumbel | 6.423 | 3.087 | 0.481 |
| Close | Normal | 5.047 | 3.565 | **0.706** |

**結論（⚠️ 已被步驟12的校準檢驗部分推翻，見下方）**：不是只有 RTop 偏大，三個變數
的離散度比例都偏高（0.48~0.71），Close 甚至是三者中最高的。原本「RTop 分布系統性
偏小」的講法不夠精確，應修正為：模型對 rtop/top/close 三個變數的預測分布，相對於
各自平均值而言，離散度普遍偏高——這代表模型對「這個配對的價差行為到底有多確定」
整體預測得不夠精準，置信區間系統性偏寬，不是單一變數的個別問題。

**重要提醒**：這裡的「σ/μ 比值偏高」只是描述性統計，**不能直接等同於「σ 校準過頭」**
（過度自信地放寬不確定性）。步驟12用真實標籤直接做校準檢驗，發現方向可能相反——
詳見步驟12。

## 步驟 10：Loss Curve 檢查（已完成，重大發現）

**背景**：checkpoint bug 修正後，`history`（每個 epoch 的 train/val/test loss）
已確認是直接存在 checkpoint 檔案裡（[model.py:375](src/model.py:375)），不需要
GPU，本機 `torch.load()` 即可讀取繪圖。用戶提供了 v0→v1 清洗一輪後訓練出的
loss curve（對應 `BestEpoch=24` 所在的那次訓練，epoch 0~75）。

**觀察**：

1. **Train loss 提早凍結**：前 10 個 epoch 從 ~4.9 快速降到 ~4.2，之後整整
   65 個 epoch 幾乎完全平坦（4.1~4.3），只在 epoch 55、66 附近有極小抖動又
   馬上恢復——不是典型「train持續降、val持續升」的 overfitting 模式。
2. **Val/Test loss 劇烈且不收斂的震盪**：初期降到 6~7 附近後，反覆出現尖峰
   （epoch ~6、~20、~32、~48-58），且震盪幅度**沒有隨訓練時間縮小**——訓練
   尾聲反而爆出最大一次尖峰（`Epoch=74_ValLoss=13.3557`，對應圖上 epoch~74
   衝到 13+）。
3. **Train 幾乎不動、val/test 卻大幅擺盪**，代表模型權重變化很小，卻能讓
   val/test loss 產生巨幅震盪——最合理的解釋是**驗證/測試資料裡仍有少數
   ill-conditioned 樣本，模型每次隨機碰到這些樣本時，NLL loss 被少數幾筆
   極端樣本的懲罰項拉爆**。

**與既有發現的串連**：

- **直接呼應步驟5**：這張圖是「v0→v1清洗一輪後」訓練出來的結果，卻依然有
  劇烈的晚期 loss 尖峰，跟步驟5發現「v0→v1清洗後 ill-conditioned 樣本不減
  反增（292MB>239MB）」完全吻合——**代表資料清洗確實還沒收斂，殘留的問題
  樣本持續在訓練過程中製造不穩定**。
- **可能解釋步驟9的高離散度發現**：當 NLL loss 對少數極端樣本異常敏感時，
  模型常見的「自保策略」是**故意把預測分布變寬（提高 σ/β）**——分布越寬，
  同樣的預測誤差對 NLL 的懲罰就越小。這可能是 RTop/Top/Close 三者 β/σ/μ
  比例都偏高的成因：不是模型「學不會」精準預測，而是在訓練過程中學到
  「預測寬一點比較安全」這個次優解，用來對抗資料裡的雜訊/極端值。
- **也解釋了 checkpoint 選擇的脆弱性**：epoch 24 被選為 best，很可能只是
  恰好落在某次震盪的低谷，不代表真正收斂到全局最優解——如果換一個隨機
  種子重新訓練，"best epoch" 的表現可能有很大差異，代表目前的 early
  stopping 機制在這種高波動的 loss 曲面下，選出來的 checkpoint 本身不夠穩健。

**結論**：這張圖是目前為止最直接指向「資料清洗不足」是根本成因的證據——
不是單純的模型架構或 loss 權重設計問題，而是訓練/驗證資料裡仍殘留會造成
數值不穩定的極端樣本，這些樣本同時可能是 Predictor1 過度自信、以及
RTop/Top/Close 離散度偏高的共同上游成因。**下一步應該優先處理步驟5（資料
清洗 v1→v2），而不是急著調整 loss function 權重或重新設計 hyperparameter**
——如果不先把資料清乾淨，即使調整 loss 設計，仍可能受同樣的極端樣本干擾。

## 步驟 11：loss.py 的 eps 保護 bug + 新增 sigma 崩塌監控（已發現、已修正）

**背景**：釐清歷史動機後確認——資料清洗機制（`cond_threshold` 相關係數矩陣條件數
過濾）當初的動機，是先前觀察到訓練時 loss 有時會跑到異常負值，**懷疑這是造成
DL model 打不贏 baseline 的可能原因之一**（不是訓練crash，是懷疑污染了梯度/
學習品質）。這代表這個問題不是程式碼細節，而是直接連結到整個診斷主線。

**發現：連續分布的 NLL loss 理論上沒有下界。** 用 Normal 分布為例：

```
NLL(x; μ, σ) = log(σ) + 常數 + e²/(2σ²)      (e = x - μ，固定的預測誤差)
```

對 σ 微分求最小值：`σ = |e|`——**當模型某筆樣本剛好預測得很準（e很小）又同時把
σ壓得極小，loss會衝向負無限大**。這是連續分布NLL的已知結構性缺陷（跟variance
inflation是同一機制的相反方向：一個是σ被推得太寬，一個是σ在少數樣本上被壓得
太窄）。

**具體 bug**：檢查 [loss.py](src/loss.py) 發現 Gumbel 項（`Rtop_Gumbel_loss`、
`Top_Gumbel_loss`）的 `log()` 有套用 `torch.max(sigma, eps)` 保護下界，但**所有
Normal 分布項（`Close_Normal_loss`，以及 `NormIndLoss`/`GaussCopNormLoss` 裡的
`Rtop_Normal_loss`/`Top_Normal_loss`）的 `log(σ²)` 都沒有這個保護**，直接用原始
未clamp的 sigma。由於目前使用的 `GaussCopGumLoss`（rtop/top用Gumbel，close用
Normal）裡，**`Close_Normal_loss` 就是整個 loss function 裡唯一沒有下界保護的
一項**，只要 `sigma_close` 被壓得夠小（如1e-10），`log(σ²)` 可以到-46甚至更負，
遠比其他有保護的項（理論最壞約-9）誇張，很可能就是「loss有時候跑很負」的
具體來源。

**現有清洗機制的盲區**：`cond_threshold`（相關係數矩陣條件數）保護的是矩陣求逆
（`torch.linalg.inv(R)`）的數值穩定性，這個過濾理由本身是對的。但**它只監控
R矩陣這一條路徑，完全沒有監控marginal分布（σ_rtop/σ_top/σ_close）本身崩塌到
接近零的情況**——一個樣本可能R完全正常（通過過濾），但σ_close被壓得極小，
一樣會透過上面的bug觸發巨大負值loss，且完全不會被記錄進
`ill_conditioned_data_v*.csv`，也就不會被之後的清洗流程剔除。這可能直接解釋了
「v0→v1清洗後樣本不減反增」的現象——清洗的偵測範圍本身就不完整，不管清幾輪，
沒被偵測到的那條路徑永遠清不掉。

**已完成的修正**：

1. **修正eps bug**（[loss.py](src/loss.py)）：全部4個loss class裡，Normal
   分布項的 `log(σ²)` 都改成 `log(square(max(sigma, eps)))`，跟Gumbel項的
   保護方式一致。
2. **新增sigma崩塌監控**（[loss.py](src/loss.py) `GaussCopGumLoss`）：仿照
   R矩陣條件數的監控機制，額外檢查 `sigma_rtop`/`sigma_top`/`sigma_close`
   是否低於 `sigma_collapse_threshold=1e-2`（正常尺度約1.5~3.5，遠高於此
   門檻），記錄到新的 `sigma_collapse_data_v{n}.csv`，並同樣從該batch的
   loss中剔除（跟R矩陣異常的處理方式一致：`bad_mask = cond_bad_mask |
   sigma_bad_mask`）。
3. **`clean_data.py` 同時讀取兩個log**（[clean_data.py](data_cleaning/clean_data.py)）：
   合併 `ill_conditioned_data_v{n}.csv` 跟 `sigma_collapse_data_v{n}.csv`
   兩個來源的 `Data_ID`，一起從訓練資料中剔除。
4. **加入訓練資料版本覆寫機制**（[hyperparameters.py](src/hyperparameters.py)、
   [main.py](src/main.py)）：新增 `GGWP_DATA_VERSION` 環境變數，可強制訓練
   使用指定版本的 `cleaned_data_v{n}.pickle`，不受「自動抓最新版本」邏輯
   影響——這次要重新從最原始的 `cleaned_data_v0.pickle` 開始清洗，套用
   修正後的完整偵測機制，而不是接續在v1的基礎上繼续（因為v0→v1這一輪
   清洗，本身就是用有bug、偵測不完整的loss跑出來的，可能沒清乾淨）。

**驗證計畫（如何知道這次修正有沒有幫助）**：

1. 新舊 loss curve 直接疊圖比較——尖峰（如原本epoch74衝到13+）是否消失或縮小
2. `sigma_collapse_data_v0.csv` 累積的樣本數量——大量樣本代表這個問題確實
   存在且頻繁，log是空的代表這條路徑影響有限
3. 清洗趨勢是否轉向下降（v0→v1→v2的異常樣本數，這次是否不再逆勢增加）
4. Backtest的Total P&L、以及兩個「跟checkpoint無關、持續存在」的指標
   （Predictor1 specificity、T_o median）這次是否終於出現改善——如果這兩個
   指標這次真的動了，才能確定問題根源真的追到這裡

單一指標不夠，需要這幾項同時成立，才能算是有力證據支持「eps bug + 監控盲區」
是問題真正成因之一。

## 步驟 12：σ 校準檢驗——「σ/μ 比值偏高」不等於「σ 校準過頭」（重大修正）

**背景**：步驟9、步驟1的簡報都用 σ/μ（或β/μ）比值來論證「模型預測分布過寬」，
並據此推導 variance inflation 機制。但這個推論被質疑：σ/μ 比值只是跟其他變數
互相比較的相對大小，從未跟任何「校準基準」比較過，不能直接說「σ偏大」。

**驗證方式**：用 `record_*.csv` 裡的真實標籤（`Norm_Rtop`/`Norm_Top`/`Norm_Close`）
對照模型預測的 μ、σ，直接算校準用的 z-score：

```
z = (真實值 - 模型預測μ) / 模型預測σ
```

若模型校準得當，z 應該服從標準分布：RTop/Top（標準Gumbel(0,1)）理論變異數
= π²/6 ≈ 1.6449，理論std ≈ 1.2825；Close（標準常態）理論std = 1.0。

**結果**（n=3,694,957，RTop濾掉Norm_Rtop=-1的sentinel後n=2,999,476）：

| 變數 | z 的 std（實際） | 理論值（校準得當時） | 倍數 |
|---|---|---|---|
| RTop | 2.98 | ≈1.28 | 2.3倍 |
| Top | 2.51 | ≈1.28 | 2.0倍 |
| Close | 1.37 | 1.00 | 1.37倍 |

**結論：方向跟「σ過寬」完全相反。** 三個變數的 z 標準差**全部遠大於**校準得當時
應有的理論值——這代表**真實誤差比模型預測的σ暗示的還要大，模型其實低估了自己
的不確定性，不是高估**。如果σ真的過寬（過度自信地放寬不確定性），應該會看到z
被壓縮（std(z) < 理論值），但實際觀察到的是相反方向。

**這推翻了步驟9、步驟1簡報裡「σ/μ比值高 → variance inflation → 模型學會放寬σ
規避NLL懲罰」這條推論鏈的第一環**——σ/μ比值偏高是真實的描述性統計沒錯，但不能
被解讀成「校準過頭」，實際校準檢驗顯示可能是相反方向（校準不足、σ跟不上真實
誤差波動）。

**後續影響**：
- 步驟9、簡報投影片1/2/3/4裡「σ過寬」的敘事需要修正或下修其確定性
- Loss curve 不穩定（步驟10）跟 eps bug（步驟11）的發現本身**不受影響**，
  仍然是獨立驗證成立的問題（那些是直接從訓練過程/程式碼邏輯驗證的，不依賴
  「σ過寬」這個描述性統計的解讀）
- 但用「variance inflation machanism」去解釋「為什麼σ/μ比值偏高」這個串聯邏輯
  需要重新檢視，不宜再直接沿用
- 真正該問的問題可能不是「σ為什麼被放大」，而是「為什麼模型的μ、σ整體校準
  都不夠準」——這可能回到訓練收斂不足（loss curve證據）或資料清洗不足（步驟5）
  這些更根本的成因，而不是「模型刻意放寬σ規避懲罰」這個特定機制

---

## 步驟 1：分段測試（regime shift 假設）— 已驗證並否證

**做法**：不需要重跑 Colab。`backtest_from_saved`/baseline 每次執行都已經把逐日
逐配對交易記錄存成 `record_YYYYMMDD.csv`（路徑見 `data/Record/{timestamp}/{data_name}/`），
寫了本機分析腳本 [analysis/diagnose_regime_split.py](analysis/diagnose_regime_split.py)
直接讀取既有記錄，按日期切兩段重新計算指標：

- 前段：2021/10/19 ~ 2022/10/18（Fed 升息劇烈期，251 個交易日）
- 後段：2022/10/19 ~ 2023/10/18（regime 相對穩定後，249 個交易日）

**驗證結果**：

| | DL Total P&L | Baseline Total P&L | DL − Baseline | DL Profit/open 差距 |
|---|---|---|---|---|
| 全期 | 385,436 | 577,465 | -192,030 | -0.0389 |
| 前段（升息劇烈期） | 222,957 | 317,574 | -94,617 | -0.0341 |
| 後段（相對穩定期） | 162,479 | 259,891 | -97,413 | **-0.0444** |

**結論：regime shift 假設被否證。** DL 在後段（regime 穩定後）輸 baseline 的幅度
反而比前段更大（絕對差距 -97,413 vs -94,617；排除交易量影響的單筆品質差距
-0.0444 vs -0.0341，差距還擴大了）。若真是「train/val 沒覆蓋到 test 的市場
regime」造成的，後段應該顯著拉近差距甚至翻盤，但實際上沒有發生，甚至略微惡化。
**問題是結構性的、跟時間區段無關**，不需要往「換時間窗口」或「walk-forward」
的方向處理。

原始輸出存於 [analysis/output/step1_regime_split_comparison.csv](analysis/output/step1_regime_split_comparison.csv)。

## 步驟 2：檢查 T_o 預測分布 — 已驗證並確認成立

用同一份本機腳本，從 DL model 全部記錄中篩選 `Final_Open_Threshold > 0`
（3,263,795 筆，佔 88.33%），統計分布：

```
mean   : 1.4013      median : 1.0406  ← 中位數只有 baseline (1.5) 的 70%
std    : 3.6440       min   : 0.0011
25%    : 0.4959       75%   : 1.8215
90%    : 2.6182       max   : 383.7809（極端值拉高了 mean，median 才是真正的集中趨勢）

比例 < 1.5（baseline 固定門檻）: 66.23%
比例 < 1.0                    : 48.35%
```

**結論：T_o 系統性偏保守假設成立。** median = 1.04，明顯低於 baseline 的 1.5，
且 66% 的預測門檻都低於 1.5。這直接證實了「贏小錢」現象（Win per open 72.42% vs
68.99%、但 True earn amount per win 1.605 vs 1.967 更小）背後的機制：模型系統性
地把門檻設得比 baseline 更小，犧牲單筆獲利空間去換取更高的觸發/反轉機率。

分布圖存於 [analysis/output/step2_To_distribution.png](analysis/output/step2_To_distribution.png)，
統計量存於 [analysis/output/step2_To_distribution_stats.csv](analysis/output/step2_To_distribution_stats.csv)。

### 步驟1+2 合併結論

**問題不是「資料沒覆蓋對 regime」，是「模型本身的門檻校準有結構性偏差」**——不管
哪個時間段，模型都傾向預測偏小的 T_o，這是訓練/loss/最佳化邏輯本身的問題，跟市場
regime 無關。診斷方向收斂為「模型」問題，不是「資料」問題。

下一步懷疑方向（待查）：
1. `find_opt_open_threshold()`（[utils.py:397](src/utils.py:397)）的網格搜尋邏輯——
   `early_stop=5` 是否讓搜尋太早停在偏小的門檻，沒找到真正讓期望利潤最大化的最優點
2. Predictor2-4 預測出的 rtop/top 分布參數（μ, β）本身是否系統性偏小
3. 訓練收斂狀況、loss 權重設計是否讓模型更傾向選保守門檻

## 步驟 6：grid search 搜尋上限被結構性鎖死在 predict_rtop（已驗證並否證）

**發現**：[backtest.py:150-156](src/backtest.py:150) 呼叫 `find_opt_open_threshold()`
時，把 `predict_rtop`（模型預測的 rtop 平均值）當作 `ux` 傳入；而
[utils.py:399](src/utils.py:399) 的網格搜尋：

```python
X = np.linspace(0, ux, opt_grain)   # 候選門檻範圍 = [0, predict_rtop]
```

**理論上的疑慮**：候選的開倉門檻 T_o 搜尋範圍被限制在 `[0, predict_rtop]`，結構上
不可能超過模型自己預測的 rtop 平均值；`rtop` 是右偏的 Gumbel 分布，真實 rtop 實現值
有相當機率落在平均值之上，理論上最優 T_o 可能需要超出這個上限。

**驗證方式**：寫了 [analysis/check_rtop_distribution.py](analysis/check_rtop_distribution.py)，
只讀取 `rtop`/`top`/`Final_Open_Threshold` 欄位（加速讀取），統計：

```
predict_rtop（僅開倉配對, n=3,263,795）：
  median: 2.1768   25%: 1.4870   75%: 3.0483

Final_Open_Threshold / predict_rtop 比例：
  median ratio          : 0.5253   ← 模型選的門檻只是搜尋上限的一半左右
  比例貼近上限 (≥0.95)   : 0.25%   ← 幾乎沒有配對被上限卡住
  比例幾乎等於上限 (≥0.99): 0.21%
```

**結論：假設被否證。** `predict_rtop` 本身中位數是 2.18（比 baseline 1.5 還大），
不算小；而模型實際選中的 `Final_Open_Threshold`（median 1.04）只是這個搜尋上限的
52.5%，貼到上限邊界的配對僅 0.25%。代表 grid search 在範圍**內部**就已經找到最優
解，不是被人為設定的上限卡住——`[0, predict_rtop]` 這個搜尋邊界的設計即使理論上
有缺陷，**也不是造成 T_o 系統性偏低的主因**。

**下一步懷疑方向收斂為**：`expect_profit_Gumbel_integral()` 積分函式本身——根據
模型預測的 rtop/top/close 分布參數跟交易成本算出的期望利潤曲線，為什麼峰值系統性
地落在比較小的門檻處？需要檢查這個積分公式的數學邏輯，判斷這是公式本身的問題，
還是模型預測的分布參數（μ, σ）跟真實市場狀況有落差所造成的合理結果（後者代表
問題還是回到「模型校準」，而非「演算法 bug」）。

## 步驟 7：expect_profit_Gumbel_integral() 數學公式檢查（已驗證，演算法無 bug）

**檢查**：[utils.py:343-367](src/utils.py:343) 對照論文 Eq 3.6：

```python
return (1-GumbelCDF_jit(T, ux, sx))*(T - c) + riemann_sum
```

`(1-GumbelCDF_jit(T,ux,sx))*(T-c)` 對應 Eq 3.6 第一項（normal close 部分，
`(T_o-c)(1-F_rtop(T_o))`），`riemann_sum` 對應第二項三重積分（force close 部分）。
**實作邏輯正確對應論文公式，沒有發現程式 bug。**

**數學機制**：第一項 `f(T) = (T-c) × P(rtop > T)` 是典型的權衡（trade-off）——
`T` 越大每次賺得越多，但 `P(rtop > T)`（真正觸發反轉的機率）越低。這個函式的峰值
位置完全由模型自己預測的 **Gumbel(μ_rtop, β_rtop) 形狀參數**決定，不是寫死的常數。
從 `model_output_summery.csv` 取得的數據：

```
Mean RTop       = 2.609   (μ_rtop)
Mean RTop Sigma = 1.384   (β_rtop)
β/μ ≈ 0.53
```

這個比例餵進積分公式後，數學上算出的最優解就會系統性地落在比 1.5 小的位置——
**這不是程式錯誤，是模型基於它自己預測的分布參數，正確計算出的「局部最優解」**。

**結論：問題不在演算法層面，回到「模型校準」。** `find_opt_open_threshold()`
（步驟6已否證搜尋上限問題）跟 `expect_profit_Gumbel_integral()`（本步驟確認公式
正確）兩個演算法都沒有 bug。問題出在 **Predictor2-4 predict 出來的
μ_rtop/β_rtop/μ_top/β_top/μ_close/σ_close 這些分布參數本身**，餵進這些（正確的）
公式後，算出來的最優門檻系統性偏小——代表模型對美股市場 rtop/top/close 分布形狀
的預測，跟真實市場狀況之間存在校準落差。這跟步驟3.5（模型自己算的
`expected_return` 跟實際 `realized profit` 有落差）形成一致的圖像，兩者都指向
**模型訓練/校準問題**，不是演算法或資料覆蓋範圍的問題。

## 步驟 3：檢查機率篩選的配對（已驗證，結果為部分有效但可能偏保守）

DL model 用 `open_prob_threshold = 0.65` 過濾 revert 機率過低的配對（對應
`Non_Open` 標籤，佔 6.82%），baseline 沒有這個機制。

**驗證方式**：寫了 [analysis/check_prob_filter.py](analysis/check_prob_filter.py)，
利用 `records_df` 對每個配對都存有 ground truth 的 `Revert` 欄位（不論是否真的
開倉，事後都知道這個配對的價差有沒有真的反轉回均值），不需重跑 backtest 即可檢查
模型 `Revert_Prob` 預測跟真實結果的校準關係。

**校準曲線（單調遞增，模型機率輸出本身是有意義的排序信號）**：

```
prob區間        n          真實revert率
(0.35-0.40]   11,664        56.5%
(0.55-0.60]   60,887        65.6%
(0.60-0.65]   78,137        67.3%   ← 被篩掉的最後一段
(0.65-0.70]  101,777        69.7%   ← 開始被保留
(0.90-0.95]1,600,079        86.8%
```

```
全部配對的 Revert率：        81.18%
Non_Open 被篩掉的 Revert率：  63.98%
其他（保留）配對的 Revert率： 82.44%
```

**結論：機率篩選方向正確（64% vs 82%確實有差距），但可能偏保守。** 被篩掉的配對裡，
雖然只有 64% 最終會反轉，但**這些「確實反轉」的配對，反轉前的價差幅度（Norm_Rtop
中位數 3.71）反而比全部配對的反轉中位數（2.73）還大**——代表這群配對命中率較低，
但一旦命中，潜在獲利空間反而更大，這部分上行空間被完全捨棄掉了。

**這跟步驟2（T_o偏小）、步驟3.5（expected profit過濾誤殺正期望值配對）是同一種
模式**：篩選機制的方向沒有錯，但 cutoff 的位置可能犧牲了「贏的時候賺更多」的那群
配對，用命中率換取單筆規模，三個獨立機制都呈現一致的結構性現象。

原始輸出存於 [analysis/output/step3_prob_calibration.csv](analysis/output/step3_prob_calibration.csv)
跟 [analysis/output/step3_prob_filter_summary.csv](analysis/output/step3_prob_filter_summary.csv)。

### 步驟3 後續：must_open=True 重跑驗證（已驗證，假設成立）

把 `hyperparameters.py` 的 `must_open` 改成 `True`（強制 `open_prob_threshold=0.0`，
[hyperparameters.py:146](src/hyperparameters.py:146)），先在 GPU 重跑
`inference_only` 重建 `inference_cache/`，再用 CPU 重跑 `backtest_from_saved`，
讓原本被篩掉的配對也真的被丟進交易模擬。

**結果**：

```
Total open count   3,207,980 → 3,423,248   (+215,268)
Total P&L            385,436 →   401,839   (+16,403, +4.3%)
MDD                  -48,283 →   -53,370   (變差)
Sharpe                 3.050 →     2.968   (微降)
```

因為 `To`/`Tsl` 的計算完全不依賴 `revert_probability`（只依賴 predict_rtop/top/close
跟相關係數），原本機率 ≥0.65 那群配對的交易結果在兩次跑法中完全相同，所以
Total P&L 的增量 (+16,403.64) 精確等於「原本被篩掉的25萬筆配對」單獨拿出來看的
真實淨損益——用 [analysis/check_prob_filter_profit.py](analysis/check_prob_filter_profit.py)
直接從新的 `record_*.csv` 驗證，數字完全吻合：

```
原本會被篩掉的配對 (n=251,913, 真開倉 215,268)：
  Total P&L        : +16,403.64   ← 確實淨賺錢，假設成立
  Profit per open   : 0.0762       ← 但效率只有保留組的 63%（保留組 0.1201）
  Win rate per open : 54.49%       ← 遠低於保留組的 ~71%
```

按 Revert_Prob 0.05 一格分箱檢查，**所有級距（含最低的 0.35-0.40）的
`total_profit` 都是正值**，沒有任何一個機率區間是淨虧損的；且低機率區間「贏的
時候」平均贏的金額（mean_win_amount≈2.7）反而比高機率主力區間（0.9-0.95 只有
1.14）更大，再次驗證「低機率但大賺一筆」的模式存在。

**結論**：假設成立，0.65 門檻確實濾掉了真實利潤，但效益有限——只填補了 DL 跟
baseline 差距（-192,030）的 8.5%，且是用更大風險（MDD惡化、Sharpe/Sortino微降）
換來的，不是能讓 DL 翻盤的決定性因素。跟步驟3.5（expected profit過濾）的結論
模式一致：篩選方向沒錯，但目前的 cutoff 都不是「總獲利最大化」的最優點，單獨調
任何一個參數效益都有限。

原始輸出存於 [analysis/output/step3_profit_by_bin.csv](analysis/output/step3_profit_by_bin.csv)。

### 附帶發現：Predictor1 的 revert 機率輸出分布異常集中在高機率區間

用相同的 `record_*.csv` 抽樣檢查 `Revert_Prob` 的最小值與分布，發現模型**幾乎不會
輸出低於 0.3 的 revert 機率**（50天樣本中最小值僅 0.345，mean=0.846，
median=0.890）。`step3_profit_by_bin.csv` 裡完全沒有 `prob_bin < 0.3` 的資料列，
不是程式或分箱邏輯的問題，是真實資料分布如此——這些低機率 bin 是空的。

**這可能是另一個校準問題的線索**：Predictor1 的 Sigmoid 輸出整體被「壓縮」在高
信心區間，模型很少表達低信心判斷。這跟步驟2/6/7看到的「Predictor2-4 預測的分布
參數跟真實市場有落差」是同一個大方向的現象，但發生在 Predictor1（二元分類）而非
Predictor2-4（分布參數回歸）——代表校準問題可能不是單一 predictor 的個別缺陷，
而是訓練流程/loss設計層面的系統性現象，值得在後續檢查訓練收斂狀況時一併納入。

## 步驟 3.5：Expected Profit 為負的配對未被過濾（已發現、已驗證、已否證並撤回）

**發現**：`get_Thresholds_ExpProf()` 已經算出每個配對在最佳門檻下的
`expected_return`（對應論文 Eq 3.6-3.8 的三重積分 Z），但原本
`do_discard_expect_prof_threshold = False`，導致即使 `expected_return < 0`
（模型自己預測這筆交易會虧錢），系統仍然照常開倉。baseline 沒有這個計算，
所以沒有「明知會虧但還是開倉」的矛盾。

**重要澄清**：方向（多/空）由價差碰到哪一側門檻自動決定（對稱式策略，
`rtop`/`top`/`close` 皆取絕對值建模），`expected_return < 0` 不是「選錯方向」，
而是這個配對在當下統計特性下「不管哪個方向都不划算」。

**測試的修正**：`do_discard_expect_prof_threshold = True`、`expect_prof_threshold = 0`，
過濾掉所有 `expected_return < 0` 的配對。

**驗證結果（用同一份 inference_cache 重跑 backtest_from_saved）**：

| 指標 | 過濾前 (False) | 過濾後 (True) | 變化 |
|---|---|---|---|
| Total open count | 3,207,980 | 2,740,084 | 少了 467,896 筆 |
| Discard per record | 0% | 12.85% | 被擋掉的比例 |
| Profit per open | 0.1201 | 0.1292 | ✅ 提升 7.6% |
| True earn per win | 1.605 | 1.756 | ✅ 提升 9.4% |
| MDD（絕對值） | -48,283 | -33,467 | ✅ 改善 31% |
| **Total P&L** | **385,436** | **354,092** | **❌ 下降 8.1%** |

**結論：假設被否證，已撤回修正**（[hyperparameters.py:156](src/hyperparameters.py:156)
改回 `do_discard_expect_prof_threshold = False`）。

被擋掉的 46.8萬筆配對，反推其真實平均獲利 = (385,436−354,092)/467,896 ≈ **+0.067**
（仍是正值，只是遠低於整體平均 0.12）。代表模型的 `expected_return` 排序方向其實
是對的（篩掉的確實是品質較差的配對），但拿「0」當絕對門檻去砍，砍過頭了——把一部分
「普通賺、但沒有很賺」的配對跟「真正會虧」的配對一起丟掉，淨效果是總量損失大於
品質提升的好處。對「Total P&L 要贏 baseline」這個目標是反效果，雖然對風險控制
（MDD 大幅改善）跟單筆品質有正面幫助。

**後續可能方向（低優先度）**：若之後想重新嘗試，可改用更負的 `expect_prof_threshold`
（例如 -0.05、-0.1）重新校準門檻，而不是用 0 一刀切。但目前判斷主因仍在步驟 1、2，
應優先處理。

## 步驟 4：確認 MDD_percent 計算公式

`utils.calculate_MDD(percent=True)` 公式為：

```
MDD_percent = max_{i,j<i} [ (cumulative_profit[j] - cumulative_profit[i]) / cumulative_profit[j] ]
```

分母是「峰值當下的累積損益值」，不是固定本金或平均資金。若峰值本身很小（接近 0），
即使絕對跌幅不大，比例也會被放大到失真（例如 DL 絕對 MDD 較小 -48,283，但
MDD_percent 反而較大 1446% vs baseline 的 496%）。

**結論**：比較風險時優先看絕對值 `MDD`（DL 風險控制其實更好），`MDD_percent` 在
損益曲線早期或接近 0 的時候容易失真，不建議當作主要比較依據。

## 步驟 5：檢查資料清洗輪數

```
├─ 還在清洗中、ill-conditioned 樣本還在持續產生
│   → 結論：「資料」品質問題，模型還沒在乾淨資料上收斂
│   → 解法：先把清洗迴圈跑完，再重新訓練評估
│
└─ 清洗已收斂（無新增 ill-conditioned 樣本）
    → 排除這個因素
```

檢查方式：確認 `data_cleaning/ill_conditioned_data_v*.csv` / `*.done` 最新版本
狀態（邏輯見 [hyperparameters.py:194-210](src/hyperparameters.py:194)）。

---

## 執行順序總結

1. 步驟 1（分段 backtest）：**已驗證並否證**（新舊 checkpoint 各驗證一次，結論穩健）
   — regime shift 不是主因。
2. 步驟 2（T_o 分布）：**已驗證並確認成立，checkpoint 修正後改善但未消除**
   — median T_o 從 1.04 回升到 1.16，仍低於 baseline 1.5。
3. 步驟 6（grid search 搜尋上限）：**已驗證並否證** — 上限設計非主因，最優解多在範圍內找到。
4. 步驟 7（expect_profit_Gumbel_integral 公式檢查）：**已驗證，演算法無 bug** —
   公式正確對應論文 Eq 3.6，問題在模型預測的分布參數本身。
5. 步驟 3（機率過濾驗證）：**已驗證並確認成立，且與 checkpoint 無關**
   — 新舊 checkpoint 效果幾乎相同（+16,403 vs +17,155），是持續存在的獨立問題。
6. 步驟 4（MDD 公式確認）：純公式釐清，已說明，不需額外行動。
7. 步驟 3.5（expected profit 過濾）：**原本否證，但用修正後 checkpoint 重新驗證後
   結論大幅翻轉** — 傷害從 -8.1% 縮小到 -0.65%，證實此問題主要由 checkpoint bug
   造成，已隨修正大幅緩解。
8. 步驟 8（checkpoint 選擇 bug）：**已發現、已修正、已驗證，效益顯著**
   — 單獨補回 Total P&L +53,135（+13.8%），補回原始差距的 28%。
9. 步驟 5（資料清洗輪數）：視後續結果決定是否需要，成本較高（可能需要重新訓練）。
10. 步驟 9（RTop/Top/Close 離散度比例）：**⚠️ 部分結論已被步驟12推翻**——σ/μ比值
    偏高的描述性統計本身沒錯，但不能直接解讀成「σ校準過頭」。
11. 步驟 12（σ 校準檢驗）：**重大修正** — 用真實標籤直接算z-score校準檢驗，發現
    z的std（RTop 2.98、Top 2.51、Close 1.37）全部遠大於校準得當時的理論值
    （≈1.28、≈1.28、1.0），方向跟「σ過寬」相反——代表模型可能低估自己的不確定性，
    不是高估。「variance inflation」這個特定機制不再是站得住腳的解釋，需要
    重新檢視。

**目前進度**：經過完整的 checkpoint 修正 + 四項假設在新舊 checkpoint 下的雙重驗證，
「模型校準」問題被進一步拆解為性質不同的兩個子問題：

- **已大幅緩解**：`expected_return`（μ_rtop/β_rtop 等分布參數）的校準問題，主因是
  checkpoint 選擇 bug（用了訓練後期已劣化的權重），修正後傷害幾乎消失（-8.1%→-0.65%）。
- **依然存在、與 checkpoint 無關**：`revert_prob`（Predictor1）機率輸出集中在高信心
  區間、T_o 系統性偏小（median 1.16 vs 1.5），這些現象在新舊 checkpoint 下表現一致，
  代表是更根本的訓練/loss/資料層面問題，不是單純的 checkpoint 選擇失誤。
- **⚠️ 已修正**：RTop/Top/Close 分布離散度比例偏高（步驟9）曾被解讀為「σ過寬/
  variance inflation」，但步驟12的直接校準檢驗顯示方向可能相反（σ相對真實誤差
  反而偏小）。這條推論鏈需要下修，不宜再直接引用「模型故意放寬σ規避NLL懲罰」
  這個特定敘事——但 loss curve 不穩定（步驟10）跟 eps bug（步驟11）本身是獨立
  驗證成立的，不受這次修正影響。

**累計進度（跟 Baseline 577,465 的差距演進）**：

```
原始（舊checkpoint, 預設參數）        : 385,436   gap = -192,030
+ checkpoint 修正                     : 438,571   gap = -138,895 （補回 53,135, 28%）
+ must_open=True（疊加，步驟8.1）      : 455,726   gap = -121,739 （再補回 17,155）
```

即使疊加兩項修正，仍有 **-121,739（原始差距的 63%）尚未解釋**，代表 checkpoint bug
只是問題的一部分，不是全部。方向依然**確定收斂為「模型校準」問題**——不是「資料
覆蓋範圍」、「regime」或「演算法 bug」，但現在更精確地知道：問題集中在 Predictor1
的機率校準、T_o 整體仍偏小、以及三個分布離散度偏高這幾點，不是 Predictor2-4 的
expected_return 估計（那部分已隨 checkpoint 修正大幅改善）。

**步驟10（Loss Curve）帶來的關鍵轉折**：用戶提供的 loss curve 顯示 train loss
提早凍結但 val/test loss 劇烈震盪且不隨訓練收斂，震盪幅度在訓練尾聲反而最大
（epoch 74 甚至衝到 13+）。這強烈指向**資料清洗不足是更上游的共同成因**——
残留的 ill-conditioned 樣本持續在訓練中製造數值不穩定，可能同時是「Predictor1
過度自信」與「三個分布離散度偏高（模型學到用寬分布自保以降低 NLL 懲罰）」的
共同上游原因。這跟步驟5（v0→v1清洗後 ill-conditioned 樣本不減反增）相互印證。

**下一步方向（已依 loss curve 發現重新排序優先順序）**：

1. **【最優先】資料清洗 v1→v2**（回到步驟5）：loss curve 已提供直接證據，指向
   資料清洗是更上游的成因。應先把清洗迴圈跑完一輪，確認 ill-conditioned 樣本數
   是否終於開始下降，再評估是否需要重新訓練。
2. **重新訓練並比較 loss curve**：用 v2 清洗後的資料重新訓練，觀察 val/test
   loss 的震盪幅度是否隨清洗減少而收斂，驗證「資料清洗不足」這個假設。
3. **loss function 檢查**（次優先，待資料清洗後再看是否仍需要）：檢查
   `loss.py` 裡 `GaussCopGumLoss` 對 Predictor1（BE loss）跟 Predictor2-4
   （joint NLL loss）的權重設計，是否有系統性偏誤導致模型傾向輸出過度自信的
   機率、或用寬分布自保。
4. **美股 hyperparameter 重新校準**（低優先度）：波動率尺度/tick size/流動性
   跟台股不同，若前兩項處理後問題仍未解決，再考慮是否需要調整。

「資料」問題與「模型」問題的解法方向完全不同——資料問題需要繼續清洗迭代；模型
問題則是調整 loss、超參數或訓練策略。**loss curve 的證據顯示，這次的根本原因
很可能主要是「資料」問題（清洗不足），而不是單純的模型設計缺陷**——建議優先
把資料清洗迭代跑完，這是成本相對較低、且有直接證據支持的方向，再視結果決定
是否需要進一步調整 loss function 或 hyperparameter。
