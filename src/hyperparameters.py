from collections import namedtuple
import utils
import os
import glob


testcase = 4
do_rolling_window = False

if testcase == 1:
    # small testcase
    start_year, start_month = 2018, 10
    end_year, end_month = 2018, 12
    split_dict = {
        "train_start": 20181019,
        "train_end": 20181119,
        "validate_start": 20181120,
        "validate_end": 20181131,
        "test_start": 20181201,
        "test_end": 20181231,
    }
elif testcase == 2:
    # full training testcase
    start_year, start_month = 2018, 10
    end_year, end_month = 2021, 12
    split_dict = {
        "train_start": 20181019,
        "train_end": 20201018,
        "validate_start": 20201019,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20211231,
    }
elif testcase == 3:
    # full testing testcase
    start_year, start_month = 2021, 10
    end_year, end_month = 2023, 10
    split_dict = {
        "train_start": 20211001,
        "train_end": 20211015,
        "validate_start": 20211016,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20231018,
    }
elif testcase == 4:
    # full testcase
    start_year, start_month = 2018, 10
    end_year, end_month = 2023, 10
    split_dict = {
        "train_start": 20181019,
        "train_end": 20201018,
        "validate_start": 20201019,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20231018,
    }
elif testcase == 5:
    # full tmp testing testcase
    start_year, start_month = 2021, 10
    end_year, end_month = 2023, 10
    split_dict = {
        "train_start": 20211001,
        "train_end": 20211015,
        "validate_start": 20211016,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20231018,
    }
else:
    raise ValueError("wrong testcase number!!!")


# PathRoot = r"C:\Users\nycu_dev1\Desktop\Chin-Yi_Code\\"
# preprocess_data_path = PathRoot + "preprocess_data/" + f'preprocess_data_{start_year}_{start_month:02d}_to_{end_year}_{end_month:02d}.pickle'

PathRoot = "/content/GGWP_US_LLTL/data/"

# ── preprocess data 版本選擇 ──────────────────────────────────────
# 資料夾結構：{PathRoot}preprocess_data/orginal_preprocess_data/
#            {PathRoot}preprocess_data/new_preprocess_data/
# "original" = 使用 orginal_preprocess_data 子資料夾
# "new"      = 使用 new_preprocess_data 子資料夾
preprocess_data_version = "new"
_preprocess_data_subdir = {
    "original": "orginal_preprocess_data",
    "new": "new_preprocess_data",
}[preprocess_data_version]
preprocess_data_dir = f"{PathRoot}preprocess_data/{_preprocess_data_subdir}"

preprocess_data_path = preprocess_data_dir + f"/preprocess_data_{start_year}_{start_month:02d}_to_{end_year}_{end_month:02d}.pickle"

current_time = utils.get_time()

# Market parameters
Tax = 0.0000278
open_delete = 16
close_delete = 0
formation_period = 150
trading_period = 224

# ── formation table 版本選擇 ──────────────────────────────────────
# 資料夾結構：{PathRoot}formation_table/orginal_formation_table/
#            {PathRoot}formation_table/new_formation_table/
# "original" = 使用 orginal_formation_table 子資料夾
# "new"      = 使用 new_formation_table 子資料夾
formation_table_version = "new"
_formation_table_subdir = {
    "original": "orginal_formation_table",
    "new": "new_formation_table",
}[formation_table_version]
formation_table_dir = f"{PathRoot}formation_table/{_formation_table_subdir}"



# model parameters
batch_size = 256
lr = 1e-4
norm_clip = 1e-3
optim = "Adam"


# training parameters
num_epochs = 1000
early_stop_count = 50
save_freq = 1
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 8
loss_set = ["GaussCopGumLoss", "GaussCopNormLoss", "GumIndLoss", "NormIndLoss"]

loss_type = loss_set[0]
if loss_type in ["GaussCopGumLoss", "GaussCopNormLoss"]:
    do_copula_model = True
else:
    do_copula_model = False

# model save path
save_root = f"{PathRoot}Train_ckpt/"

# ── 是否接續現有 checkpoint 的權重 ──────────────────────────────
# True  = 接續 save_root 底下最新的 checkpoint（依修改時間排序）
# False = 全新訓練，忽略既有 checkpoint（例如這次要套用修正後的 loss，
#         想從頭訓練而不是拿舊 loss 訓練出來的權重當初始值）
resume_from_checkpoint = False

def find_latest_checkpoint(save_root):
    if not resume_from_checkpoint:
        print("[Auto Resume] 已選擇全新訓練，忽略既有 checkpoint")
        return None
    pattern = os.path.join(save_root, "**", "*.pth")
    all_ckpts = glob.glob(pattern, recursive=True)
    if not all_ckpts:
        return None
    return max(all_ckpts, key=os.path.getmtime)

model_path = find_latest_checkpoint(save_root)
#model_path = "/content/GGWP_US_LLTL/data/Train_ckpt/2026-07-11 15-44-20/BestEpoch=24_ValLoss=5.7244.pth"

if model_path:
    print(f"[Auto Resume] 偵測到 checkpoint：{model_path}")
else:
    print("[Auto Resume] 無 checkpoint，從頭開始訓練")
# 執行模式：
#   "train"              - 訓練 + 回測（GPU）
#   "test"               - 完整回測，inference + backtest 一起跑（GPU）
#   "inference_only"     - 只跑 GPU inference，結果存到 inference_cache/，不做回測
#   "backtest_from_saved"- 從 inference_cache/ 載入 inference 結果，純 CPU 跑回測
mode = "test"  # 重新從最原始資料(v0)訓練，套用修正後的 loss（eps保護 + sigma崩塌監控）
                # 資料版本由下面的 data_version_override 控制，不需要另外設定







# trading behavior
must_open = False  # 步驟3實驗第1輪已完成並記錄，改回預設值，避免跟第2輪的變數混在一起
open_prob_threshold = 0.65
do_dynamic_sl = True
if do_dynamic_sl is True:
    sl_offset = None
    sl_sigma = 0.999
else:
    sl_offset = 100000
    sl_sigma = None

do_discard_expect_prof_threshold = False  # 步驟：「拿掉停損」實驗改為 False，
                                           # 避免跟 disable_stop_loss 的效果混在一起
if do_discard_expect_prof_threshold:
    expect_prof_threshold = 0
else:
    expect_prof_threshold = None

use_adj_expect_prof = False

# ── 停損機制實驗（拿掉停損，只保留到期強制平倉/正常平倉）──────────────
# 不重新訓練模型：只在交易模擬前，把 Tsl（停損門檻）覆寫成一個實務上
# 不可能被觸發到的數值，讓 trade.py 的邏輯自然退化成「正常平倉 vs 到期強制平倉」，
# 不需要改動 trade.py 本身（跟 baseline_mode 用同一套「Tsl 設超大值=不停損」的機制）。
# True  = 停損不會被觸發（結果存到獨立的 _noStopLoss 資料夾，方便對照）
# False = 維持原本的動態停損
disable_stop_loss = False

# ── Post-hoc sigma 校正（驗證「σ低估 → 模型過度自信」假說用）────────────
# 不重新訓練模型：只在 mode="backtest_from_saved" 讀取 inference_cache 之後、
# 交易模擬之前，把模型預測的 sigma_rtop/top/close 乘上校正倍率。
# 倍率來自 z-score 校準測試量出的偏差（z_std 實際值 / 理論值）。
# True  = 套用校正並重跑回測（結果存到獨立的 _sigmaCorrected 資料夾，方便對照）
# False = 維持原始模型輸出，不校正
do_posthoc_sigma_correction = False
posthoc_sigma_correction_factors = {
    "rtop":  2.32,
    "top":   1.96,
    "close": 1.37,
}

# baseline model（固定1.5σ開倉門檻，不使用DL模型）
# True  = 跑1.5σ固定門檻 baseline 回測
# False = 使用DL模型預測門檻
baseline_mode = False

# constant threshold
do_constant_output = False

# Global parameters:
# 2016~201710
global_paras = dict()
global_paras["rtop"] = [2.4693460030466663, 1.0608914863005185]
global_paras["top"] = [3.70266662086769, 1.6990610342437076]
global_paras["close"] = [3.0172715942390242, 3.0833261860426373]
global_paras["cormat"] = [0.1790868, -0.10757489, 0.75402701]

arg_define = namedtuple("arguments",
                 'batch_size, lr, norm_clip, num_epochs, optim, early_stop_count, save_root, '
                 'device, save_freq, num_workers, do_constant_output, global_paras, must_open,'
                 'Loss, sl_offset, sl_sigma, model_path, mode, do_discard_expect_prof_threshold, expect_prof_threshold,'
                 'do_dynamic_sl, use_adj_expect_prof, do_copula_model, open_prob_threshold, baseline_mode')
args = arg_define(batch_size, lr, norm_clip, num_epochs, optim, early_stop_count, save_root,
           device, save_freq, num_workers, do_constant_output, global_paras, must_open,
           loss_type, sl_offset, sl_sigma, model_path, mode, do_discard_expect_prof_threshold, expect_prof_threshold,
           do_dynamic_sl, use_adj_expect_prof, do_copula_model, open_prob_threshold, baseline_mode)
split_info_define = namedtuple("arguments",
                 'split_dict, start_year, start_month, end_year, end_month')
split_info = split_info_define(split_dict, start_year, start_month, end_year, end_month)


# ── 訓練資料版本覆寫 ──────────────────────────────────────────────
# None  = 維持原本行為，main.py 自動偵測最新的 cleaned_data_v{n}.pickle
# 整數n = 強制使用 cleaned_data_v{n}.pickle，忽略「自動抓最新版本」的邏輯
#         （例如設成 0，重新從最原始資料開始清洗，套用修正後的 loss）
# clean_data.py 也會讀這個設定，兩邊資料版本會自動保持一致。
data_version_override = 0

# ── ill-conditioned data CSV 路徑 ────────────────────────────────
# 判斷邏輯：
#   - v{n}.done 存在 → 上次訓練正常完成 → 往下一個版本 (v{n+1})
#   - v{n}.done 不存在但 v{n}.csv 存在 → 上次中途中斷 → 接續 v{n}
#   - 都不存在 → 第一次執行 → 建 v0
_ill_csv_base = '/content/GGWP_US_LLTL/data_cleaning/ill_conditioned_data_v'
n = 0
while os.path.exists(f"{_ill_csv_base}{n}.csv"):
    if os.path.exists(f"{_ill_csv_base}{n}.done"):
        n += 1  # 這版已正常完成，用下一個版本
    else:
        break   # 這版尚未完成（中途中斷），接續寫入
log_path = f"{_ill_csv_base}{n}.csv"
if os.path.exists(log_path):
    print(f"[Ill-data log] 接續寫入 v{n}：{log_path}")
else:
    print(f"[Ill-data log] 新建 v{n}：{log_path}")
if data_version_override is not None:
    print(f"[Data Version Override] 訓練資料強制使用 cleaned_data_v{data_version_override}.pickle"
          f"（本次 log 版本號 v{n} 跟資料版本號可能不同，這是預期行為——"
          f"log 只是為了記錄「用修正後 loss 訓練 v{data_version_override} 時發現的異常樣本」，"
          f"清洗時要對應套用到 v{data_version_override}.pickle，不是套用到自動偵測的最新版本）")

# ── sigma 崩塌監控 CSV 路徑（跟 ill-conditioned log 共用同一個版本號 n）──
# 記錄 sigma_rtop/sigma_top/sigma_close 崩塌到接近 0 的樣本，這是導致
# loss 跑到異常負值的第二條路徑（跟相關係數矩陣條件數過大是獨立的兩件事）
_sigma_csv_base = '/content/GGWP_US_LLTL/data_cleaning/sigma_collapse_data_v'
sigma_log_path = f"{_sigma_csv_base}{n}.csv"
sigma_collapse_threshold = 1e-2  # 低於此值視為異常崩塌（正常尺度約1.5~3.5，遠高於此門檻）
