import pandas as pd
import os
import re
import sys

# 讀取 src/hyperparameters.py 的設定（統一設定入口，不要在 Colab 額外設環境變數）
sys.path.insert(0, "/content/GGWP_US_LLTL/src")
import hyperparameters

# ==========================================
# 1. 設定與路徑配置 (Colab 本地路徑)
# ==========================================
BASE_DIR = "/content/GGWP_US_LLTL/data_cleaning"
os.makedirs(BASE_DIR, exist_ok=True)

def get_versioned_path(base_name, extension, folder=BASE_DIR, next_version=False):
    """
    偵測目前資料夾中最高的版本號。
    next_version=False: 回傳目前「最新」的檔案路徑 (用於讀取來源)
    next_version=True:  回傳「下一個」可用的檔案路徑 (用於儲存結果)
    """
    # 建立正則表達式，處理 v0, v1, v2...
    pattern = re.compile(fr"{base_name}_v(\d+)\{extension}")
    max_v = -1
    
    if os.path.exists(folder):
        for f in os.listdir(folder):
            match = pattern.fullmatch(f)
            if match:
                max_v = max(max_v, int(match.group(1)))
    
    if next_version:
        # 儲存時，序號自動 +1
        return os.path.join(folder, f"{base_name}_v{max_v + 1}{extension}")
    else:
        # 讀取時，若沒檔案則預設 v0
        current_v = max(0, max_v)
        return os.path.join(folder, f"{base_name}_v{current_v}{extension}")

# --- 自動取得路徑 ---
# INPUT_PICKLE_PATH 跟 main.py 訓練時讀取的資料版本保持一致：
# 若 hyperparameters.data_version_override 有設定，強制使用該版本；
# 否則自動抓「目前最新」的 cleaned_data_v{n}.pickle。
if hyperparameters.data_version_override is not None:
    _n = int(hyperparameters.data_version_override)
    INPUT_PICKLE_PATH = os.path.join(BASE_DIR, f"cleaned_data_v{_n}.pickle")
    print(f"[Data Version Override] 強制清洗 v{_n}（忽略自動偵測最新版本的邏輯）")
else:
    INPUT_PICKLE_PATH = get_versioned_path("cleaned_data", ".pickle")

# 注意：這裡不能直接用 hyperparameters.log_path/sigma_log_path——那兩個是給
# main.py 訓練時規劃「這次要寫入哪個新版本」用的，若對應版本的 .done 已存在，
# 邏輯上會自動推進到下一個版本號。clean_data.py 要的是「剛訓練完、實際已經
# 寫好的那個版本」，所以維持用 get_versioned_path 單純找目前已存在的最大版本號。
CSV_LOG_PATH = get_versioned_path("ill_conditioned_data", ".csv")
SIGMA_CSV_LOG_PATH = get_versioned_path("sigma_collapse_data", ".csv")

# 準備存入下一個版本
OUTPUT_PICKLE_PATH = get_versioned_path("cleaned_data", ".pickle", next_version=True)

print(f"📦 讀取原始資料版本：{os.path.basename(INPUT_PICKLE_PATH)}")
print(f"📝 參考異常 ID 紀錄（相關係數矩陣接近奇異）：{os.path.basename(CSV_LOG_PATH)}")
print(f"📝 參考異常 ID 紀錄（sigma 崩塌）：{os.path.basename(SIGMA_CSV_LOG_PATH)}")
print(f"🚀 清洗後將儲存為：{os.path.basename(OUTPUT_PICKLE_PATH)}")

# ==========================================
# 2. 主要清洗流程
# ==========================================
def main():
    print("\n" + "="*30)
    print("🚀 開始執行資料清洗流程")
    print("="*30)
    
    # 1. 收集壞資料 ID（兩個獨立來源都要讀：相關係數矩陣異常 + sigma 崩塌）
    #
    # 注意：log 檔案可能是訓練中途斷線、用不同版本的 loss.py 接續寫入同一個檔案，
    # 導致同一個 CSV 裡欄位數量不一致（例如舊版本沒有 epoch 欄位、新版本有）。
    # pandas.read_csv 對欄位數量不固定的檔案會直接拋錯，所以改用逐行讀取、
    # 只取每行「第一個逗號分隔值」（也就是 Data_ID，不管 schema 後面多幾欄都不影響）。
    def extract_data_ids_robust(path):
        ids = set()
        bad_lines = 0
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            next(f, None)  # 跳過 header
            for line in f:
                first_field = line.split(',', 1)[0].strip()
                if not first_field:
                    continue
                try:
                    ids.add(int(first_field))
                except ValueError:
                    bad_lines += 1
        return ids, bad_lines

    bad_ids = set()
    for path, label in [(CSV_LOG_PATH, "相關係數矩陣接近奇異"),
                         (SIGMA_CSV_LOG_PATH, "sigma 崩塌")]:
        if os.path.exists(path):
            try:
                ids_in_file, bad_lines = extract_data_ids_robust(path)
                bad_ids.update(ids_in_file)
                msg = f"✅ 成功載入 {len(ids_in_file)} 筆異常 ID（{label}）"
                if bad_lines:
                    msg += f"（{bad_lines} 行無法解析，已跳過）"
                print(msg)
            except Exception as e:
                print(f"❌ 讀取 CSV 失敗 ({label}): {e}")
        else:
            print(f"ℹ️ 找不到 {label} 的紀錄檔（{os.path.basename(path)}），略過。")

    if not bad_ids:
        print("🎉 無異常 ID 需要處理，流程結束。")
        return

    # 2. 讀取並清洗數據
    if not os.path.exists(INPUT_PICKLE_PATH):
        print(f"❌ 錯誤：找不到來源檔案 {INPUT_PICKLE_PATH}")
        return

    print(f"正在讀取 {os.path.basename(INPUT_PICKLE_PATH)}...")
    df = pd.read_pickle(INPUT_PICKLE_PATH)
    original_len = len(df)

    # 確保 Data_ID 存在 (防止原始資料沒這欄位)
    if 'Data_ID' not in df.columns:
        print("⚠️ 原始資料無 Data_ID 欄位，自動將 index 設為 Data_ID")
        df['Data_ID'] = df.index

    # 核心邏輯：剔除在 bad_ids 名單內的資料
    cleaned_df = df[~df['Data_ID'].isin(bad_ids)].copy()
    
    # 3. 重新編排索引並儲存
    cleaned_df.reset_index(drop=True, inplace=True)
    
    print(f"正在儲存至 {os.path.basename(OUTPUT_PICKLE_PATH)}...")
    cleaned_df.to_pickle(OUTPUT_PICKLE_PATH)

    # 4. 統計報告
    new_len = len(cleaned_df)
    drop_count = original_len - new_len
    print("\n📊 清洗結果統計")
    print(f"  - 原始總筆數: {original_len:,}")
    print(f"  - 剔除異常數: {drop_count:,}")
    print(f"  - 剩餘淨筆數: {new_len:,}")
    if original_len > 0:
        print(f"  - 剔除百分比: {(drop_count/original_len)*100:.4f}%")
    print("="*30)

if __name__ == '__main__':
    main()