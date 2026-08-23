# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:19:53 2026


@author: Hao Han Chang

正式產生formation table用的平行化程式：
讀取 data/full_data_AB 的分鐘股價，對每個交易日的每一對股票配對做
portmanteau test、Normality test、Johansen共整合檢定，輸出符合
trade.py/preprocess.py讀取格式的formation table CSV，存到
hyperparameters.formation_table_dir（依formation_table_version切換
orginal_formation_table / new_formation_table）。

支援中斷後接續執行：每個交易日一個輸出檔，執行前會先檢查對應輸出檔案
是否已存在，已存在就跳過該日，不會重算。
"""

# ---- 務必在 import numpy / statsmodels 之前設定，避免 BLAS 內部平行與
#      multiprocessing 的行程平行互搶 CPU 資源 ----
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 確保跟 mt.py 同層時可以直接 import
sys.path.insert(0, "/content/GGWP_US_LLTL/src")  # 讀取共用的 hyperparameters.py
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import mt
import hyperparameters
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm


# 開盤捨棄、建模期間：直接沿用 hyperparameters.py 的值（不再各自寫死一份），
# 確保輸出檔名（{date}for{inNum}del{form_del_min}_AB.csv）永遠跟實際計算用的
# 期間一致，不會因為兩邊各自維護一份常數而漂移不同步。
form_del_min = hyperparameters.open_delete
inNum = hyperparameters.formation_period

dat_path = hyperparameters.PathRoot + "full_data_AB/"
save_path = hyperparameters.formation_table_dir + "/"

# 欄位名稱對齊 trade.py / preprocess.py 讀取formation table時預期的欄位
# （VECM(q)、VECM_Model_Type、Johansen_intercept/slope/std、w1、w2）
TABLE_COLUMNS = ['S1', 'S2', 'VECM(q)', 'Johansen_intercept', 'Johansen_slope',
                  'Johansen_std', 'VECM_Model_Type', 'w1', 'w2', 'Del_min']

# 每個task打包幾個配對一起送給worker，減少submit()/IPC次數（見前次優化的說明）。
CHUNK_SIZE = 200


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def formation_table_filename(input_filename):
    """輸入 '2018-10-24_AB.csv' -> 輸出 '20181024for150del16_AB.csv'（trade.py讀取的檔名格式）"""
    date_str = input_filename[:10].replace('-', '')
    return f"{date_str}for{inNum}del{form_del_min}_AB.csv"


# ── worker：處理單一配對，回傳一個結果 row（或 None 代表被捨棄）──
def process_one_pair(task):
    s1_name, s2_name, rowAS = task
    try:
        p = mt.order_select(rowAS, max_p=5)

        # portmanteau test
        model = VAR(rowAS)
        # if model.fit(p).test_whiteness(nlags=p + 1).pvalue < 0.05:
        #     return None  # 沒通過portmanteau test，捨棄該配對
        # # Normality test
        # if model.fit(p).test_normality().pvalue < 0.05:
        #     return None  # 沒通過Normality test，捨棄該配對

        opt_model = mt.JCI_AutoSelection(rowAS, p - 1)  # bic based model selection
        if opt_model <= 0:
            return None  # 無共整合關係，捨棄該配對

        F_a, F_b, F_ct, F_ut, F_gam, ct, omega_hat = mt.JCItestpara_spilCt(rowAS, opt_model, p - 1)

        CW = F_b / np.sum(np.absolute(F_b))

        Johansen_intcept, Johansen_slope = mt.Johansen_mean(F_a, F_b, F_gam, F_ct, p - 1)
        Johansen_var_correct = mt.Johansen_std_correct(F_a, F_b, F_ut, F_gam, p - 1)

        if Johansen_var_correct < 0:
            return None  # Johansen_var小於0，捨棄該配對
        Johansen_std = np.sqrt(Johansen_var_correct)

        return {
            'S1': s1_name,
            'S2': s2_name,
            'VECM(q)': p - 1,
            'Johansen_intercept': Johansen_intcept[0, 0],
            'Johansen_slope': Johansen_slope[0, 0],
            'Johansen_std': Johansen_std[0, 0],
            'VECM_Model_Type': opt_model,
            'w1': CW[0, 0],
            'w2': CW[1, 0],
            'Del_min': form_del_min,
        }
    except Exception as e:
        # 單一配對計算失敗（例如數值不穩定）不中斷整體流程，當作捨棄該配對處理，
        # 但印出來方便事後排查是不是特定配對有問題。
        print(f"  [配對失敗] {s1_name}-{s2_name}：{repr(e)}")
        return None


def process_batch(batch):
    return [process_one_pair(task) for task in batch]


def save_file_result(input_filename, rows):
    df = pd.DataFrame(rows, columns=TABLE_COLUMNS) if rows else pd.DataFrame(columns=TABLE_COLUMNS)
    full_path = os.path.join(save_path, formation_table_filename(input_filename))
    # 先寫暫存檔再原子性覆蓋（os.replace），避免程式中途被中斷時留下寫一半的檔案
    # ——resume邏輯是靠「輸出檔案是否存在」判斷該交易日是否已完成，寫一半的檔案
    # 會被誤判為「已完成」，之後就再也不會被補算。
    tmp_path = full_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, full_path)


def process_one_file(input_filename, executor):
    """
    處理單一交易日：讀檔→展開配對→分批送進(已存在的)executor→收集結果→存檔。
    只在單一檔案的範圍內建立task list，記憶體用量跟檔案數無關（不會因為
    data/full_data_AB底下有上千個交易日就爆記憶體）。
    """
    Smin = pd.read_csv(dat_path + input_filename)
    maxcompanynu = Smin.shape[1]
    col_name = {i: name for i, name in enumerate(Smin.columns.values)}
    LSmin = np.log(Smin.iloc[form_del_min:, :])
    LSmin = LSmin.reset_index(drop=True)

    combs = mt.Binal_comb(range(maxcompanynu))
    tasks = []
    for c1, c2 in combs:
        rowS = LSmin.iloc[0:inNum, [int(c1), int(c2)]]
        rowAS = np.array(rowS)
        tasks.append((col_name[int(c1)], col_name[int(c2)], rowAS))

    batches = list(chunk_list(tasks, CHUNK_SIZE))
    rows = []

    # Colab用 !python script.py 執行時，捕捉stdout沒辦法用\r覆蓋同一行，
    # 逐批印進度會讓每個交易日洗出一大串行。改成完全不印中途進度，
    # 每個交易日只在main()裡的[存檔]那一行輸出一次結果。
    futures = [executor.submit(process_batch, batch) for batch in batches]
    for future in as_completed(futures):
        try:
            batch_results = future.result()
        except Exception as e:
            print(f"  [批次失敗] {input_filename}：{repr(e)}")
            batch_results = []
        for row in batch_results:
            if row is not None:
                rows.append(row)

    save_file_result(input_filename, rows)
    return len(rows)


def main():
    os.makedirs(save_path, exist_ok=True)
    data_list = sorted(os.listdir(dat_path))

    # ── 中斷接續：輸出檔已存在的交易日直接跳過，不重新讀檔/計算 ──
    todo_list = []
    skipped = 0
    for filename in data_list:
        out_path = os.path.join(save_path, formation_table_filename(filename))
        if os.path.exists(out_path):
            skipped += 1
        else:
            todo_list.append(filename)

    print(f"共 {len(data_list):,} 個交易日，已完成（跳過）{skipped:,} 個，待處理 {len(todo_list):,} 個")

    if not todo_list:
        print("全部交易日都已經處理完成，沒有需要跑的工作。")
        return

    # 這台機器 nproc / os.cpu_count() 皆回報 8，代表是真實可用核心數，
    # 保留1個給主行程做結果彙整跟I/O，避免全部搶滿。
    max_workers = max(1, (os.cpu_count() or 8) - 1)
    print(f"用 {max_workers} 個worker平行處理，每批最多 {CHUNK_SIZE} 個配對，"
          f"同一個worker pool會沿用到所有交易日（不會每個檔案重開一次）")

    startime = time.time()
    # 整個run共用同一個ProcessPoolExecutor，避免每個交易日都重新啟動worker的開銷。
    # 外層保留tqdm（整體進度，看得到ETA/累計耗時），只有內層每個檔案內部的
    # 逐配對進度拿掉——那個才是洗版的來源（每15秒還是會累積出很多行）。
    # 外層一個檔案只會refresh一次，不會有同樣的洗版問題。
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for file_i, filename in enumerate(tqdm(todo_list, desc="整體進度", unit="file")):
            n_kept = process_one_file(filename, executor)
            elapsed = int(time.time() - startime)
            tqdm.write(
                f"[存檔] {filename[:10]} 完成（{file_i + 1}/{len(todo_list)}），"
                f"保留 {n_kept} 筆配對，累計耗時 {elapsed} 秒"
            )

    print(f"\n全部完成，總耗時 {int(time.time() - startime)} 秒")


if __name__ == "__main__":
    # concurrent.futures.ProcessPoolExecutor 本身跨平台，不是Windows專屬——
    # Windows預設用spawn（子行程重新import整支檔案），Colab/Linux預設用fork
    # （子行程直接複製已執行完的parent記憶體，不會重新跑一次頂層程式碼）。
    # 這層 if __name__ == "__main__" 保護在spawn上是必要的（沒有會無窮遞迴
    # 啟動子行程），在fork上雖非必要但仍是跨平台安全的標準寫法，予以保留。
    main()
