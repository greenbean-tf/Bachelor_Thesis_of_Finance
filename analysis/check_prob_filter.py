"""
步驟3：機率篩選驗證

驗證 open_prob_threshold=0.65 是否合理：被篩掉的配對（revert_probability < 0.65,
note == "Non_Open"）有沒有可能其實大多會反轉/獲利？

關鍵：records_df 對「每一個配對」都存了 ground truth 的 Revert 欄位（這個配對的
價差最終是否真的反轉回均值），這個欄位跟是否真的開倉無關——即使 Non_Open（沒開倉）
的配對，仍然知道它「事後」到底有沒有反轉。這讓我們可以不需要重跑 backtest，直接
檢查模型預測的 Revert_Prob 跟真實 Revert 結果的校準關係（在完整機率範圍，包含
< 0.65 被篩掉的那一段）。

注意：Revert=True 只代表「會反轉回均值」，不直接等於「會獲利」（還要看 rtop 夠不
夠大去打過交易成本），所以這是必要不充分條件的檢查，但已經是不重跑 backtest 情況
下能做的最直接驗證。
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL_DIR = os.path.join(ROOT, "data", "Record", "2026-06-21 14-31-57",
                       "Test_sl_sigma=0.999_Loss=GaussCopGumLoss")
OUT_DIR = os.path.join(ROOT, "analysis", "output")

COLS = ["Revert_Prob", "Revert", "note", "Backtest_Profit", "Norm_Rtop"]
OPEN_PROB_THRESHOLD = 0.65


def load_cols(record_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(record_dir, "record_*.csv")))
    dfs = [pd.read_csv(f, usecols=COLS) for f in files]
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("讀取資料中（只取必要欄位）...")
    df = load_cols(DL_DIR)
    print(f"總筆數: {len(df):,}")

    non_open = df[df["note"] == "Non_Open"]
    opened_or_other = df[df["note"] != "Non_Open"]

    print(f"\nNon_Open 配對數 (revert_probability < {OPEN_PROB_THRESHOLD}): {len(non_open):,} "
          f"({len(non_open)/len(df)*100:.2f}%)")

    # --- 核心檢查：被篩掉的配對，事後真實 Revert 率 ---
    revert_rate_non_open = non_open["Revert"].mean()
    revert_rate_overall   = df["Revert"].mean()
    revert_rate_others    = opened_or_other["Revert"].mean()

    print(f"\n真實 Revert 率（ground truth，跟是否開倉無關）：")
    print(f"  全部配對             : {revert_rate_overall*100:.2f}%")
    print(f"  Non_Open 被篩掉的配對 : {revert_rate_non_open*100:.2f}%")
    print(f"  其他（機率>=0.65）配對: {revert_rate_others*100:.2f}%")

    # --- 完整機率區間的校準曲線（含 <0.65 被篩掉的部分） ---
    print(f"\n--- Revert_Prob 完整分布的校準曲線（按機率分箱，看真實 Revert 率） ---")
    bins = np.arange(0, 1.05, 0.05)
    df["prob_bin"] = pd.cut(df["Revert_Prob"], bins=bins, include_lowest=True)
    calib = df.groupby("prob_bin", observed=True).agg(
        n=("Revert", "size"),
        true_revert_rate=("Revert", "mean"),
    )
    print(calib.to_string())

    # --- 如果被篩掉的配對其實有開倉，事後反轉的那些筆，rtop 夠不夠大去打過成本？ ---
    # Norm_Rtop = -1 表示沒有反轉(sentinel)；>0 表示有反轉，數值是反轉前的峰值偏離(已正規化)
    non_open_reverted = non_open[non_open["Revert"] == True]
    print(f"\nNon_Open 裡「事後真的反轉」的配對 (n={len(non_open_reverted):,})：")
    print(f"  Norm_Rtop 統計 (這是它們本可以開倉賺到的價差大小，正規化後)：")
    print(non_open_reverted["Norm_Rtop"].describe().to_string())
    print(f"\n  Norm_Rtop 百分位數：")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}%: {np.percentile(non_open_reverted['Norm_Rtop'], p):.4f}")

    # 跟全部配對裡「有反轉」的 Norm_Rtop 比較（看分布形狀是否類似/偏小）
    all_reverted = df[df["Revert"] == True]
    print(f"\n全部配對裡「有反轉」的 Norm_Rtop 中位數（對照組, n={len(all_reverted):,}）："
          f"{all_reverted['Norm_Rtop'].median():.4f}")

    # 存檔
    calib.to_csv(os.path.join(OUT_DIR, "step3_prob_calibration.csv"), encoding="utf-8-sig")
    summary = pd.DataFrame({
        "metric": ["revert_rate_overall", "revert_rate_non_open", "revert_rate_others",
                   "non_open_count", "non_open_pct"],
        "value": [revert_rate_overall, revert_rate_non_open, revert_rate_others,
                  len(non_open), len(non_open)/len(df)],
    })
    summary.to_csv(os.path.join(OUT_DIR, "step3_prob_filter_summary.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"\n[已存檔] {OUT_DIR}/step3_prob_calibration.csv")
    print(f"[已存檔] {OUT_DIR}/step3_prob_filter_summary.csv")
