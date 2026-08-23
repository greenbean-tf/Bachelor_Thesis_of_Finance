"""
驗證 grid search 上限被鎖死問題：確認 predict_rtop (records_df 的 'rtop' 欄位)
的實際數值分布，特別是 median/percentile，看是否普遍偏低導致
find_opt_open_threshold() 的搜尋上限 [0, predict_rtop] 結構性地限制了 T_o。

只讀取需要的欄位 (rtop, top, Final_Open_Threshold, note) 加速讀取。
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL_DIR = os.path.join(ROOT, "data", "Record", "2026-06-21 14-31-57",
                       "Test_sl_sigma=0.999_Loss=GaussCopGumLoss")
OUT_DIR = os.path.join(ROOT, "analysis", "output")

COLS = ["rtop", "top", "Final_Open_Threshold", "note"]


def load_cols(record_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(record_dir, "record_*.csv")))
    if not files:
        raise FileNotFoundError(f"No record_*.csv found in {record_dir}")
    dfs = [pd.read_csv(f, usecols=COLS) for f in files]
    return pd.concat(dfs, ignore_index=True)


def percentile_report(name, vals):
    vals = np.asarray(vals)
    print(f"\n--- {name} (n={len(vals):,}) ---")
    stats = {
        "mean": np.mean(vals), "std": np.std(vals),
        "min": np.min(vals), "max": np.max(vals),
        "1%": np.percentile(vals, 1), "5%": np.percentile(vals, 5),
        "10%": np.percentile(vals, 10), "25%": np.percentile(vals, 25),
        "median(50%)": np.percentile(vals, 50),
        "75%": np.percentile(vals, 75), "90%": np.percentile(vals, 90),
        "95%": np.percentile(vals, 95), "99%": np.percentile(vals, 99),
    }
    for k, v in stats.items():
        print(f"  {k:<14}: {v:,.4f}")
    return stats


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("讀取資料中（只取必要欄位，加速讀取）...")
    df = load_cols(DL_DIR)
    print(f"總筆數: {len(df):,}")

    # 全部配對（不管是否開倉）的 predict_rtop 分布
    rtop_all_stats = percentile_report("predict_rtop（全部配對）", df["rtop"].values)
    top_all_stats = percentile_report("predict_top（全部配對）", df["top"].values)

    # 只看真正開倉的配對（Final_Open_Threshold > 0），這才是真正餵進 grid search 的母體
    opened = df[df["Final_Open_Threshold"] > 0]
    rtop_opened_stats = percentile_report("predict_rtop（僅開倉配對）", opened["rtop"].values)
    top_opened_stats = percentile_report("predict_top（僅開倉配對）", opened["top"].values)

    # 直接比較：對開倉配對，搜尋上限(predict_rtop) vs 實際選到的門檻(Final_Open_Threshold)
    # 看門檻被卡在搜尋上限附近的比例（搜尋範圍 [0, rtop]，若 To 非常接近 rtop，代表
    # 很可能是被上限卡住，而非真正在範圍內找到的內部最優解）
    ratio = opened["Final_Open_Threshold"] / opened["rtop"]
    print(f"\n--- Final_Open_Threshold / predict_rtop 比例 (僅開倉配對, n={len(opened):,}) ---")
    print(f"  median ratio : {np.median(ratio):.4f}")
    print(f"  >= 0.95 的比例 (門檻幾乎貼到搜尋上限) : {np.mean(ratio >= 0.95)*100:.2f}%")
    print(f"  >= 0.99 的比例 (門檻幾乎等於搜尋上限) : {np.mean(ratio >= 0.99)*100:.2f}%")

    # top 比 rtop 大多少？驗證「改用 top 當上限」這個修正方向的合理性
    top_rtop_ratio = df["top"] / df["rtop"]
    print(f"\n--- predict_top / predict_rtop 比例 (全部配對, n={len(df):,}) ---")
    print(f"  median ratio : {np.median(top_rtop_ratio):.4f}")
    print(f"  mean ratio   : {np.mean(top_rtop_ratio):.4f}")
    print(f"  比例 > 1 (top > rtop) 的配對佔比 : {np.mean(top_rtop_ratio > 1)*100:.2f}%")

    # 存檔
    summary = pd.DataFrame({
        "rtop_all": rtop_all_stats, "top_all": top_all_stats,
        "rtop_opened": rtop_opened_stats, "top_opened": top_opened_stats,
    })
    out_path = os.path.join(OUT_DIR, "step6_rtop_distribution_stats.csv")
    summary.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n[已存檔] {out_path}")
