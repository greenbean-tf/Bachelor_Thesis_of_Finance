"""
步驟3 (must_open=True 重跑版)：驗證機率篩選掉的配對「真實獲利」分布

前提：必須先在 Colab 用 must_open=True 重跑 backtest_from_saved，產生新的
record_*.csv（這次原本 Revert_Prob<0.65 的配對也會被真的丟進交易模擬，
Backtest_Profit 才有意義）。

用法：把 NEW_RECORD_DIR 改成新跑出來的資料夾路徑（看 main.py 印出的
experiment_file 時間戳，或直接看 data/Record/ 底下最新的資料夾）。
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "analysis", "output")

NEW_RECORD_DIR = os.path.join(ROOT, "data", "Record", "2026-06-26 06-30-05",
                                "Test_sl_sigma=0.999_Loss=GaussCopGumLoss")

OLD_PROB_THRESHOLD = 0.65  # 原本的篩選門檻，用來標記「原本會被擋掉」的配對

COLS = ["Revert_Prob", "Revert", "note", "Backtest_Profit", "Norm_Rtop",
        "Final_Open_Threshold"]

NOTES_SET = ["Non_Open", "Discard_NegExp", "Below_Tax",
             "Normal_Close", "Stop_Loss", "Exit", "Above"]


def load_cols(record_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(record_dir, "record_*.csv")))
    if not files:
        raise FileNotFoundError(f"No record_*.csv found in {record_dir}")
    dfs = [pd.read_csv(f, usecols=COLS) for f in files]
    return pd.concat(dfs, ignore_index=True)


def profit_by_bin(df: pd.DataFrame, bins) -> pd.DataFrame:
    df = df.copy()
    df["prob_bin"] = pd.cut(df["Revert_Prob"], bins=bins, include_lowest=True)

    open_mask = df["note"].isin(["Normal_Close", "Stop_Loss", "Exit"])

    def agg_fn(g):
        opened = g[open_mask.loc[g.index]]
        win = opened[opened["Backtest_Profit"] > 0]
        lose = opened[opened["Backtest_Profit"] < 0]
        return pd.Series({
            "n_total": len(g),
            "n_opened": len(opened),
            "true_revert_rate": g["Revert"].mean(),
            "win_rate_per_open": len(win) / len(opened) if len(opened) else np.nan,
            "total_profit": opened["Backtest_Profit"].sum(),
            "profit_per_open": opened["Backtest_Profit"].sum() / len(opened) if len(opened) else np.nan,
            "mean_win_amount": win["Backtest_Profit"].mean() if len(win) else np.nan,
            "mean_lose_amount": lose["Backtest_Profit"].mean() if len(lose) else np.nan,
        })

    return df.groupby("prob_bin", observed=True).apply(agg_fn, include_groups=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"讀取資料中: {NEW_RECORD_DIR}")
    df = load_cols(NEW_RECORD_DIR)
    print(f"總筆數: {len(df):,}")

    # 細分箱：0.05 一格，方便看 0.65 前後的變化
    bins = np.arange(0, 1.05, 0.05)
    result = profit_by_bin(df, bins)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== 按 Revert_Prob 級距 (0.05 一格) 的完整獲利數據 ===")
    print(result.to_string())

    # 核心比較：原本會被 0.65 篩掉 vs 保留的兩組，加總真實獲利
    would_be_filtered = df[df["Revert_Prob"] < OLD_PROB_THRESHOLD]
    would_be_kept = df[df["Revert_Prob"] >= OLD_PROB_THRESHOLD]

    open_mask_all = df["note"].isin(["Normal_Close", "Stop_Loss", "Exit"])
    wf_opened = would_be_filtered[open_mask_all.loc[would_be_filtered.index]]
    wk_opened = would_be_kept[open_mask_all.loc[would_be_kept.index]]

    print(f"\n=== 核心結論：原本被 0.65 篩掉的配對，真實淨損益 ===")
    print(f"原本會被篩掉的配對數: {len(would_be_filtered):,}  "
          f"(真的開倉: {len(wf_opened):,})")
    print(f"  Total P&L (這群配對的真實淨損益總和) : {wf_opened['Backtest_Profit'].sum():,.2f}")
    print(f"  Profit per open                       : "
          f"{wf_opened['Backtest_Profit'].sum()/len(wf_opened) if len(wf_opened) else float('nan'):,.4f}")
    print(f"  Win rate per open                      : "
          f"{(wf_opened['Backtest_Profit']>0).mean()*100:.2f}%")

    print(f"\n原本會保留的配對數: {len(would_be_kept):,}  (真的開倉: {len(wk_opened):,})")
    print(f"  Total P&L                             : {wk_opened['Backtest_Profit'].sum():,.2f}")
    print(f"  Profit per open                       : "
          f"{wk_opened['Backtest_Profit'].sum()/len(wk_opened) if len(wk_opened) else float('nan'):,.4f}")

    print(f"\n全部配對 (must_open=True 全開放) Total P&L: {df.loc[open_mask_all, 'Backtest_Profit'].sum():,.2f}")
    print(f"(對照：原本 open_prob_threshold=0.65 跑出來的 Total P&L = 385,435.52)")

    print(f"\n[判讀]")
    wf_total = wf_opened['Backtest_Profit'].sum()
    if wf_total > 0:
        print(f"  被篩掉的配對真實淨損益為正 (+{wf_total:,.2f}) → 0.65 門檻確實濾掉了真實利潤，")
        print(f"  建議調低 open_prob_threshold")
    else:
        print(f"  被篩掉的配對真實淨損益為負 ({wf_total:,.2f}) → 0.65 門檻是對的，維持現狀")

    result.to_csv(os.path.join(OUT_DIR, "step3_profit_by_bin.csv"), encoding="utf-8-sig")
    print(f"\n[已存檔] {OUT_DIR}/step3_profit_by_bin.csv")
