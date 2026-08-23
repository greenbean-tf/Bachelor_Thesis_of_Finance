"""
診斷腳本：regime shift 假設驗證 (步驟1) + T_o 預測分布檢查 (步驟2)

純本機分析，讀取已存在的逐日逐配對交易記錄 (record_YYYYMMDD.csv)，
不需要重跑 GPU/CPU backtest。對應 diagnosis_plan.md 步驟1、步驟2。

資料來源（已改用 BestEpoch=24 修正後的正確 checkpoint 結果，2026-07-03）：
- DL model : data/Record/2026-07-03 07-10-12/Test_sl_sigma=0.999_Loss=GaussCopGumLoss/
- Baseline : data/Record/2026-06-21 08-42-06/Test_Baseline_1.5sigma/（不受 checkpoint 影響，沿用原資料）
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL_DIR = os.path.join(ROOT, "data", "Record", "2026-07-03 07-10-12",
                       "Test_sl_sigma=0.999_Loss=GaussCopGumLoss")
BASELINE_DIR = os.path.join(ROOT, "data", "Record", "2026-06-21 08-42-06",
                             "Test_Baseline_1.5sigma")
OUT_DIR = os.path.join(ROOT, "analysis", "output_fixed_ckpt")

SPLIT_DATE = 20221019  # regime split 分界點

NOTES_SET = ["Non_Open", "Discard_NegExp", "Below_Tax",
             "Normal_Close", "Stop_Loss", "Exit", "Above"]


def load_records(record_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(record_dir, "record_*.csv")))
    if not files:
        raise FileNotFoundError(f"No record_*.csv found in {record_dir}")
    dfs = [pd.read_csv(f, index_col=0) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """複刻 backtest.py: Backtest.summary_record() 的計算邏輯"""
    records_size = len(df)
    note = df["note"]

    non_open_case      = df[note == NOTES_SET[0]]
    discard_case        = df[note == NOTES_SET[1]]
    below_tax_case       = df[note == NOTES_SET[2]]
    normal_close_case    = df[note == NOTES_SET[3]]
    stop_loss_case       = df[note == NOTES_SET[4]]
    force_close_case     = df[note == NOTES_SET[5]]
    above_top_case        = df[note == NOTES_SET[6]]

    win_case  = df[df["Backtest_Profit"] > 0]
    lose_case = df[df["Backtest_Profit"] < 0]
    tie_case  = df[df["Backtest_Profit"] == 0]

    open_count = len(normal_close_case) + len(stop_loss_case) + len(force_close_case)

    s = {}
    s["records_size"] = records_size
    s["Total open count"] = open_count
    s["Win per open"]  = len(win_case) / open_count if open_count else np.nan
    s["Lose per open"] = len(lose_case) / open_count if open_count else np.nan
    s["Win per record"]  = len(win_case) / records_size if records_size else np.nan
    s["Lose per record"] = len(lose_case) / records_size if records_size else np.nan
    s["Tie per record"]  = len(tie_case) / records_size if records_size else np.nan

    s["Non Open per record"]  = len(non_open_case) / records_size if records_size else np.nan
    s["Discard per record"]   = len(discard_case) / records_size if records_size else np.nan
    s["Below Tax per record"] = len(below_tax_case) / records_size if records_size else np.nan
    s["Above Top per record"] = len(above_top_case) / records_size if records_size else np.nan

    s["Normal Close per open"] = len(normal_close_case) / open_count if open_count else np.nan
    s["Stop Loss per open"]    = len(stop_loss_case) / open_count if open_count else np.nan
    s["Force Close per open"]  = len(force_close_case) / open_count if open_count else np.nan

    s["Earn amount"]  = win_case["Backtest_Profit"].sum()
    s["Loss amount"] = lose_case["Backtest_Profit"].sum()
    s["Total P&L"]    = df["Backtest_Profit"].sum()
    s["Profit per open"] = s["Total P&L"] / open_count if open_count else np.nan
    s["True earn amount per win count"]  = win_case["Backtest_Profit"].mean() if len(win_case) else np.nan
    s["True Lose amount per lose count"] = lose_case["Backtest_Profit"].mean() if len(lose_case) else np.nan

    return s


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def compare_table(results: dict, keys_order):
    """results: { label: summary_dict }"""
    rows = ["Total open count", "Win per open", "Lose per open",
            "Normal Close per open", "Stop Loss per open", "Force Close per open",
            "Non Open per record", "Discard per record", "Below Tax per record", "Above Top per record",
            "Earn amount", "Loss amount", "Total P&L", "Profit per open",
            "True earn amount per win count", "True Lose amount per lose count"]
    header = f"{'Metric':<32}" + "".join(f"{k:>20}" for k in keys_order)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r:<32}"
        for k in keys_order:
            v = results[k].get(r, np.nan)
            if isinstance(v, float):
                line += f"{v:>20,.4f}"
            else:
                line += f"{v:>20,}"
        print(line)


def step1_regime_split(dl_df: pd.DataFrame, baseline_df: pd.DataFrame):
    print_section(f"步驟1：Regime Split 驗證 (分界日: {SPLIT_DATE})")

    dl_front = dl_df[dl_df["Date"] < SPLIT_DATE]
    dl_back  = dl_df[dl_df["Date"] >= SPLIT_DATE]
    bl_front = baseline_df[baseline_df["Date"] < SPLIT_DATE]
    bl_back  = baseline_df[baseline_df["Date"] >= SPLIT_DATE]

    print(f"\nDL model   前段日期範圍: {dl_front['Date'].min()} ~ {dl_front['Date'].max()}  "
          f"(共 {dl_front['Date'].nunique()} 個交易日, {len(dl_front):,} 筆配對記錄)")
    print(f"DL model   後段日期範圍: {dl_back['Date'].min()} ~ {dl_back['Date'].max()}  "
          f"(共 {dl_back['Date'].nunique()} 個交易日, {len(dl_back):,} 筆配對記錄)")
    print(f"Baseline   前段日期範圍: {bl_front['Date'].min()} ~ {bl_front['Date'].max()}  "
          f"(共 {bl_front['Date'].nunique()} 個交易日, {len(bl_front):,} 筆配對記錄)")
    print(f"Baseline   後段日期範圍: {bl_back['Date'].min()} ~ {bl_back['Date'].max()}  "
          f"(共 {bl_back['Date'].nunique()} 個交易日, {len(bl_back):,} 筆配對記錄)")

    results = {
        "DL-全期":     summarize(dl_df),
        "DL-前段":     summarize(dl_front),
        "DL-後段":     summarize(dl_back),
        "Baseline-全期": summarize(baseline_df),
        "Baseline-前段": summarize(bl_front),
        "Baseline-後段": summarize(bl_back),
    }

    print()
    compare_table(results, ["DL-全期", "Baseline-全期"])
    print()
    compare_table(results, ["DL-前段", "Baseline-前段"])
    print()
    compare_table(results, ["DL-後段", "Baseline-後段"])

    # 額外：DL 跟 Baseline 在各段的 Total P&L 差距（DL - Baseline）
    print_section("步驟1 摘要：DL 對 Baseline 的 Total P&L 差距")
    diff_full  = results["DL-全期"]["Total P&L"] - results["Baseline-全期"]["Total P&L"]
    diff_front = results["DL-前段"]["Total P&L"] - results["Baseline-前段"]["Total P&L"]
    diff_back  = results["DL-後段"]["Total P&L"] - results["Baseline-後段"]["Total P&L"]
    print(f"全期 : DL - Baseline = {diff_full:>15,.2f}   "
          f"({'DL贏' if diff_full > 0 else 'Baseline贏'})")
    print(f"前段 : DL - Baseline = {diff_front:>15,.2f}   "
          f"({'DL贏' if diff_front > 0 else 'Baseline贏'})")
    print(f"後段 : DL - Baseline = {diff_back:>15,.2f}   "
          f"({'DL贏' if diff_back > 0 else 'Baseline贏'})")

    # Profit per open 差距（排除交易量差異的影響，看單筆品質）
    print()
    ppo_full  = results["DL-全期"]["Profit per open"] - results["Baseline-全期"]["Profit per open"]
    ppo_front = results["DL-前段"]["Profit per open"] - results["Baseline-前段"]["Profit per open"]
    ppo_back  = results["DL-後段"]["Profit per open"] - results["Baseline-後段"]["Profit per open"]
    print(f"全期 : DL - Baseline (Profit per open) = {ppo_full:>10,.4f}")
    print(f"前段 : DL - Baseline (Profit per open) = {ppo_front:>10,.4f}")
    print(f"後段 : DL - Baseline (Profit per open) = {ppo_back:>10,.4f}")

    # 存成 CSV
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "step1_regime_split_comparison.csv")
    out_df.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n[已存檔] {out_path}")

    return results


def step2_to_distribution(dl_df: pd.DataFrame):
    print_section("步驟2：DL model T_o (Final_Open_Threshold) 預測分布檢查")

    opened = dl_df[dl_df["Final_Open_Threshold"] > 0]
    to_vals = opened["Final_Open_Threshold"].values

    print(f"\n真正開倉的配對數: {len(opened):,} / 全部記錄數: {len(dl_df):,} "
          f"({len(opened)/len(dl_df)*100:.2f}%)")

    stats = {
        "mean":   np.mean(to_vals),
        "std":    np.std(to_vals),
        "min":    np.min(to_vals),
        "max":    np.max(to_vals),
        "median (50%)": np.percentile(to_vals, 50),
        "10%":    np.percentile(to_vals, 10),
        "25%":    np.percentile(to_vals, 25),
        "75%":    np.percentile(to_vals, 75),
        "90%":    np.percentile(to_vals, 90),
        "95%":    np.percentile(to_vals, 95),
        "99%":    np.percentile(to_vals, 99),
    }
    print("\nFinal_Open_Threshold 統計量：")
    for k, v in stats.items():
        print(f"  {k:<15}: {v:,.4f}")

    pct_below_1_5 = np.mean(to_vals < 1.5) * 100
    pct_below_median_vs_baseline = np.mean(to_vals < 1.0) * 100
    print(f"\n  比例 < 1.5 (baseline 固定門檻) : {pct_below_1_5:.2f}%")
    print(f"  比例 < 1.0                     : {pct_below_median_vs_baseline:.2f}%")
    print(f"  比例 < median({stats['median (50%)']:.3f})        : 50.00% (定義)")

    # 判讀
    median_val = stats["median (50%)"]
    print("\n[判讀]")
    if median_val < 1.5:
        print(f"  median ({median_val:.4f}) < 1.5 → 支持「T_o 系統性偏保守」假設")
    else:
        print(f"  median ({median_val:.4f}) >= 1.5 → 不支持「T_o 系統性偏保守」假設，"
              f"需要從其他角度（機率篩選/停損/樣本選擇偏誤）解釋「贏小錢」現象")

    # 畫圖：完整分布 (log-x，因為右偏 + 有極端值)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(to_vals, bins=200, range=(0, 10), edgecolor="black", alpha=0.7)
    axes[0].axvline(1.5, color="red", linestyle="--", linewidth=2, label="baseline 1.5")
    axes[0].axvline(median_val, color="blue", linestyle="-", linewidth=2,
                     label=f"median={median_val:.3f}")
    axes[0].axvline(stats["mean"], color="green", linestyle=":", linewidth=2,
                     label=f"mean={stats['mean']:.3f}")
    axes[0].set_title("Final_Open_Threshold Distribution (zoom: 0~10)")
    axes[0].set_xlabel("Final_Open_Threshold")
    axes[0].legend()

    axes[1].hist(np.log10(to_vals), bins=100, edgecolor="black", alpha=0.7)
    axes[1].axvline(np.log10(1.5), color="red", linestyle="--", linewidth=2, label="baseline 1.5")
    axes[1].axvline(np.log10(median_val), color="blue", linestyle="-", linewidth=2,
                     label=f"median={median_val:.3f}")
    axes[1].set_title("Final_Open_Threshold Distribution (log10 scale, 全範圍)")
    axes[1].set_xlabel("log10(Final_Open_Threshold)")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "step2_To_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[已存檔] {out_path}")

    # 存統計量成 CSV
    stats_df = pd.DataFrame(list(stats.items()), columns=["stat", "value"])
    stats_df.loc[len(stats_df)] = ["pct_below_1.5", pct_below_1_5]
    out_path = os.path.join(OUT_DIR, "step2_To_distribution_stats.csv")
    stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[已存檔] {out_path}")

    return stats


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print_section("讀取資料中...")
    dl_df = load_records(DL_DIR)
    baseline_df = load_records(BASELINE_DIR)
    print(f"DL model   : {len(dl_df):,} 筆記錄 (來自 {DL_DIR})")
    print(f"Baseline   : {len(baseline_df):,} 筆記錄 (來自 {BASELINE_DIR})")

    step1_regime_split(dl_df, baseline_df)
    step2_to_distribution(dl_df)

    print_section("分析完成")
