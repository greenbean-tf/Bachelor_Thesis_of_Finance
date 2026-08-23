# -*- coding: utf-8 -*-
"""
驗證 create_formationtable.py（非平行）與 multiprocessing/Main_Check_parallel.py（平行）
在 Colab 上跑出來的結果，是否跟 CheckTable2（原本在本機 Windows 上跑出來的參考結果）一致。

三方比較：
  reference   : CheckTable2/{date}_CheckTable.csv        （本機 Windows 版）
  non_parallel: check_table/{date}_CheckTable.csv         （Colab 非平行版）
  parallel    : check_table/{date}_CheckTable_M.csv       （Colab 平行版）

比對邏輯：用 (S1, S2) 當 key 對齊兩張表，數值欄位用容忍誤差比較
（跨平台 BLAS 實作不同，浮點數末位可能有極小差異，不代表計算錯誤），
VECM_q / Model / Del_min 這幾個整數欄位用完全相等比較。
"""
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = "/content/GGWP_US_LLTL/create_formationtable"
REF_DIR = os.path.join(BASE_DIR, "CheckTable2")
OUT_DIR = os.path.join(BASE_DIR, "check_table")

NUMERIC_TOL_COLS = ["Johansen intercept", "Johansen slope", "Johansen std", "W1", "W2"]
EXACT_COLS = ["VECM_q", "Model", "Del_min"]
RTOL, ATOL = 1e-6, 1e-9


def load_table(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["S1"] = df["S1"].astype(str)
    df["S2"] = df["S2"].astype(str)
    df = df.set_index(["S1", "S2"])
    return df


def compare_pair(name_a, df_a, name_b, df_b):
    """比較兩張表（同一天），回傳這一天是否完全一致，以及不一致細節的文字列表。"""
    issues = []
    if df_a is None or df_b is None:
        missing = name_a if df_a is None else name_b
        issues.append(f"  [跳過] {missing} 找不到檔案")
        return None, issues

    keys_a, keys_b = set(df_a.index), set(df_b.index)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        issues.append(f"  只在 {name_a} 出現的配對數：{len(only_a)}（例如 {list(only_a)[:3]}）")
    if only_b:
        issues.append(f"  只在 {name_b} 出現的配對數：{len(only_b)}（例如 {list(only_b)[:3]}）")

    common = list(keys_a & keys_b)
    if not common:
        issues.append("  [警告] 沒有共同配對可以比對")
        return False, issues

    sub_a = df_a.loc[common]
    sub_b = df_b.loc[common]

    all_ok = not only_a and not only_b
    for col in EXACT_COLS:
        mismatch = sub_a[col].astype(int) != sub_b[col].astype(int)
        if mismatch.any():
            all_ok = False
            bad_keys = [common[i] for i in np.where(mismatch)[0]][:3]
            issues.append(f"  欄位 [{col}] 有 {mismatch.sum()} 筆配對數值不同，例如 {bad_keys}")

    for col in NUMERIC_TOL_COLS:
        a_vals = sub_a[col].astype(float).values
        b_vals = sub_b[col].astype(float).values
        close = np.isclose(a_vals, b_vals, rtol=RTOL, atol=ATOL)
        if not close.all():
            all_ok = False
            bad_idx = np.where(~close)[0]
            max_diff = np.max(np.abs(a_vals[bad_idx] - b_vals[bad_idx]))
            bad_keys = [common[i] for i in bad_idx][:3]
            issues.append(
                f"  欄位 [{col}] 有 {len(bad_idx)} 筆超出容忍誤差"
                f"（最大差異 {max_diff:.3e}），例如 {bad_keys}"
            )

    return all_ok, issues


def main():
    if not os.path.isdir(REF_DIR):
        print(f"找不到參考資料夾：{REF_DIR}")
        sys.exit(1)

    dates = sorted(
        f[: -len("_CheckTable.csv")]
        for f in os.listdir(REF_DIR)
        if f.endswith("_CheckTable.csv")
    )
    print(f"共 {len(dates)} 個日期要驗證：{dates}\n")

    overall_np_ok = True
    overall_mp_ok = True

    for date in dates:
        ref_path = os.path.join(REF_DIR, f"{date}_CheckTable.csv")
        np_path = os.path.join(OUT_DIR, f"{date}_CheckTable.csv")
        mp_path = os.path.join(OUT_DIR, f"{date}_CheckTable_M.csv")

        ref_df = load_table(ref_path)
        np_df = load_table(np_path)
        mp_df = load_table(mp_path)

        print(f"=== {date} ===")

        np_ok, np_issues = compare_pair("CheckTable2(參考)", ref_df, "check_table非平行版", np_df)
        if np_ok is None:
            overall_np_ok = False
        elif np_ok:
            print("  [OK] 非平行版 與 參考結果一致")
        else:
            overall_np_ok = False
            print("  [FAIL] 非平行版 與 參考結果不一致：")
        for line in np_issues:
            print(line)

        mp_ok, mp_issues = compare_pair("CheckTable2(參考)", ref_df, "check_table平行版", mp_df)
        if mp_ok is None:
            overall_mp_ok = False
        elif mp_ok:
            print("  [OK] 平行版 與 參考結果一致")
        else:
            overall_mp_ok = False
            print("  [FAIL] 平行版 與 參考結果不一致：")
        for line in mp_issues:
            print(line)

        print()

    print("=" * 50)
    print(f"非平行版整體結果：{'全部一致 ✅' if overall_np_ok else '有不一致，見上方 FAIL ❌'}")
    print(f"平行版整體結果  ：{'全部一致 ✅' if overall_mp_ok else '有不一致，見上方 FAIL ❌'}")


if __name__ == "__main__":
    main()
