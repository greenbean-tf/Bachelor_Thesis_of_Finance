# -*- coding: utf-8 -*-
"""
檢查 data/full_data_AB 底下所有交易日檔案的資料完整度。

背景：Main_Check_parallel.py 產生formation table時，process_one_file()對整個
檔案做 np.log(Smin.iloc[form_del_min:,:]) 完全沒有檢查0值/NaN，只要某股票在
formation window內有一個0或NaN，log後就會變成-inf/NaN，毒害這檔股票牽涉到的
「每一個」配對——這正是實際發生過的VICI.N/TSN.N大量[配對失敗]事件的根本原因
（已用真實資料驗證：VICI.N、TSN.N在2020-03-09、2020-03-16、2021-05-05、
2023-06-05這幾天formation window內同時出現0值，其中3/9、3/16正是2020年3月
COVID熔斷那波劇烈行情期間）。

檢查涵蓋4個層次：
A. 檔案層級：交易日曆缺口、列數異常、欄位（股票）數量的時間趨勢
B. 股票層級：股票「先出現一段時間、中間消失、後來又出現」的不合理斷層、
   同檔案內重複欄位
C. 數值品質（核心）：NaN、0值/負值、formation window內最長連續不變run、
   單分鐘異常跳動報酬率
D. 跨資料源：每個交易日是否有對應的formation table輸出檔

輸出4份CSV到 OUTPUT_DIR：
  integrity_file_summary.csv    每個檔案一列的摘要統計
  integrity_suspicious_detail.csv  逐筆(日期,股票,問題類型,嚴重度,細節)清單
  integrity_ticker_presence.csv    每檔股票的首末出現日期＋是否有中間斷層
  integrity_calendar_gaps.csv      應該有交易但沒有檔案的日期清單
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, "/content/GGWP_US_LLTL/src")
import hyperparameters

# ==========================================
# 設定
# ==========================================
DAT_PATH = hyperparameters.PathRoot + "full_data_AB"
FORMATION_TABLE_DIR = hyperparameters.formation_table_dir
OPEN_DELETE = hyperparameters.open_delete
FORMATION_PERIOD = hyperparameters.formation_period
TRADING_PERIOD = hyperparameters.trading_period
NORMAL_ROW_COUNT = OPEN_DELETE + FORMATION_PERIOD + TRADING_PERIOD  # 標準交易日應有的分鐘數

OUTPUT_DIR = "/content/GGWP_US_LLTL/data_cleaning/integrity_check"

# ── 閾值：flat-run跟extreme-jump是校準過的，NaN/0值/負值則是「只要出現就標記」──
#
# 校準依據（掃過全部1,247個檔案，隨機抽5000+筆股票-日期樣本得到的flat-run分布）：
#   50百分位=2, 90百分位=3, 95百分位=3, 99百分位=7, 99.9百分位=15, 99.99百分位=29
# 用兩層閾值：略高於99百分位算「留意」，接近99.9百分位算「優先檢查」。
# 注意：flat-run本身並不是我們已知那次VICI.N/TSN.N大量配對失敗事件的成因
# （VICI.N歷史最長run只有15，並沒有明顯偏離一般股票的尾端分布），純粹是
# 「forward-fill久未成交」的資料異味(data smell)，嚴重度定為較低的MEDIUM/LOW。
FLAT_RUN_WATCH_THRESHOLD = 10
FLAT_RUN_SUSPICIOUS_THRESHOLD = 20
# 單分鐘報酬率超過±15%視為異常跳動（可能是fat-finger或除權息未調整）
EXTREME_RETURN_THRESHOLD = 0.15


def formation_table_filename(date_str):
    """輸入 '2018-10-24' -> 輸出跟Main_Check_parallel.py一致的檔名"""
    return f"{date_str.replace('-', '')}for{FORMATION_PERIOD}del{OPEN_DELETE}_AB.csv"


def max_flat_run_per_column(arr):
    """
    arr: 2D numpy array, shape (n_rows, n_cols)
    回傳每一欄「最長連續完全相等run」的長度（NaN視為斷點，不算相等）。
    全程用numpy向量化，避免對每一欄各自呼叫pandas groupby的開銷
    （1,247個檔案 x 每檔案幾百欄，逐欄呼叫groupby會太慢）。
    """
    n_rows, n_cols = arr.shape
    if n_rows == 0:
        return np.zeros(n_cols)
    valid = ~np.isnan(arr)
    same_as_prev = np.zeros_like(arr, dtype=bool)
    same_as_prev[1:] = (arr[1:] == arr[:-1]) & valid[1:] & valid[:-1]
    is_new_group = ~same_as_prev
    group_id = is_new_group.cumsum(axis=0)

    max_runs = np.zeros(n_cols)
    for j in range(n_cols):
        col_valid = valid[:, j]
        if not col_valid.any():
            continue
        counts = np.bincount(group_id[col_valid, j])
        max_runs[j] = counts.max() if len(counts) else 0
    return max_runs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(DAT_PATH) if f.endswith("_AB.csv"))
    print(f"共 {len(files)} 個檔案，開始掃描...")

    file_summary_rows = []
    detail_rows = []
    ticker_dates = {}  # ticker -> list of date_str（用來做B.股票出現斷層分析）

    for fname in tqdm(files, desc="掃描完整度", unit="day", mininterval=1):
        date_str = fname[:10]
        path = os.path.join(DAT_PATH, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            file_summary_rows.append({"date": date_str, "read_error": repr(e)})
            continue

        n_rows, n_cols = df.shape
        for col in df.columns:
            ticker_dates.setdefault(col, []).append(date_str)

        # 只看open_delete之後、實際會被formation table/preprocess用到的區間，
        # 不看開盤前open_delete那幾分鐘（那段本來就會被跳過，不影響下游）
        used = df.iloc[OPEN_DELETE:OPEN_DELETE + FORMATION_PERIOD + TRADING_PERIOD]
        formation_part = used.iloc[:FORMATION_PERIOD]
        arr_used = used.to_numpy(dtype=float)
        arr_formation = formation_part.to_numpy(dtype=float)

        n_nan = np.isnan(arr_used).sum(axis=0)
        n_zero = (arr_used == 0).sum(axis=0)
        n_neg = (arr_used < 0).sum(axis=0)
        max_run = max_flat_run_per_column(arr_formation)  # C.8只看formation period內的run

        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.diff(arr_used, axis=0) / arr_used[:-1]
        n_extreme = (np.abs(ret) > EXTREME_RETURN_THRESHOLD).sum(axis=0)

        cols = df.columns.to_numpy()
        n_tickers_nan = int((n_nan > 0).sum())
        n_tickers_zero_neg = int(((n_zero > 0) | (n_neg > 0)).sum())
        n_tickers_suspicious_run = int((max_run >= FLAT_RUN_SUSPICIOUS_THRESHOLD).sum())
        n_tickers_extreme = int((n_extreme > 0).sum())

        # 逐筆detail：任何NaN/0值/負值都記（高風險，直接對應log()產生-inf/NaN的機制）
        for idx in np.where((n_nan > 0) | (n_zero > 0) | (n_neg > 0))[0]:
            detail_rows.append({
                "date": date_str, "ticker": cols[idx], "issue": "nan_or_zero_or_negative",
                "severity": "HIGH", "n_nan": int(n_nan[idx]), "n_zero": int(n_zero[idx]),
                "n_negative": int(n_neg[idx]),
            })
        # flat run：分兩層嚴重度
        for idx in np.where(max_run >= FLAT_RUN_SUSPICIOUS_THRESHOLD)[0]:
            detail_rows.append({
                "date": date_str, "ticker": cols[idx], "issue": "flat_run",
                "severity": "MEDIUM", "max_flat_run": int(max_run[idx]),
            })
        for idx in np.where((max_run >= FLAT_RUN_WATCH_THRESHOLD) & (max_run < FLAT_RUN_SUSPICIOUS_THRESHOLD))[0]:
            detail_rows.append({
                "date": date_str, "ticker": cols[idx], "issue": "flat_run",
                "severity": "LOW", "max_flat_run": int(max_run[idx]),
            })
        # 異常跳動
        for idx in np.where(n_extreme > 0)[0]:
            detail_rows.append({
                "date": date_str, "ticker": cols[idx], "issue": "extreme_jump",
                "severity": "MEDIUM", "n_extreme_jumps": int(n_extreme[idx]),
            })

        has_ft = os.path.exists(os.path.join(FORMATION_TABLE_DIR, formation_table_filename(date_str)))

        file_summary_rows.append({
            "date": date_str,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "row_count_abnormal": n_rows != NORMAL_ROW_COUNT,
            "n_tickers_with_nan_or_zero_or_neg": n_tickers_zero_neg,
            "n_tickers_with_nan": n_tickers_nan,
            "n_tickers_suspicious_flat_run": n_tickers_suspicious_run,
            "n_tickers_extreme_jump": n_tickers_extreme,
            "has_formation_table": has_ft,
            "read_error": None,
        })

    file_summary = pd.DataFrame(file_summary_rows)
    detail = pd.DataFrame(detail_rows)

    # ── A. 交易日曆缺口 ──────────────────────────────────────────
    all_dates = sorted(file_summary["date"].dropna().unique())
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=all_dates[0], end_date=all_dates[-1])
        expected_dates = set(schedule.index.strftime("%Y-%m-%d"))
        calendar_source = "pandas_market_calendars (NYSE精確交易日曆)"
    except ImportError:
        expected_dates = set(pd.bdate_range(all_dates[0], all_dates[-1]).strftime("%Y-%m-%d"))
        calendar_source = "純平日推算（無法排除國定假日，缺口清單需人工用行事曆確認）"

    missing_dates = sorted(expected_dates - set(all_dates))
    calendar_gaps = pd.DataFrame({"missing_date": missing_dates})

    # ── B. 股票出現斷層分析 ──────────────────────────────────────
    presence_rows = []
    all_dates_sorted = all_dates
    for ticker, dates in ticker_dates.items():
        present = sorted(set(dates))
        start, end = present[0], present[-1]
        middle_range = [d for d in all_dates_sorted if start <= d <= end]
        missing_in_middle = sorted(set(middle_range) - set(present))
        # 同一檔案內重複欄位：dates list裡同一天出現超過一次
        dup_days = pd.Series(dates).value_counts()
        dup_days = dup_days[dup_days > 1].index.tolist()
        presence_rows.append({
            "ticker": ticker,
            "first_date": start,
            "last_date": end,
            "n_days_present": len(present),
            "n_days_expected_in_range": len(middle_range),
            "n_gap_days_in_middle": len(missing_in_middle),
            "example_gap_dates": missing_in_middle[:5],
            "duplicate_column_dates": dup_days[:5],
        })
    ticker_presence = pd.DataFrame(presence_rows)

    # ── 存檔 ──────────────────────────────────────────────────
    file_summary.to_csv(os.path.join(OUTPUT_DIR, "integrity_file_summary.csv"), index=False)
    detail.to_csv(os.path.join(OUTPUT_DIR, "integrity_suspicious_detail.csv"), index=False)
    ticker_presence.to_csv(os.path.join(OUTPUT_DIR, "integrity_ticker_presence.csv"), index=False)
    calendar_gaps.to_csv(os.path.join(OUTPUT_DIR, "integrity_calendar_gaps.csv"), index=False)

    # ── 摘要報告 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("完整度檢查摘要")
    print("=" * 60)
    print(f"\n[A] 交易日曆缺口（比對來源：{calendar_source}）")
    print(f"  缺少 {len(missing_dates)} 個應有交易日的檔案")
    if missing_dates:
        print(f"  範例: {missing_dates[:10]}")

    abnormal_rows = file_summary[file_summary["row_count_abnormal"] == True]
    print(f"\n[A] 列數不是標準值({NORMAL_ROW_COUNT})的檔案數: {len(abnormal_rows)}")
    if len(abnormal_rows) > 0:
        print(abnormal_rows[["date", "n_rows"]].to_string(index=False))

    print(f"\n[A] 欄位（股票）數量趨勢: 最早 {file_summary.iloc[0]['n_cols']} 檔 "
          f"-> 最新 {file_summary.iloc[-1]['n_cols']} 檔")

    print(f"\n[B] 有「中間斷層」的股票數: {(ticker_presence['n_gap_days_in_middle'] > 0).sum()}")
    gap_leaders = ticker_presence[ticker_presence["n_gap_days_in_middle"] > 0] \
        .sort_values("n_gap_days_in_middle", ascending=False).head(15)
    if len(gap_leaders) > 0:
        print(gap_leaders[["ticker", "first_date", "last_date", "n_gap_days_in_middle", "example_gap_dates"]]
              .to_string(index=False))

    dup_cols = ticker_presence[ticker_presence["duplicate_column_dates"].apply(len) > 0]
    print(f"\n[B] 同檔案內重複欄位的股票數: {len(dup_cols)}")

    print(f"\n[C] 高風險（NaN/0值/負值）明細筆數: {(detail['issue'] == 'nan_or_zero_or_negative').sum() if len(detail) else 0}")
    high_risk_leaders = detail[detail["severity"] == "HIGH"]["ticker"].value_counts().head(15) if len(detail) else pd.Series(dtype=int)
    if len(high_risk_leaders) > 0:
        print("  最常出問題的股票 Top 15:")
        print(high_risk_leaders.to_string())

    print(f"\n[C] flat-run標記筆數（含MEDIUM+LOW）: "
          f"{detail['issue'].eq('flat_run').sum() if len(detail) else 0}")
    print(f"[C] 異常跳動標記筆數: {detail['issue'].eq('extreme_jump').sum() if len(detail) else 0}")

    date_leaders = detail["date"].value_counts().head(15) if len(detail) else pd.Series(dtype=int)
    print("\n[C] 最常出問題的日期 Top 15（所有severity合計）:")
    if len(date_leaders) > 0:
        print(date_leaders.to_string())

    print(f"\n[D] 沒有對應formation table的交易日數: {(file_summary['has_formation_table'] == False).sum()}")

    print(f"\n所有明細已存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
