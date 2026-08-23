import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
import datetime
from datetime import datetime
from matplotlib.dates import DateFormatter, DayLocator
from hyperparameters import PathRoot, current_time
from scipy.optimize import minimize
import json
import glob
from trade import real_env_reward, _get_formation_table
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import warnings
warnings.filterwarnings("error")

_N_WORKERS = 8   # Colab GPU runtime 實際分配約 4–8 vCPU（os.cpu_count()=48 是 host 機器總數，非獨占）


class Backtest():
    def __init__(self, model,
                       Data,
                       open_prob_threshold=0.65,
                       mode="GaussCopGumLoss",
                       experiment_file=None,
                       must_open=False,
                       do_copula_model=True,
                       sl_offset=None,
                       sl_sigma=0.999,
                       do_dynamic_sl=True,
                       data_name="Test",
                       do_discard_expect_prof_threshold=False,
                       expect_prof_threshold=None,
                       use_adj_expect_prof=True,
                       args=None,
                       split_info=None,
                       baseline_mode=False,
                       disable_stop_loss=False
                 ):
        self.model = model
        self.mode = mode
        self.baseline_mode = baseline_mode
        self.disable_stop_loss = disable_stop_loss

        if self.model:
            self.do_copula_model = model.do_copula_model
        else:
            self.do_copula_model = do_copula_model

        if not experiment_file:
            self.experiment_file = current_time
        else:
            self.experiment_file = experiment_file
        self.do_dynamic_sl = do_dynamic_sl
        if self.baseline_mode:
            self.sl_sigma = None   # baseline 不使用 sl 參數（Tsl 已硬編碼為 100000.0）
            self.sl_offset = None
        elif self.do_dynamic_sl:
            assert sl_sigma is not None
            self.sl_sigma = sl_sigma
        else:
            assert sl_offset is not None
            self.sl_offset = sl_offset

        if must_open:
            self.open_prob_threshold = 0.0
        else:
            self.open_prob_threshold = open_prob_threshold

        self.do_discard_expect_prof_threshold = do_discard_expect_prof_threshold
        self.expect_prof_threshold = expect_prof_threshold
        self.use_adj_expect_prof = use_adj_expect_prof

        self.data = Data.copy()
        self.data_size = len(self.data)
        print("Backtest data size :  ", self.data_size)

        self.notes_set = ["Non_Open",
                           "Discard_NegExp",
                           "Below_Tax",
                           "Normal_Close",
                           "Stop_Loss",
                           "Exit",
                           "Above"
                           ]

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.today_profit = 0.0
        self.day_profit_history = []
        self.day_cumulative_profit = []
        self.today_capital = 0.0
        self.day_capital_history = []
        self.day_return_history = []

        self.records_list = []
        self.records_df = None
        self.data_name = data_name
        self.current_date = None

        self.backtest_dir = f"Record/{self.experiment_file}/{self.data_name}/"
        if not os.path.exists(PathRoot + self.backtest_dir):
            os.makedirs(PathRoot + self.backtest_dir)
        if args is not None:
            args_dict = args._asdict()
            with open(PathRoot + self.backtest_dir + 'hyperparameters.json', 'w') as json_file:
                json.dump(args_dict, json_file, indent=4)
        if split_info is not None:
            split_info_dict = split_info._asdict()
            with open(PathRoot + self.backtest_dir + 'split_info.json', 'w') as json_file:
                json.dump(split_info_dict, json_file, indent=4)


    def summarize_daily_performence(self, date):
        if len(self.day_cumulative_profit) == 0:
            self.day_cumulative_profit = [[int(self.current_date),
                                           self.today_profit]]
        else:
            self.day_cumulative_profit.append([int(self.current_date),
                                               self.day_cumulative_profit[-1][1] + self.today_profit])
        self.day_profit_history.append([int(self.current_date), self.today_profit])
        self.day_capital_history.append([int(self.current_date), self.today_capital])
        self.day_return_history.append(
            [int(self.current_date), self.today_profit / self.today_capital if self.today_capital != 0 else 0])

        self.records_df = pd.concat(self.records_list, axis=0)
        self.records_df.to_csv(PathRoot + self.backtest_dir + f"record_{self.current_date}.csv")
        print(f'Save {self.current_date}')

        self.records_list = []

        # reset daily data
        self.today_profit = 0.0
        self.today_capital = 0.0
        self.current_date = date

    def get_Thresholds_ExpProf(self, mode, use_adj_expect_prof, predict_rtop, predict_sigma_rtop,
                                  predict_top, predict_sigma_top,
                                  predict_close, predict_sigma_close,
                                  r12, r13, r23,
                                  c):

        if mode == "GaussCopGumLoss":
            assert self.do_copula_model
            To, expected_return = utils.find_opt_open_threshold(1, r12, r13,
                                               r12, 1, r23,
                                               r13, r23, 1,
                                               predict_rtop, predict_sigma_rtop,
                                               predict_top, predict_sigma_top,
                                               predict_close, predict_sigma_close,
                                               c,
                                               mode="Gumbel", adj=use_adj_expect_prof,
                                               opt_grain=100,
                                               integral_grain=5,  # 原始 10 → 5：計算量減少 8x（10³→5³），精度損失可接受
                                               early_stop=5)
            if self.do_dynamic_sl:
                Tsl = utils.Gumbel_distribution_q(predict_rtop, predict_sigma_rtop, q=self.sl_sigma)
            else:
                Tsl = self.sl_offset + To
        elif mode == "GaussCopNormLoss":
            assert self.do_copula_model
            To, expected_return = utils.find_opt_open_threshold(1, r12, r13,
                                                                r12, 1, r23,
                                                                r13, r23, 1,
                                                                predict_rtop, predict_sigma_rtop,
                                                                predict_top, predict_sigma_top,
                                                                predict_close, predict_sigma_close,
                                                                c,
                                                                mode="Normal", adj=use_adj_expect_prof,
                                                                opt_grain=100,
                                                                integral_grain=5,  # 原始 10 → 5：計算量減少 8x（10³→5³），精度損失可接受
                                                                early_stop=5)
            if self.do_dynamic_sl:
                Tsl = utils.Normal_distribution_q(predict_rtop, predict_sigma_rtop, q=self.sl_sigma)
            else:
                Tsl = self.sl_offset + To
        elif mode == "GumIndLoss":
            assert not self.do_copula_model

            opt_result = minimize(utils.neg_expect_profit_Gumbel,x0=0,
                     args=(predict_rtop, predict_sigma_rtop, predict_top,
                       predict_sigma_top, predict_close, c, use_adj_expect_prof))
            To = opt_result.x[0]
            expected_return = -opt_result.fun

            if self.do_dynamic_sl:
                Tsl = utils.Gumbel_distribution_q(predict_rtop, predict_sigma_rtop, q=self.sl_sigma)
            else:
                Tsl = self.sl_offset + To
        elif mode == "NormIndLoss":
            assert not self.do_copula_model
            opt_result = minimize(utils.neg_expect_profit_Normal, x0=0,
                                  args=(predict_rtop, predict_sigma_rtop, predict_top,
                                        predict_sigma_top, predict_close, c, use_adj_expect_prof))
            To = opt_result.x[0]
            expected_return = -opt_result.fun

            if self.do_dynamic_sl:
                Tsl = utils.Normal_distribution_q(predict_rtop, predict_sigma_rtop, q=self.sl_sigma)
            else:
                Tsl = self.sl_offset + To
        else:
            raise ValueError("Only GaussCopNormLos, GaussCopNormLos, GumIndLoss, or NormIndLoss")

        if self.disable_stop_loss:
            # 停損實質上不會被觸發，只剩「正常平倉」跟「到期強制平倉」兩種出場方式。
            # 用變數型別（float64）的最大值，確保 trade.py 裡 UpStopTrend/DownStopTrend
            # 的比較永遠不會被觸發，且不像 baseline_mode 用 100000.0 是取近似值。
            Tsl = np.finfo(np.float64).max

        return To, Tsl, expected_return

    def _process_single_pair(self, i,
                              s1_arr, s2_arr, date_arr, revert_arr, c_arr,
                              norm_rtop_arr, norm_top_arr, norm_close_arr,
                              paras, revert_prob, cor_mat, formation_lookup):
        """
        計算單一配對的完整 CPU 工作。
        不讀寫任何 self 的可變狀態，所有 thread 可安全並行呼叫。
        回傳一個 dict，由 trade() 的 Phase 3 串行合併。
        """
        s1   = s1_arr[i]
        s2   = s2_arr[i]
        date = date_arr[i]
        pair_revert        = bool(revert_arr[i])
        c                  = float(c_arr[i])
        revert_probability = float(revert_prob[i, 1])
        predict_rtop       = float(paras[i, 0])
        predict_sigma_rtop = float(paras[i, 1])
        predict_top        = float(paras[i, 2])
        predict_sigma_top  = float(paras[i, 3])
        predict_close      = float(paras[i, 4])
        predict_sigma_close= float(paras[i, 5])
        r12 = float(cor_mat[i, 0])
        r13 = float(cor_mat[i, 1])
        r23 = float(cor_mat[i, 2])

        To = To_final = Tsl = expected_return = 0.0
        backtest_profit = capital = 0.0
        w1 = w2 = 0.0
        stock1_num = stock2_num = 0
        close_timing = open_timing = -999
        note = None
        is_tp = is_tn = is_fp = is_fn = False

        if self.baseline_mode:
            # 固定1.5σ baseline：跳過 prob check 和三重積分
            is_tp = pair_revert
            is_fp = not pair_revert
            To    = 1.5
            Tsl   = 100000.0        # 對應 trade.py 原始 StopS（不止損）
            expected_return = 0.0
            To_final = -1 if To < c else To
            if To_final == -1:
                note = self.notes_set[2]
        else:
            if revert_probability < self.open_prob_threshold:
                is_tn = not pair_revert
                is_fn = pair_revert
                To_final = -1
                note = self.notes_set[0]
            else:
                is_tp = pair_revert
                is_fp = not pair_revert

            if To_final != -1:
                To, Tsl, expected_return = self.get_Thresholds_ExpProf(
                    self.mode, self.use_adj_expect_prof,
                    predict_rtop, predict_sigma_rtop,
                    predict_top, predict_sigma_top,
                    predict_close, predict_sigma_close,
                    r12, r13, r23, c
                )
                if To < c:
                    To_final = -1
                    note = self.notes_set[2]
                elif self.do_discard_expect_prof_threshold and expected_return < self.expect_prof_threshold:
                    To_final = -1
                    note = self.notes_set[1]
                else:
                    To_final = To

        if To_final != -1:
            try:
                fp = formation_lookup.get((int(date), str(s1), str(s2)), None)
                trade_result = real_env_reward(s1, s2, date, [To_final, Tsl], formation_params=fp)
                backtest_profit = trade_result["reward"]
                close_timing    = trade_result["close_timing"]
                open_timing     = trade_result["open_timing"]
                capital         = trade_result["capital"]
                stock1_num, stock2_num = trade_result['stock1_num'], trade_result['stock2_num']
                w1, w2 = trade_result["w1"], trade_result["w2"]
                rec = trade_result["record"]
                if   rec == 666: note = self.notes_set[3]
                elif rec == -2:  note = self.notes_set[4]
                elif rec == -4:  note = self.notes_set[5]
                elif rec == 0:   note = self.notes_set[6]
                else: raise ValueError("Note error")
            except ValueError as e:
                if "Johansen_std * OpenS < CostS" in str(e):
                    backtest_profit = 0.0
                    note = self.notes_set[2]
                else:
                    raise

        return {
            "i": i, "date": date, "s1": s1, "s2": s2,
            "predict_rtop": predict_rtop, "predict_sigma_rtop": predict_sigma_rtop,
            "predict_top": predict_top,   "predict_sigma_top": predict_sigma_top,
            "predict_close": predict_close, "predict_sigma_close": predict_sigma_close,
            "r12": r12, "r13": r13, "r23": r23,
            "To": To, "Tsl": Tsl, "To_final": To_final,
            "revert_probability": revert_probability,
            "expected_return": expected_return,
            "backtest_profit": backtest_profit, "capital": capital,
            "stock1_num": stock1_num, "stock2_num": stock2_num,
            "w1": w1, "w2": w2,
            "close_timing": close_timing, "open_timing": open_timing,
            "pair_revert": pair_revert,
            "norm_rtop":  float(norm_rtop_arr[i]),
            "norm_top":   float(norm_top_arr[i]),
            "norm_close": float(norm_close_arr[i]),
            "norm_tax":   c, "note": note,
            "is_tp": is_tp, "is_tn": is_tn, "is_fp": is_fp, "is_fn": is_fn,
        }

    def trade(self,
              existed_paras=None,
              existed_revert_prob=None,
              existed_cor_mat=None,
              timer_precision=100):

        # pred_paras = predict parameters ; revert_prob = open probability ; cor_mat = R12, R13, R23
        if existed_paras is not None and existed_revert_prob is not None:
            paras, revert_prob = existed_paras, existed_revert_prob
            if existed_cor_mat is not None and self.do_copula_model:
                cor_mat = existed_cor_mat
            else:
                # independent assumption
                cor_mat = np.zeros((self.data_size, 3))  # data_size x (r12, r13, r23)
        elif self.model:
            x = np.stack([np.stack(self.data['Norm_Spread'].values),
                          np.stack(self.data['S1_Return'].values),
                          np.stack(self.data['S2_Return'].values)], axis=1)
            if self.do_copula_model:
                
                revert_prob, paras, cor_mat = self.model.select_action(x)
            else:
                
                revert_prob, paras = self.model.select_action(x)
                cor_mat = np.zeros((len(x),3))
        elif self.baseline_mode:
            # baseline 模式：不使用任何模型預測，_process_single_pair 直接用固定1.5σ
            paras       = np.zeros((self.data_size, 6))
            revert_prob = np.zeros((self.data_size, 2))
            cor_mat     = np.zeros((self.data_size, 3))
        else:
            raise ValueError("If no model given, please give predetermined parameters")

        # ── 預先將 DataFrame 欄位取出為 numpy array ──────────────────────────────
        # 避免在多執行緒中重複呼叫 iloc（pandas iloc 有 Python 層開銷）
        s1_arr         = self.data['S1'].values
        s2_arr         = self.data['S2'].values
        date_arr       = self.data['Date'].values
        revert_arr     = self.data['Revert'].values
        c_arr          = self.data['Norm_Tax'].values
        norm_rtop_arr  = self.data['Norm_Rtop'].values
        norm_top_arr   = self.data['Norm_Top'].values
        norm_close_arr = self.data['Norm_Close'].values

        # ── Phase 2: 多執行緒平行計算 ─────────────────────────────────────────
        # 按日分組，每日內所有 pair 並行送入 thread pool；
        # 等本日全部完成再提交下一日，確保日邊界的 summarize 時機正確。
        # Numba @njit 不佔 GIL → 6 個 thread 真正同時跑 6 核。
        day_indices = defaultdict(list)
        for i in range(self.data_size):
            day_indices[date_arr[i]].append(i)
        sorted_dates = sorted(day_indices.keys())

        # ── 預先建立 formation lookup（消除 real_env_reward 的 pandas 篩選 GIL 瓶頸）────
        # 把每天 formation table 的 (s1, s2) → params 轉成 Python dict，
        # 讓每個 thread 做 O(1) dict 查找，不再做 DataFrame.loc / 布林篩選。
        print("Building formation lookup...")
        formation_lookup: dict = {}
        for _date in tqdm(sorted_dates, desc="Formation lookup", ncols=110):
            _ds = str(int(_date))
            _yr, _mo, _dy = _ds[:4], _ds[4:6], _ds[6:8]
            _ft = _get_formation_table(_yr, _mo, _dy)
            _s1c  = _ft['S1'].values
            _s2c  = _ft['S2'].values
            _w1c  = _ft['w1'].values
            _w2c  = _ft['w2'].values
            _lqc  = _ft['VECM(q)'].values.astype(int)
            _mtc  = _ft['VECM_Model_Type'].values.astype(int)
            _icc  = _ft['Johansen_intercept'].values
            _slc  = _ft['Johansen_slope'].values
            _stc  = _ft['Johansen_std'].values
            for _idx in range(len(_ft)):
                formation_lookup[(int(_date), str(_s1c[_idx]), str(_s2c[_idx]))] = (
                    float(_w1c[_idx]), float(_w2c[_idx]),
                    int(_lqc[_idx]),   int(_mtc[_idx]),
                    float(_icc[_idx]), float(_slc[_idx]), float(_stc[_idx])
                )
        print(f"Formation lookup ready: {len(formation_lookup):,} entries")

        worker = lambda i: self._process_single_pair(
            i, s1_arr, s2_arr, date_arr, revert_arr, c_arr,
            norm_rtop_arr, norm_top_arr, norm_close_arr,
            paras, revert_prob, cor_mat, formation_lookup
        )

        all_results = [None] * self.data_size

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            with tqdm(total=self.data_size, desc="Backtest", ncols=110, unit="pair") as pbar:
                for date in sorted_dates:
                    indices = day_indices[date]
                    day_futures = {executor.submit(worker, i): i for i in indices}
                    for fut in as_completed(day_futures):
                        all_results[day_futures[fut]] = fut.result()
                        pbar.update(1)


        # ── Phase 3: 串行合併（極快，純 Python 加法 + list append）────────────
        # 必須按日期排序，因為 all_results 的 index 順序跟著 DataFrame 行號，
        # 而 DataFrame 是「先按配對、再按日期」排列，直接遍歷會讓日期非單調，
        # 造成 summarize_daily_performence 在同一天被多次觸發，圖表出現多段跳線。
        all_results_sorted = sorted(all_results, key=lambda r: r["date"])
        self.current_date = all_results_sorted[0]["date"]
        for result in all_results_sorted:
            date = result["date"]
            if date != self.current_date:
                self.summarize_daily_performence(date)

            self.TP += result["is_tp"]
            self.TN += result["is_tn"]
            self.FP += result["is_fp"]
            self.FN += result["is_fn"]

            # capital/backtest_profit 在未開倉時已為 0，直接累加無需 if 判斷
            self.today_capital += result["capital"]
            self.today_profit  += result["backtest_profit"]

            i = result["i"]
            current_record = pd.DataFrame({
                "Date":                 result["date"],
                "S1":                   result["s1"],
                "S2":                   result["s2"],
                "rtop":                 result["predict_rtop"],
                "rtop_sigma":           result["predict_sigma_rtop"],
                "top":                  result["predict_top"],
                "top_sigma":            result["predict_sigma_top"],
                "close":                result["predict_close"],
                "close_sigma":          result["predict_sigma_close"],
                "r12":                  result["r12"],
                "r13":                  result["r13"],
                "r23":                  result["r23"],
                "Open_Threshold":       result["To"],
                "Stop_Loss_Threshold":  result["Tsl"],
                "Stop_Loss_offset":     result["Tsl"] - result["To"],
                "Final_Open_Threshold": result["To_final"],
                "Revert_Prob":          result["revert_probability"],
                "Expected_Return":      result["expected_return"],
                "Backtest_Profit":      result["backtest_profit"],
                "stock1_num":           result["stock1_num"],
                "stock2_num":           result["stock2_num"],
                "W1":                   result["w1"],
                "W2":                   result["w2"],
                "Capital":              result["capital"],
                "Close_timing":         result["close_timing"],
                "Open_timing":          result["open_timing"],
                "Revert":               result["pair_revert"],
                "Norm_Rtop":            result["norm_rtop"],
                "Norm_Top":             result["norm_top"],
                "Norm_Close":           result["norm_close"],
                "Norm_Tax":             result["norm_tax"],
                "note":                 result["note"],
            }, index=[i])
            self.records_list.append(current_record)

        # 最後一天
        self.summarize_daily_performence(self.current_date)

        '''-------------------------------------------Store results-------------------------------------------'''

        for f in glob.glob(PathRoot + self.backtest_dir + f"record_*.csv"):
            day_df = pd.read_csv(f)
            self.records_list.append(day_df)

        self.records_df = pd.concat(self.records_list, axis=0)
        self.records_df.to_csv(PathRoot + self.backtest_dir + f"record.csv")
        print("record.csv saved")

        return self.records_df, paras, revert_prob, cor_mat

    def summary_output(self):

        records_size = len(self.records_df)

        model_output_summery = dict()
        model_output_summery["TP"] = self.TP / records_size
        model_output_summery["TN"] = self.TN / records_size
        model_output_summery["FP"] = self.FP / records_size
        model_output_summery["FN"] = self.FN / records_size
        model_output_summery["accuracy"] = (self.TP + self.TN) / records_size
        if (self.TN + self.FP) > 0:
            model_output_summery["recall"] = self.TN / (self.TN + self.FP)

        model_output_summery["Mean Top"] = np.mean(self.records_df["top"])
        model_output_summery["Max Top"] = np.max(self.records_df["top"])
        model_output_summery["Min Top"] = np.min(self.records_df["top"])
        model_output_summery["STD Top"] = np.std(self.records_df["top"])

        model_output_summery["Mean Top Sigma"] = np.mean(self.records_df["top_sigma"])
        model_output_summery["Max Top Sigma"] = np.max(self.records_df["top_sigma"])
        model_output_summery["Min Top Sigma"] = np.min(self.records_df["top_sigma"])
        model_output_summery["STD Top Sigma"] = np.std(self.records_df["top_sigma"])

        model_output_summery["Mean RTop"] = np.mean(self.records_df["rtop"])
        model_output_summery["Max RTop"] = np.max(self.records_df["rtop"])
        model_output_summery["Min RTop"] = np.min(self.records_df["rtop"])
        model_output_summery["STD RTop"] = np.std(self.records_df["rtop"])

        model_output_summery["Mean RTop Sigma"] = np.mean(self.records_df["rtop_sigma"])
        model_output_summery["Max RTop Sigma"] = np.max(self.records_df["rtop_sigma"])
        model_output_summery["Min RTop Sigma"] = np.min(self.records_df["rtop_sigma"])
        model_output_summery["STD RTop Sigma"] = np.std(self.records_df["rtop_sigma"])

        model_output_summery["Mean Close"] = np.mean(self.records_df["close"])
        model_output_summery["Max Close"] = np.max(self.records_df["close"])
        model_output_summery["Min Close"] = np.min(self.records_df["close"])
        model_output_summery["STD Close"] = np.std(self.records_df["close"])

        model_output_summery["Mean Open Threshold"] = np.mean(self.records_df["Open_Threshold"])
        model_output_summery["Max Open Threshold"] = np.max(self.records_df["Open_Threshold"])
        model_output_summery["Min Open Threshold"] = np.min(self.records_df["Open_Threshold"])
        model_output_summery["STD Open Threshold"] = np.std(self.records_df["Open_Threshold"])

        opened_records = self.records_df[self.records_df["Final_Open_Threshold"] > 0]
        if len(opened_records) > 0:
            model_output_summery["Mean Opened Threshold"] = np.mean(opened_records["Final_Open_Threshold"])
            model_output_summery["Max Opened Threshold"] = np.max(opened_records["Final_Open_Threshold"])
            model_output_summery["Min Opened Threshold"] = np.min(opened_records["Final_Open_Threshold"])
            model_output_summery["STD Opened Threshold"] = np.std(opened_records["Final_Open_Threshold"])

        if self.disable_stop_loss:
            # Stop_Loss_Threshold 欄位是 float64 最大值，加總數百萬筆會直接溢位成 inf，
            # 而本檔開頭 warnings.filterwarnings("error") 會把這個 RuntimeWarning 升級成例外。
            # 停損被停用時，這組統計本身也沒有意義（沒有真正的停損門檻可以總結），直接跳過。
            model_output_summery["Mean Stop Loss"] = None
            model_output_summery["Max Stop Loss"] = None
            model_output_summery["Min Stop Loss"] = None
            model_output_summery["STD Stop Loss"] = None
        else:
            model_output_summery["Mean Stop Loss"] = np.mean(self.records_df["Stop_Loss_Threshold"])
            model_output_summery["Max Stop Loss"] = np.max(self.records_df["Stop_Loss_Threshold"])
            model_output_summery["Min Stop Loss"] = np.min(self.records_df["Stop_Loss_Threshold"])
            model_output_summery["STD Stop Loss"] = np.std(self.records_df["Stop_Loss_Threshold"])

        '''-------------------------------------------Store results-------------------------------------------'''

        pd.DataFrame(list(model_output_summery.items()), columns=["Summery", "Value"]).to_csv(
            PathRoot + self.backtest_dir + f"model_output_summery.csv")
        print("model_output_summery.csv saved")

    def summary_record(self):

        records_size = len(self.records_df)

        # true environment summery
        backtest_performance_summery = dict()

        non_open_case = self.records_df[self.records_df["note"] == self.notes_set[0]]
        discard_case = self.records_df[self.records_df["note"] == self.notes_set[1]]
        below_tax_case = self.records_df[self.records_df["note"] == self.notes_set[2]]
        normal_close_case = self.records_df[self.records_df["note"] == self.notes_set[3]]
        stop_loss_case = self.records_df[self.records_df["note"] == self.notes_set[4]]
        force_close_case = self.records_df[self.records_df["note"] == self.notes_set[5]]
        above_top_case = self.records_df[self.records_df["note"] == self.notes_set[6]]

        backtest_win_case = self.records_df[self.records_df["Backtest_Profit"] > 0 ]
        backtest_lose_case = self.records_df[self.records_df["Backtest_Profit"] < 0]
        backtest_tie_case = self.records_df[self.records_df["Backtest_Profit"] == 0]

        Backtest_Profit = self.records_df["Backtest_Profit"]

        open_count = len(normal_close_case) + len(stop_loss_case) + len(force_close_case)
        backtest_performance_summery["Total open count"] = open_count
        # Win rate + Tie rate + Lose rate should equal to 1
        if open_count > 0:
            backtest_performance_summery["Win per open"] = len(backtest_win_case) / open_count
            backtest_performance_summery["Lose per open"] = len(backtest_lose_case) / open_count
        backtest_performance_summery["Win per record"] = len(backtest_win_case) / records_size
        backtest_performance_summery["Lose per record"] = len(backtest_lose_case) / records_size
        backtest_performance_summery["Tie per record"] = len(backtest_tie_case) / records_size

        backtest_performance_summery["Non Open per record"] = len(non_open_case) / records_size
        backtest_performance_summery["Discard per record"] = len(discard_case) / records_size
        backtest_performance_summery["Below Tax per record"] = len(below_tax_case) / records_size
        backtest_performance_summery["Above Top per record"] = len(above_top_case) / records_size


        if open_count > 0:
            backtest_performance_summery["Normal Close per open"] = len(normal_close_case) / open_count
            backtest_performance_summery["Stop Loss per open"] = len(stop_loss_case) / open_count
            backtest_performance_summery["Force Close per open"] = len(force_close_case) / open_count

        backtest_performance_summery["Earn amount"] = backtest_win_case["Backtest_Profit"].sum()
        backtest_performance_summery["Loss amount"] = backtest_lose_case["Backtest_Profit"].sum()
        backtest_performance_summery["Total P&L"] = Backtest_Profit.sum()

        if open_count > 0:
            backtest_performance_summery["Profit per open"] = Backtest_Profit.sum() / open_count
        if len(backtest_win_case) > 0:
            backtest_performance_summery["True earn amount per win count"] = backtest_win_case["Backtest_Profit"].mean()
        if len(backtest_lose_case) > 0:
            backtest_performance_summery["True Lose amount per lose count"] = backtest_lose_case["Backtest_Profit"].mean()

        if len(self.day_return_history) > 0:
            SHR = utils.calculate_Sharpe_Ratio(np.array(self.day_return_history)[:, 1])
            SOR = utils.calculate_Sortino_Ratio(np.array(self.day_return_history)[:, 1])
            backtest_performance_summery["Sharp ratio(daily)"] = SHR
            backtest_performance_summery["Sortino ratio(daily)"] = SOR

        if len(self.day_cumulative_profit) > 0:
            MDD = utils.calculate_MDD(np.array(self.day_cumulative_profit)[:, 1], percent=False)
            MDD_percent = utils.calculate_MDD(np.array(self.day_cumulative_profit)[:, 1], percent=True)
            backtest_performance_summery["MDD"] = MDD
            backtest_performance_summery["MDD_percent"] = MDD_percent

        '''-------------------------------------------Store results-------------------------------------------'''

        pd.DataFrame(list(backtest_performance_summery.items()), columns=["Summery", "Value"]).to_csv(
            PathRoot + self.backtest_dir + f"backtest_performance_summery.csv")
        print("backtest_performance_summery.csv saved")

    def summary_daily_record(self):

        eachday_record = pd.DataFrame({
            "Date": np.array(self.day_profit_history)[:, 0],
            "profit": np.array(self.day_profit_history)[:, 1],
            "cumulative_profit": np.array(self.day_cumulative_profit)[:, 1],
            "capital": np.array(self.day_capital_history)[:, 1],
            "return": np.array(self.day_return_history)[:, 1],
        }, index=np.array(self.day_profit_history)[:, 0])

        '''-------------------------------------------Store results-------------------------------------------'''

        eachday_record.to_csv(PathRoot + self.backtest_dir + f"eachday_record.csv")
        print("eachday_record.csv saved")

    def plot_single_output_distribution(self, title, name, data, clip_percentile=99):
        clip_max = np.percentile(data, clip_percentile)
        plt.title(title)
        plt.hist(data, bins=100, range=(data.min(), clip_max), density=True, edgecolor='black')
        plt.axvline(np.median(data), c='r', label='median')
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.savefig(name)
        plt.close()

    def plot_output_distributions(self):
        dir = self.backtest_dir + "output_distributions/"
        if not os.path.exists(PathRoot + dir):
            os.makedirs(PathRoot + dir)

        revert_pairs = self.records_df[self.records_df["Norm_Rtop"] != -1]
        revert_open_pairs = revert_pairs[revert_pairs["Final_Open_Threshold"] != -1]
        if len(revert_open_pairs) == 0:
            print("[plot_output_distributions] No reverted+opened pairs, skipping distribution plots.")
            return
        final_action = revert_open_pairs["Final_Open_Threshold"].values
        self.plot_single_output_distribution("Final Open Threshold Distribution (reverted pairs and opened pairs only)",
                                             PathRoot + dir + f"Final_Open_Threshold.png",
                                             final_action)

        rtop = revert_open_pairs["rtop"].values
        self.plot_single_output_distribution("Predicted Rtop Distribution (reverted pairs and opened pairs only)",
                                             PathRoot + dir + f"Rtop.png",
                                             rtop)

        top = revert_open_pairs["top"].values
        self.plot_single_output_distribution("Predicted Top Distribution (reverted pairs and opened pairs only)",
                                             PathRoot + dir + f"Top.png",
                                             top)

        close = revert_open_pairs["close"].values
        self.plot_single_output_distribution("Predicted Close Distribution (reverted pairs and opened pairs only)",
                                             PathRoot + dir + f"Close.png",
                                             close)

    def summary(self, do_summary_output=True,
                      do_summary_record=True,
                      do_summary_daily_record=True,
                      do_plot_output_distributions=True,
                      do_plot_daily_tradings=True):
        records_size = len(self.records_df)
        print("records_size is equal to data_size: ", records_size == self.data_size)

        if do_summary_output and not self.baseline_mode:
            self.summary_output()
        if do_summary_record:
            self.summary_record()
        if do_summary_daily_record:
            self.summary_daily_record()
        if do_plot_output_distributions and not self.baseline_mode:
            self.plot_output_distributions()
        if do_plot_daily_tradings:
            self.plot_daily_tradings()

    def read_record(self, record_path):
        if ".csv" not in record_path:
            raise ValueError(".csv file only")
        if os.path.exists(record_path):
            self.records_df = pd.read_csv(record_path, index_col=0)
            print("Record.csv length:",len(self.records_df))
        else:
            raise ValueError(record_path, "not exist")

    def plot_single_daily_trading(self, title, name, data, dates):
        plt.title(title)
        dates_list = [datetime.strptime(str(int(date))[2:], "%y%m%d") for date in dates]
        plt.plot(dates_list, data)
        date_format = DateFormatter("%y%m%d")
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.gca().xaxis.set_major_locator(
            DayLocator(interval=(len(dates) // 10 if len(dates) // 10 > 1 else 1)))
        plt.gcf().autofmt_xdate()
        plt.savefig(name)
        plt.close()

    def plot_daily_tradings(self):
        dir = self.backtest_dir + "daily_tradings/"
        if not os.path.exists(PathRoot + dir):
            os.makedirs(PathRoot + dir)

        self.plot_single_daily_trading("Each Day Profit",
                                       PathRoot + dir + f"EachDay_profit.png",
                                       np.array(self.day_profit_history)[:, 1],
                                       np.array(self.day_profit_history)[:, 0])

        self.plot_single_daily_trading("Cumulative Each Day Profit",
                                       PathRoot + dir + f"Cumulative_profit.png",
                                       np.array(self.day_cumulative_profit)[:, 1],
                                       np.array(self.day_cumulative_profit)[:, 0])

        self.plot_single_daily_trading("Each Day Capital",
                                       PathRoot + dir + f"Each_day_capital.png",
                                       np.array(self.day_capital_history)[:, 1],
                                       np.array(self.day_capital_history)[:, 0])

        self.plot_single_daily_trading("Each Day Return",
                                       PathRoot + dir + f"Each_day_return.png",
                                       np.array(self.day_return_history)[:, 1],
                                       np.array(self.day_return_history)[:, 0])

