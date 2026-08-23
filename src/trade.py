# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import MyTool as mt
import time
import os
import threading
from numba import njit
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from hyperparameters import *

@njit(cache=True)
def trade(cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS, FinalOpen,
          capital, trCost):
    Profit = np.zeros((1, 5))
    Count  = np.zeros((1, 5))
    UpOpenTrend   = cy_mean + Johansen_std * OpenS
    UpStopTrend   = cy_mean + Johansen_std * StopS
    DownOpenTrend = cy_mean - Johansen_std * OpenS
    DownStopTrend = cy_mean - Johansen_std * StopS
    Position = 0
    ibeta0 = 0.0
    ibeta1 = 0.0
    openP  = 0.0
    ForceP = 0.0
    opencount    = 0
    opentime     = 0
    closetime    = 0
    opens1payoff  = 0.0
    opens2payoff  = 0.0
    closes1payoff = 0.0
    closes2payoff = 0.0
    CrossTime = 0

    for ti in range(form_del_min + 1, form_del_min + form_len - 1):
        if (cy[ti-1] < cy_mean[ti] < cy[ti+1]) or (cy[ti-1] > cy_mean[ti] > cy[ti+1]):
            CrossTime += 1

    for ti in range(form_len + form_del_min, cy.shape[0] + 1):
        if ti == cy.shape[0]:
            if Position == 1:
                closes1payoff = Position * rawS[-1, 0] * ibeta0
                closes2payoff = Position * rawS[-1, 1] * ibeta1
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] += 1
                else:
                    Profit[0, 4] = ForceP
                    Count[0, 4] += 1
                Position = 0
                closetime = cy.shape[0]
            elif Position == -1:
                closes1payoff = Position * rawS[-1, 0] * ibeta0
                closes2payoff = Position * rawS[-1, 1] * ibeta1
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] += 1
                else:
                    Profit[0, 4] = ForceP
                    Count[0, 4] += 1
                Position = 0
                closetime = cy.shape[0]
        else:
            if opencount <= 1:
                if Position == 1 and cy[ti] >= cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti] != np.inf:
                    closes1payoff = Position * rawS[ti, 0] * ibeta0
                    closes2payoff = Position * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1]  += 1
                    Position  = 0
                    closetime = ti
                elif Position == -1 and cy[ti] <= cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti] != -np.inf:
                    closes1payoff = Position * rawS[ti, 0] * ibeta0
                    closes2payoff = Position * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1]  += 1
                    Position  = 0
                    closetime = ti
                elif Position == 1 and cy[ti] <= DownStopTrend[ti] and ti < cy.shape[0]-1 and cy[ti] != -np.inf:
                    closes1payoff = Position * rawS[ti, 0] * ibeta0
                    closes2payoff = Position * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2]  += 1
                    Position  = -10
                    closetime = ti
                elif Position == -1 and cy[ti] >= UpStopTrend[ti] and ti < cy.shape[0]-1 and cy[ti] != np.inf:
                    closes1payoff = Position * rawS[ti, 0] * ibeta0
                    closes2payoff = Position * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2]  += 1
                    Position  = -10
                    closetime = ti
                elif Position == 0 and cy[ti] <= DownOpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[ti] != -np.inf:
                    Position = 1
                    ibeta0, ibeta1 = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1],
                                                   rawS[ti, 0], rawS[ti, 1], Maxi, capital)
                    opens1payoff = -Position * rawS[ti, 0] * ibeta0
                    opens2payoff = -Position * rawS[ti, 1] * ibeta1
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] += 1
                    opencount   += 1
                    opentime     = ti
                elif Position == 0 and cy[ti] >= UpOpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[ti] != np.inf:
                    Position = -1
                    ibeta0, ibeta1 = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1],
                                                   rawS[ti, 0], rawS[ti, 1], Maxi, capital)
                    opens1payoff = -Position * rawS[ti, 0] * ibeta0
                    opens2payoff = -Position * rawS[ti, 1] * ibeta1
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] += 1
                    opencount   += 1
                    opentime     = ti
            else:
                break

    Profit[0, 0] = np.sum(Profit[0, 1:5])
    trade_capital = 0.0
    if opens1payoff > 0 and opens2payoff > 0:
        trade_capital = abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0:
        trade_capital = abs(opens1payoff) + trCost * abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0:
        trade_capital = trCost * abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0:
        trade_capital = trCost * abs(opens1payoff) + trCost * abs(opens2payoff)

    return (Profit, Count, opentime, closetime, trade_capital, ibeta0, ibeta1, CrossTime)


@njit(cache=True)
def trade_down(cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS,
               FinalOpen, capital, trCost):
    Profit = np.zeros((1, 5))
    Count  = np.zeros((1, 5))
    OpenTrend = cy_mean + Johansen_std * OpenS
    StopTrend = cy_mean + Johansen_std * StopS
    Position    = 0
    ibeta0      = 0.0
    ibeta1      = 0.0
    LongOrShort = -1
    openP       = 0.0
    ForceP      = 0.0
    opencount    = 0
    opentime     = 0
    closetime    = 0
    opens1payoff  = 0.0
    opens2payoff  = 0.0
    closes1payoff = 0.0
    closes2payoff = 0.0
    CrossTime = 0

    for ti in range(form_del_min + 1, form_del_min + form_len - 1):
        if (cy[ti-1] < cy_mean[ti] < cy[ti+1]) or (cy[ti-1] > cy_mean[ti] > cy[ti+1]):
            CrossTime += 1

    for ti in range(form_del_min + form_len, cy.shape[0] + 1):
        if ti == cy.shape[0]:
            if Position == 1:
                closes1payoff = LongOrShort * rawS[-1, 0] * ibeta0
                closes2payoff = LongOrShort * rawS[-1, 1] * ibeta1
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] += 1
                else:
                    Profit[0, 4] = ForceP
                    Count[0, 4] += 1
                Position  = 0
                closetime = cy.shape[0]
        else:
            if opencount <= 1:
                if Position == 1 and cy[ti] <= cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti] != -np.inf:
                    closes1payoff = LongOrShort * rawS[ti, 0] * ibeta0
                    closes2payoff = LongOrShort * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1]  += 1
                    Position  = 0
                    closetime = ti
                elif Position == 1 and cy[ti] >= StopTrend[ti] and ti < cy.shape[0]-1 and cy[ti] != np.inf:
                    closes1payoff = LongOrShort * rawS[ti, 0] * ibeta0
                    closes2payoff = LongOrShort * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2]  += 1
                    Position  = -1
                    closetime = ti
                elif Position == 0 and cy[ti] >= OpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[ti] != np.inf:
                    Position = 1
                    ibeta0, ibeta1 = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1],
                                                   rawS[ti, 0], rawS[ti, 1], Maxi, capital)
                    opens1payoff = -LongOrShort * rawS[ti, 0] * ibeta0
                    opens2payoff = -LongOrShort * rawS[ti, 1] * ibeta1
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] += 1
                    opencount   += 1
                    opentime     = ti
            else:
                break

    Profit[0, 0] = np.sum(Profit[0, 1:5])
    trade_capital = 0.0
    if opens1payoff > 0 and opens2payoff > 0:
        trade_capital = abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0:
        trade_capital = abs(opens1payoff) + trCost * abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0:
        trade_capital = trCost * abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0:
        trade_capital = trCost * abs(opens1payoff) + trCost * abs(opens2payoff)

    return (Profit, Count, opentime, closetime, trade_capital, ibeta0, ibeta1, CrossTime)


@njit(cache=True)
def trade_up(cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS,
             FinalOpen, capital, trCost):
    Profit = np.zeros((1, 5))
    Count  = np.zeros((1, 5))
    OpenTrend = cy_mean - Johansen_std * OpenS
    StopTrend = cy_mean - Johansen_std * StopS
    Position    = 0
    ibeta0      = 0.0
    ibeta1      = 0.0
    LongOrShort = 1
    openP        = 0.0
    ForceP       = 0.0
    opencount    = 0
    opentime     = 0
    closetime    = 0
    opens1payoff  = 0.0
    opens2payoff  = 0.0
    closes1payoff = 0.0
    closes2payoff = 0.0
    CrossTime = 0

    for ti in range(form_del_min + 1, form_del_min + form_len):
        if (cy[ti-1] < cy_mean[ti] < cy[ti+1]) or (cy[ti-1] > cy_mean[ti] > cy[ti+1]):
            CrossTime += 1

    for ti in range(form_del_min + form_len, cy.shape[0] + 1):
        if ti == cy.shape[0]:
            if Position == 1:
                closes1payoff = LongOrShort * rawS[-1, 0] * ibeta0
                closes2payoff = LongOrShort * rawS[-1, 1] * ibeta1
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] += 1
                else:
                    Profit[0, 4] = ForceP
                    Count[0, 4] += 1
                Position  = 0
                closetime = cy.shape[0]
        else:
            if opencount <= 1:
                if Position == 1 and cy[ti] >= cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti] != np.inf:
                    closes1payoff = LongOrShort * rawS[ti, 0] * ibeta0
                    closes2payoff = LongOrShort * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1]  += 1
                    Position  = 0
                    closetime = ti
                elif Position == 1 and cy[ti] <= StopTrend[ti] and ti < cy.shape[0]-1 and cy[ti] != -np.inf:
                    closes1payoff = LongOrShort * rawS[ti, 0] * ibeta0
                    closes2payoff = LongOrShort * rawS[ti, 1] * ibeta1
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2]  += 1
                    Position  = -1
                    closetime = ti
                elif Position == 0 and cy[ti] <= OpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[ti] != -np.inf:
                    Position = 1
                    ibeta0, ibeta1 = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1],
                                                   rawS[ti, 0], rawS[ti, 1], Maxi, capital)
                    opens1payoff = -LongOrShort * rawS[ti, 0] * ibeta0
                    opens2payoff = -LongOrShort * rawS[ti, 1] * ibeta1
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] += 1
                    opencount   += 1
                    opentime     = ti
            else:
                break

    Profit[0, 0] = np.sum(Profit[0, 1:5])
    trade_capital = 0.0
    if opens1payoff > 0 and opens2payoff > 0:
        trade_capital = abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0:
        trade_capital = abs(opens1payoff) + trCost * abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0:
        trade_capital = trCost * abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0:
        trade_capital = trCost * abs(opens1payoff) + trCost * abs(opens2payoff)

    return (Profit, Count, opentime, closetime, trade_capital, ibeta0, ibeta1, CrossTime)


# 參數設定
form_len = 150  # 估計期      #150
form_del_min = 16
OpenS = 1.5  # 開倉門檻x倍標準差
StopS = 100000.0  # 強迫平倉倍數x倍標準差(無限大=沒有強迫平倉) (與喬登對齊)
Cost = 0.0000278+0.0000051  # 交易成本
CostS = 0.0000278+0.0000051  # 開倉要大過的門檻
FinalOpen = 1000  # 最後開倉時間 #250
Maxi = 5  # 最大張數
trCost = 1  # 放空保證金
# Slippage = 1                               #滑價設定 0=t點訊號t點交易，1=t點訊號t+1點交易
CrossScreeningSet = 0
trade_time = 390  # 總開盤時間
capital = 50000000
ErrorLog = []

# 讀取日期


years = ['2018']
months = ['10', "11", '12']
days = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]

OpenS_profit = np.zeros((4, 24))
OpenS_tradenum = np.zeros((4, 24))
OpenS_closerate = np.zeros((4, 24))
OpenS_winrate = np.zeros((4, 24))
OpenS_count = 0

allcross = 0
alltrade = 0
ADF = 0


# ── 日期 CSV 快取 ────────────────────────────────────────────────────────────
# 回測按時間序跑，同一天的數千個配對會連續呼叫 real_env_reward。
# 若每次都重新 pd.read_csv，I/O 是最大瓶頸（3.7M 筆 × 2 個 CSV = 740 萬次讀檔）。
# 改用 dict 快取：每個交易日只讀一次磁碟，後續直接從 RAM 取用，
# 約 500 個交易日，formation table 每日幾 MB，price data 每日 10~50 MB，
# 合計記憶體佔用可接受（< 30 GB）。
# key = "YYYYMMDD"，value = 未篩選的原始 DataFrame（各 call 自行篩選，不影響 cache）

_cache_lock = threading.Lock()     # 多執行緒同時到同一天 cache miss 時，只讓一個 thread 做 I/O
_formation_table_cache: dict = {}  # { "YYYYMMDD": pd.DataFrame }
_price_data_cache: dict = {}       # { "YYYYMMDD": pd.DataFrame }


def _get_formation_table(year: str, month: str, day: str) -> pd.DataFrame:
    """Formation table 快取讀取：同一天只做一次磁碟 I/O，後續從 RAM 回傳。"""
    key = f"{year}{month}{day}"
    if key not in _formation_table_cache:
        with _cache_lock:
            if key not in _formation_table_cache:  # double-check：搶到 lock 後再確認一次
                path = os.path.join(
                    formation_table_dir,
                    f'{year}{month}{day}for150del16_AB.csv'
                )
                _formation_table_cache[key] = pd.read_csv(path, index_col=False)
    return _formation_table_cache[key]


def _get_price_data(year: str, month: str, day: str) -> pd.DataFrame:
    """分鐘股價快取讀取：同一天只做一次磁碟 I/O，後續從 RAM 回傳。"""
    key = f"{year}{month}{day}"
    if key not in _price_data_cache:
        with _cache_lock:
            if key not in _price_data_cache:  # double-check：搶到 lock 後再確認一次
                path = os.path.join(
                    r'/content/GGWP_US_LLTL/data/full_data_AB',
                    f'{year}-{month}-{day}_AB.csv'
                )
                _price_data_cache[key] = pd.read_csv(path, index_col=False)
    return _price_data_cache[key]


# ─────────────────────────────────────────────────────────────────────────────

def real_env_reward(stock_a_ind, stock_b_ind, date, actions, formation_params=None):
    date = str(date)
    OpenS = actions[0]
    StopS = actions[1]
    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    MinSPriceRawData = _get_price_data(year, month, day)

    if formation_params is not None:
        # 快速路徑：使用預先建立的 lookup dict，跳過 pandas 篩選（消除主要 GIL 瓶頸）
        w1, w2, lag_q, JCI_ModelType, Johansen_intcept, Johansen_slope, Johansen_std = formation_params
        CapitalWeight = np.expand_dims(np.array([w1, w2]), axis=0)
    else:
        # 原始路徑（backward compatible，未提供 formation_params 時使用）
        FormationTab = _get_formation_table(year, month, day)
        FormationTab = FormationTab[FormationTab['S1']==stock_a_ind]
        FormationTab = FormationTab[FormationTab["S2"]==stock_b_ind]
        si = FormationTab.index[0]
        CapitalWeight = np.expand_dims(np.array([FormationTab["w1"][si], FormationTab["w2"][si]]), axis=0)
        lag_q = int(FormationTab['VECM(q)'][si])
        JCI_ModelType = int(FormationTab['VECM_Model_Type'][si])
        Johansen_intcept = FormationTab["Johansen_intercept"][si]
        Johansen_slope = FormationTab["Johansen_slope"][si]
        Johansen_std = FormationTab["Johansen_std"][si]

    if JCI_ModelType > 3:
        raise ValueError("Model 4, 5 are not used")

    if Johansen_std < 0:
        # 如果std為負值，放棄該配對，並且把天數、股票代號、std記下來
        raise ValueError("Johansen_std < 0")

    if (Johansen_std * OpenS > CostS):
        # 先選出 2 欄再算 log，不對全部股票做 np.log（節省約 200x 計算量）
        rawS = MinSPriceRawData[[stock_a_ind, stock_b_ind]].to_numpy()
        rawLS = np.log(rawS)  # shape: (num_minutes, 2)

        m_rawLS = np.array(rawLS)
        m_CapitalWeight = np.array(CapitalWeight)
        cy = np.matmul(m_rawLS,m_CapitalWeight.T)
        cy = np.squeeze(cy)
        cy_mean_temp1 = Johansen_intcept + Johansen_slope * np.linspace(-form_del_min, -1,
                                                                        form_del_min)
        cy_mean_temp2 = Johansen_intcept + Johansen_slope * np.linspace(0,
                                                                        trade_time - form_del_min - 2,
                                                                        trade_time - form_del_min)
        cy_mean = np.hstack((cy_mean_temp1, cy_mean_temp2))

        Profit_13, Count_13, opentime, closetime, trade_capital, beta0, beta1, CrossTime = trade(
            cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost,
            OpenS, StopS, FinalOpen, capital, trCost)
        TradingResult = dict()
        TradingResult["reward"] = Profit_13[0, 0]
        TradingResult["close_timing"] = closetime
        TradingResult["open_timing"]  = opentime
        if Count_13[0, 1]:
            TradingResult["record"] = 666
        elif Count_13[0, 2]:
            TradingResult["record"] = -2
        elif Count_13[0, 3]:
            TradingResult["record"] = -4
        elif Count_13[0, 4]:
            TradingResult["record"] = -4
        elif beta0 == 0 and beta1 == 0:
            TradingResult["record"] = 0
        else:
            TradingResult["record"] = 0

        TradingResult["capital"]    = trade_capital
        TradingResult['stock1_num'] = beta0
        TradingResult['stock2_num'] = beta1
        TradingResult['w1'] = CapitalWeight[0, 0]
        TradingResult['w2'] = CapitalWeight[0, 1]

        return TradingResult

    else:
        raise ValueError("Johansen_std * OpenS < CostS and not caught by backtest.py")




