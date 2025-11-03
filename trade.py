# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import MyTool as mt
import time
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from hyperparameters import *

def trade(cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS, FinalOpen,
          capital, trCost):
    # [總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1, 5))
    # [開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1, 5))
    UpOpenTrend = cy_mean + Johansen_std * OpenS
    UpStopTrend = cy_mean + Johansen_std * StopS
    DownOpenTrend = cy_mean - Johansen_std * OpenS
    DownStopTrend = cy_mean - Johansen_std * StopS
    Position = 0  # 部位控制
    Ibeta = [0, 0]
    IntNum = [0, 0]
    LogTradeTime = np.zeros((1, cy.shape[0]))  # 時間紀錄
    openP = 0
    ForceP = 0
    opencount = 0
    opentime = 0
    closetime = 0
    opens1payoff = 0
    opens2payoff = 0
    closes1payoff = 0
    closes2payoff = 0
    CrossTime = 0

    for ti in range(form_del_min + 1, form_del_min + form_len - 1):
        if (cy[ti - 1] < cy_mean[ti] < cy[ti + 1]) or (cy[ti - 1] > cy_mean[ti] > cy[ti + 1]):  # cross over mean
            CrossTime += 1

    for ti in range(form_len + form_del_min, cy.shape[0] + 1):  # between trading time
        # 尾盤的強制平倉處理
        if ti == cy.shape[0]:
            # 若有倉則強制平倉
            if Position == 1:  # 若有多倉強制平多倉
                closes1payoff = Position * rawS[-1, 0] * Ibeta[0]
                closes2payoff = Position * rawS[-1, 1] * Ibeta[1]
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] = Count[0, 3] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
                elif ForceP <= 0:
                    Profit[0, 4] = ForceP
                    Count[0, 4] = Count[0, 4] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
            elif Position == -1:  # 若有空倉強制平空倉
                closes1payoff = Position * rawS[-1, 0] * Ibeta[0]
                closes2payoff = Position * rawS[-1, 1] * Ibeta[1]
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] = Count[0, 3] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
                elif ForceP <= 0:
                    Profit[0, 4] = ForceP
                    Count[0, 4] = Count[0, 4] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
        # 尾盤前的交易
        else:
            # 限制最大開倉次數，opencount<=1，強制每配對至多開倉一次
            if opencount <= 1:
                # 到期前若碰到平倉門檻且有多倉，平多倉
                if Position == 1 and cy[ti] >= cy_mean[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('inf'):
                    closes1payoff = Position * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = Position * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1] = Count[0, 1] + 1
                    Position = 0
                    LogTradeTime[0, ti] = -1  # note:trade time
                    closetime = ti
                # 到期前若碰到平倉門檻且有空倉，平空倉
                elif Position == -1 and cy[ti] <= cy_mean[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('-inf'):
                    closes1payoff = Position * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = Position * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1] = Count[0, 1] + 1
                    Position = 0
                    LogTradeTime[0, ti] = 1  # note:trade time
                    closetime = ti

                # 到期前若碰到停損門檻且有多倉，停損
                elif Position == 1 and cy[ti] <= DownStopTrend[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('-inf'):
                    closes1payoff = Position * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = Position * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2] = Count[0, 2] + 1
                    Position = -10  # 強制每配對至多開倉一次
                    LogTradeTime[0, ti] = -2
                    closetime = ti

                # 到期前若碰到停損門檻且有空倉，停損
                elif Position == -1 and cy[ti] >= UpStopTrend[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('inf'):
                    closes1payoff = Position * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = Position * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2] = Count[0, 2] + 1
                    Position = -10  # 強制每配對至多開倉一次
                    LogTradeTime[0, ti] = -2
                    closetime = ti

                # 到期前，若碰到下開倉門檻、無倉，開多倉
                elif Position == 0 and cy[ti] <= DownOpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[
                    ti] != float('-inf'):
                    Position = 1
                    Ibeta[0], Ibeta[1] = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1], rawS[ti, 0],
                                                       rawS[ti, 1], Maxi, capital)
                    opens1payoff = -Position * rawS[ti, 0] * Ibeta[0]
                    opens2payoff = -Position * rawS[ti, 1] * Ibeta[1]
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] = Count[0, 0] + 1
                    LogTradeTime[0, ti] = 1
                    opencount += 1
                    opentime = ti

                # 到期前，若碰到上開倉門檻、無倉，開空倉
                elif Position == 0 and cy[ti] >= UpOpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[
                    ti] != float('inf'):
                    Position = -1
                    Ibeta[0], Ibeta[1] = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1], rawS[ti, 0],
                                                       rawS[ti, 1], Maxi, capital)
                    opens1payoff = -Position * rawS[ti, 0] * Ibeta[0]
                    opens2payoff = -Position * rawS[ti, 1] * Ibeta[1]
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] = Count[0, 0] + 1
                    LogTradeTime[0, ti] = -1
                    opencount += 1
                    opentime = ti
            else:
                break
    Profit[0, 0] = sum(Profit[0, 1:5])
    trade_capital = 0
    if opens1payoff > 0 and opens2payoff > 0:
        trade_capital = abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0:
        trade_capital = abs(opens1payoff) + trCost * abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0:
        trade_capital = trCost * abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0:
        trade_capital = trCost * abs(opens1payoff) + trCost * abs(opens2payoff)

    return [Profit, Count, opentime, closetime, trade_capital, Ibeta, CrossTime]


def trade_down(cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS,
               FinalOpen, capital, trCost):
    # [總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1, 5))
    # [開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1, 5))
    OpenTrend = cy_mean + Johansen_std * OpenS
    StopTrend = cy_mean + Johansen_std * StopS
    Position = 0  # 部位控制
    Ibeta = [0, 0]
    IntNum = [0, 0]
    LogTradeTime = np.zeros((1, cy.shape[0]))  # 時間紀錄
    openP = 0
    ForceP = 0
    opencount = 0
    opentime = 0
    closetime = 0
    LongOrShort = -1
    opens1payoff = 0
    opens2payoff = 0
    closes1payoff = 0
    closes2payoff = 0
    CrossTime = 0

    for ti in range(form_del_min + 1, form_del_min + form_len - 1):
        if (cy[ti - 1] < cy_mean[ti] < cy[ti + 1]) or (cy[ti - 1] > cy_mean[ti] > cy[ti + 1]):
            CrossTime += 1

    for ti in range(form_del_min + form_len, cy.shape[0] + 1):
        # 尾盤的強制平倉處理
        if ti == cy.shape[0]:
            # 若有倉則強制平倉
            if Position == 1:
                closes1payoff = LongOrShort * rawS[-1, 0] * Ibeta[0]
                closes2payoff = LongOrShort * rawS[-1, 1] * Ibeta[1]
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] = Count[0, 3] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
                elif ForceP <= 0:
                    Profit[0, 4] = ForceP
                    Count[0, 4] = Count[0, 4] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
        # 尾盤前的交易
        else:
            if opencount <= 1:
                # 到期前若碰到平倉門檻且有倉，平倉
                if Position == 1 and cy[ti] <= cy_mean[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('-inf'):
                    closes1payoff = LongOrShort * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = LongOrShort * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1] = Count[0, 1] + 1
                    Position = 0
                    LogTradeTime[0, ti] = -1
                    closetime = ti

                # 到期前若碰到停損門檻且有倉，停損
                elif Position == 1 and cy[ti] >= StopTrend[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('inf'):
                    closes1payoff = LongOrShort * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = LongOrShort * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2] = Count[0, 2] + 1
                    Position = -1  # 強制每配對至多開倉一次
                    LogTradeTime[0, ti] = -2
                    closetime = ti

                # 到期前，若碰到開倉門檻、無倉、之前未開倉過，開倉
                elif Position == 0 and cy[ti] >= OpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[
                    ti] != float('inf'):
                    Position = 1
                    Ibeta[0], Ibeta[1] = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1], rawS[ti, 0],
                                                       rawS[ti, 1], Maxi, capital)
                    opens1payoff = -LongOrShort * rawS[ti, 0] * Ibeta[0]
                    opens2payoff = -LongOrShort * rawS[ti, 1] * Ibeta[1]
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] = Count[0, 0] + 1
                    LogTradeTime[0, ti] = 1
                    opencount += 1
                    opentime = ti
            else:
                break
    Profit[0, 0] = sum(Profit[0, 1:5])
    trade_capital = 0
    if opens1payoff > 0 and opens2payoff > 0:
        trade_capital = abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0:
        trade_capital = abs(opens1payoff) + trCost * abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0:
        trade_capital = trCost * abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0:
        trade_capital = trCost * abs(opens1payoff) + trCost * abs(opens2payoff)

    return [Profit, Count, opentime, closetime, trade_capital, Ibeta, CrossTime]


def trade_up(cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS,
             FinalOpen, capital, trCost):
    # [總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1, 5))
    # [開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1, 5))
    OpenTrend = cy_mean - Johansen_std * OpenS  # 下開倉
    StopTrend = cy_mean - Johansen_std * StopS
    Position = 0  # 部位控制
    Ibeta = [0, 0]
    IntNum = [0, 0]
    LogTradeTime = np.zeros((1, cy.shape[0]))  # 時間紀錄
    openP = 0
    ForceP = 0
    opencount = 0
    opentime = 0
    closetime = 0
    LongOrShort = 1
    opens1payoff = 0
    opens2payoff = 0
    closes1payoff = 0
    closes2payoff = 0
    CrossTime = 0

    for ti in range(form_del_min + 1, form_del_min + form_len):
        if (cy[ti - 1] < cy_mean[ti] < cy[ti + 1]) or (cy[ti - 1] > cy_mean[ti] > cy[ti + 1]):
            CrossTime += 1

    for ti in range(form_del_min + form_len, cy.shape[0] + 1):
        # 尾盤的強制平倉處理
        if ti == cy.shape[0]:
            # 若有倉則強制平倉
            if Position == 1:
                closes1payoff = LongOrShort * rawS[-1, 0] * Ibeta[0]
                closes2payoff = LongOrShort * rawS[-1, 1] * Ibeta[1]
                ForceP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP

                if ForceP > 0:
                    Profit[0, 3] = ForceP
                    Count[0, 3] = Count[0, 3] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
                elif ForceP <= 0:
                    Profit[0, 4] = ForceP
                    Count[0, 4] = Count[0, 4] + 1
                    Position = 0
                    LogTradeTime[0, ti - 1] = 3
                    closetime = cy.shape[0]
        # 尾盤前的交易
        else:
            if opencount <= 1:
                # 到期前若碰到平倉門檻且有倉，平倉
                if Position == 1 and cy[ti] >= cy_mean[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('inf'):
                    closes1payoff = LongOrShort * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = LongOrShort * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 1] = CloseP
                    Count[0, 1] = Count[0, 1] + 1
                    Position = 0
                    LogTradeTime[0, ti] = -1
                    closetime = ti

                # 到期前若碰到停損門檻且有倉，停損
                elif Position == 1 and cy[ti] <= StopTrend[ti] and ti < cy.shape[0] - 1 and cy[ti] != float('-inf'):
                    closes1payoff = LongOrShort * rawS[ti, 0] * Ibeta[0]
                    closes2payoff = LongOrShort * rawS[ti, 1] * Ibeta[1]
                    CloseP = mt.tax(closes1payoff, Cost) + mt.tax(closes2payoff, Cost) + openP
                    Profit[0, 2] = CloseP
                    Count[0, 2] = Count[0, 2] + 1
                    Position = -1  # 強制每配對至多開倉一次
                    LogTradeTime[0, ti] = -2
                    closetime = ti

                # 到期前，若碰到開倉門檻、無倉、之前未開倉過，開倉
                elif Position == 0 and cy[ti] <= OpenTrend[ti] and ti < FinalOpen and opencount != 1 and cy[
                    ti] != float('-inf'):
                    Position = 1
                    Ibeta[0], Ibeta[1] = mt.num_weight(CapitalWeight[0, 0], CapitalWeight[0, 1], rawS[ti, 0],
                                                       rawS[ti, 1], Maxi, capital)
                    opens1payoff = -LongOrShort * rawS[ti, 0] * Ibeta[0]
                    opens2payoff = -LongOrShort * rawS[ti, 1] * Ibeta[1]
                    openP = mt.tax(opens1payoff, Cost) + mt.tax(opens2payoff, Cost)
                    Count[0, 0] = Count[0, 0] + 1
                    LogTradeTime[0, ti] = 1
                    opencount += 1
                    opentime = ti
            else:
                break
    Profit[0, 0] = sum(Profit[0, 1:5])
    trade_capital = 0
    if opens1payoff > 0 and opens2payoff > 0:
        trade_capital = abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0:
        trade_capital = abs(opens1payoff) + trCost * abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0:
        trade_capital = trCost * abs(opens1payoff) + abs(opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0:
        trade_capital = trCost * abs(opens1payoff) + trCost * abs(opens2payoff)

    return [Profit, Count, opentime, closetime, trade_capital, Ibeta, CrossTime]


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


def real_env_reward(stock_a_ind, stock_b_ind, date, actions):
    date = str(date)
    OpenS = actions[0]
    StopS = actions[1]
    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    FormationTabPath = r'C:\Users\nycu_dev1\Desktop\Stock\formationtable'
    MinSPricePath = r'C:\Users\nycu_dev1\Desktop\Stock\full_data_AB'
    FormationTab = pd.read_csv(
        os.path.join(FormationTabPath, '{}{}{}for150del16_AB.csv'.format(year, month, day)),
        index_col=False)
    MinSPrice = pd.read_csv(os.path.join(MinSPricePath, '{}-{}-{}_AB.csv'.format(year, month, day)),
                            index_col=False)
    #print("Now import: {}-{}-{}".format(year, month, day))
    FormationTab = FormationTab.reset_index(inplace=False)

    MinSPriceRawData = MinSPrice
    MinSPriceRawData = MinSPriceRawData.reset_index(inplace=False)
    MinSPriceRawData = MinSPriceRawData.drop(['index'], axis=1)
    LogPrice = np.log(MinSPriceRawData)

    # 依據formation table讀取資金權重、最適lag_q、VECM modeltype、股價資料
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

        rawS = MinSPriceRawData[[stock_a_ind, stock_b_ind]]
        rawS = rawS.to_numpy()
        rawLS = LogPrice[[stock_a_ind, stock_b_ind]]
        rawLS = rawLS.to_numpy()

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

        [Profit_13, Count_13, opentime, closetime, trade_capital, TradeNum, CrossTime] = trade(
            cy, cy_mean, rawS, form_len, form_del_min, CapitalWeight, Maxi, Johansen_std, Cost,
            OpenS, StopS, FinalOpen, capital, trCost)
        TradingResult = dict()
        TradingResult["reward"] = Profit_13[0,0]
        TradingResult["close_timing"] = closetime
        TradingResult["open_timing"] = opentime
        if Count_13[0,1]:
            # NC
            TradingResult["record"] = 666
        elif Count_13[0,2]:
            # SL
            TradingResult["record"] = -2
        elif Count_13[0,3]:
            # EX and profit
            TradingResult["record"] = -4
        elif Count_13[0,4]:
            # EX and loss
            TradingResult["record"] = -4
        elif TradeNum[0]==0 and TradeNum[1]==0:
            # capital weight differs too much so does not open
            TradingResult["record"] = 0
        else:
            #TODO: check what the hell is going on here
            TradingResult["record"] = 0

        TradingResult["capital"] = trade_capital
        TradingResult['stock1_num'] = TradeNum[0]
        TradingResult['stock2_num'] = TradeNum[1]
        TradingResult['w1'] = CapitalWeight[0,0]
        TradingResult['w2'] = CapitalWeight[0,1]

        return TradingResult

    else:
        raise ValueError("Johansen_std * OpenS < CostS and not caught by backtest.py")




