# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:02:57 2024

@author: nycu_dev1
"""

import pandas as pd
import numpy as np
import numpy.matlib
import math
from scipy.linalg import eigh
import MyTool as mt
from statsmodels.tsa.api import VAR
import time
import os 
import csv
from numba import njit
#import matplotlib.pyplot as plt

def append_debug_log(filepath, row_dict):
    # 建立資料夾
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    file_exists = os.path.isfile(filepath)

    if file_exists:
        # 檔案已存在：以「既有的header」為準，不要用這次呼叫的row_dict.keys()
        # 重新決定欄位順序/數量。訓練中途斷線、換新版程式碼接續寫入同一個檔案時，
        # 如果row_dict的欄位變了（例如新增了epoch），但header是舊版寫的，
        # 兩者不一致會導致同一個CSV檔案裡欄位數量前後不同，事後用pandas讀取會報錯。
        with open(filepath, 'r', newline='') as f:
            fieldnames = next(csv.reader(f))
    else:
        fieldnames = list(row_dict.keys())

    # 寫入 CSV
    with open(filepath, 'a', newline='') as csvfile:
        # extrasaction='ignore'：row_dict裡有header沒有的欄位就直接忽略，不報錯
        # restval=''：header有的欄位但row_dict沒提供，就補空字串，維持欄位數量一致
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                 extrasaction='ignore', restval='')

        # 第一次寫入 → 寫欄位名稱
        if not file_exists:
            writer.writeheader()

        # 寫入資料
        writer.writerow(row_dict)

def trade(cy, cy_mean, rawS, form_len, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS, FinalOpen, capital, CrossScreeningSet, trCost):
    
    #[總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1,5))
    #[開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1,5))
    UpOpenTrend = cy_mean + Johansen_std * OpenS
    UpStopTrend = cy_mean + Johansen_std * StopS
    DownOpenTrend = cy_mean - Johansen_std * OpenS
    DownStopTrend = cy_mean - Johansen_std * StopS
    Position = 0 # 部位控制
    Ibeta = [0, 0]
    IntNum = [0, 0]
    LogTradeTime = np.zeros((1,cy.shape[0])) # 時間紀錄
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
    
    for ti in range(1, form_len):
        if (cy[ti-1] < cy_mean[ti] < cy[ti+1]) or (cy[ti-1] > cy_mean[ti] > cy[ti+1]):
            CrossTime += 1
    
    if CrossTime >= CrossScreeningSet:
        for ti in range(form_len, cy.shape[0]+1):
            #尾盤的強制平倉處理
            if ti == cy.shape[0]:
                #若有倉則強制平倉
                if Position == 1: #若有多倉強制平多倉
                    closes1payoff = Position * rawS[-1,0] * Ibeta[0]
                    closes2payoff = Position * rawS[-1,1] * Ibeta[1]
                    ForceP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                    if ForceP > 0:
                        Profit[0,3] = ForceP
                        Count[0,3] = Count[0,3] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
                    elif ForceP <= 0:
                        Profit[0,4] = ForceP
                        Count[0,4] = Count[0,4] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
                elif Position == -1: #若有空倉強制平空倉
                    closes1payoff = Position * rawS[-1,0] * Ibeta[0]
                    closes2payoff = Position * rawS[-1,1] * Ibeta[1]
                    ForceP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                    if ForceP > 0:
                        Profit[0,3] = ForceP
                        Count[0,3] = Count[0,3] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
                    elif ForceP <= 0:
                        Profit[0,4] = ForceP
                        Count[0,4] = Count[0,4] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
            #尾盤前的交易
            else:
                #限制最大開倉次數，opencount<=1，強制每配對至多開倉一次
                if opencount <= 1 :
                    #到期前若碰到平倉門檻且有多倉，平多倉
                    if Position == 1 and cy[ti]>=cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('inf'):
                        closes1payoff = Position * rawS[ti,0] * Ibeta[0]
                        closes2payoff = Position * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,1] = CloseP
                        Count[0,1] = Count[0,1] + 1
                        Position = 0
                        LogTradeTime[0,ti] = -1
                        closetime = ti
                        
                    #到期前若碰到平倉門檻且有空倉，平空倉
                    elif Position == -1 and cy[ti]<=cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('-inf'):
                        closes1payoff = Position * rawS[ti,0] * Ibeta[0]
                        closes2payoff = Position * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,1] = CloseP
                        Count[0,1] = Count[0,1] + 1
                        Position = 0
                        LogTradeTime[0,ti] = 1
                        closetime = ti
                        
                    #到期前若碰到停損門檻且有多倉，停損    
                    elif Position == 1 and cy[ti]<=DownStopTrend[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('-inf'):
                        closes1payoff = Position * rawS[ti,0] * Ibeta[0]
                        closes2payoff = Position * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,2] = CloseP
                        Count[0,2] = Count[0,2] + 1
                        Position = -10           #強制每配對至多開倉一次
                        LogTradeTime[0,ti] = -2
                        closetime = ti
                        
                    #到期前若碰到停損門檻且有空倉，停損    
                    elif Position == -1 and cy[ti]>=UpStopTrend[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('inf'):
                        closes1payoff = Position * rawS[ti,0] * Ibeta[0]
                        closes2payoff = Position * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,2] = CloseP
                        Count[0,2] = Count[0,2] + 1
                        Position = -10           #強制每配對至多開倉一次
                        LogTradeTime[0,ti] = -2
                        closetime = ti
                        
                    #到期前，若碰到下開倉門檻、無倉，開多倉    
                    elif Position == 0 and cy[ti]<=DownOpenTrend[ti] and ti<FinalOpen and opencount != 1 and cy[ti]!=float('-inf') and CrossTime >= CrossScreeningSet:
                         Position = 1
                         Ibeta[0] , Ibeta[1] = mt.num_weight(CapitalWeight[0,0],CapitalWeight[0,1],rawS[ti,0],rawS[ti,1],Maxi,capital)
                         opens1payoff = -Position * rawS[ti, 0] * Ibeta[0]
                         opens2payoff = -Position * rawS[ti, 1] * Ibeta[1]
                         openP = mt.tax(opens1payoff,Cost) + mt.tax(opens2payoff,Cost)
                         Count[0,0] = Count[0,0] + 1
                         LogTradeTime[0,ti] = 1
                         opencount += 1
                         opentime = ti
                         
                    #到期前，若碰到上開倉門檻、無倉，開空倉  
                    elif Position == 0 and cy[ti]>=UpOpenTrend[ti] and ti<FinalOpen and opencount != 1 and cy[ti]!=float('inf') and CrossTime >= CrossScreeningSet:
                        Position = -1
                        Ibeta[0] , Ibeta[1] = mt.num_weight(CapitalWeight[0,0],CapitalWeight[0,1],rawS[ti,0],rawS[ti,1],Maxi,capital)
                        opens1payoff = -Position * rawS[ti, 0] * Ibeta[0]
                        opens2payoff = -Position * rawS[ti, 1] * Ibeta[1]
                        openP = mt.tax(opens1payoff,Cost) + mt.tax(opens2payoff,Cost)
                        Count[0,0] = Count[0,0] + 1
                        LogTradeTime[0,ti] = -1
                        opencount += 1
                        opentime = ti
                else:
                    break
    Profit[0,0]=sum(Profit[0,1:5])
    trade_capital = 0
    if opens1payoff > 0 and  opens2payoff > 0:
        trade_capital = abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0 :
        trade_capital = abs(opens1payoff)+trCost*abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0 :
        trade_capital = trCost*abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0 :
        trade_capital = trCost*abs(opens1payoff)+trCost*abs(opens2payoff)
                     
    return [Profit, Count,opentime,closetime,trade_capital,Ibeta,CrossTime]
    
def trade_down(cy, cy_mean, rawS, form_len, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS, 
               FinalOpen, capital, CrossScreeningSet,trCost):
    
    #[總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1,5))
    #[開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1,5))
    OpenTrend = cy_mean + Johansen_std * OpenS
    StopTrend = cy_mean + Johansen_std * StopS
    Position = 0 # 部位控制
    Ibeta = [0, 0]
    IntNum = [0, 0]
    LogTradeTime = np.zeros((1,cy.shape[0])) # 時間紀錄
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
    
    for ti in range(1, form_len):
        if (cy[ti-1] < cy_mean[ti] < cy[ti+1]) or (cy[ti-1] > cy_mean[ti] > cy[ti+1]):
            CrossTime += 1
            
    if CrossTime >= CrossScreeningSet:
        for ti in range(form_len, cy.shape[0]+1):
            #尾盤的強制平倉處理
            if ti == cy.shape[0]:
                #若有倉則強制平倉
                if Position == 1:
                    closes1payoff = LongOrShort * rawS[-1,0] * Ibeta[0]
                    closes2payoff = LongOrShort * rawS[-1,1] * Ibeta[1]
                    ForceP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                    if ForceP > 0:
                        Profit[0,3] = ForceP
                        Count[0,3] = Count[0,3] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
                    elif ForceP <= 0:
                        Profit[0,4] = ForceP
                        Count[0,4] = Count[0,4] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
            #尾盤前的交易
            else:
                if opencount <= 1 :
                    #到期前若碰到平倉門檻且有倉，平倉
                    if Position == 1 and cy[ti] <= cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('-inf'):
                        closes1payoff = LongOrShort * rawS[ti,0] * Ibeta[0]
                        closes2payoff = LongOrShort * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,1] = CloseP
                        Count[0,1] = Count[0,1] + 1
                        Position = 0 
                        LogTradeTime[0,ti] = -1
                        closetime = ti
                        
                    #到期前若碰到停損門檻且有倉，停損    
                    elif Position == 1 and cy[ti]>=StopTrend[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('inf'):
                        closes1payoff = LongOrShort * rawS[ti,0] * Ibeta[0]
                        closes2payoff = LongOrShort * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,2] = CloseP
                        Count[0,2] = Count[0,2] + 1
                        Position = -1           #強制每配對至多開倉一次
                        LogTradeTime[0,ti] = -2
                        closetime = ti
                    
                    #到期前，若碰到開倉門檻、無倉、之前未開倉過，開倉    
                    elif Position == 0 and cy[ti]>=OpenTrend[ti] and ti<FinalOpen and opencount != 1 and cy[ti]!=float('inf') and CrossTime >= CrossScreeningSet:
                        Position = 1
                        Ibeta[0] , Ibeta[1] = mt.num_weight(CapitalWeight[0,0],CapitalWeight[0,1],rawS[ti,0],rawS[ti,1],Maxi,capital)
                        opens1payoff = -LongOrShort * rawS[ti, 0] * Ibeta[0]
                        opens2payoff = -LongOrShort * rawS[ti, 1] * Ibeta[1]
                        openP = mt.tax(opens1payoff,Cost) + mt.tax(opens2payoff,Cost)
                        Count[0,0] = Count[0,0] + 1
                        LogTradeTime[0,ti] = 1
                        opencount += 1
                        opentime = ti
                else:
                    break
    Profit[0,0]=sum(Profit[0,1:5])
    trade_capital = 0
    if opens1payoff > 0 and  opens2payoff > 0:
        trade_capital = abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0 :
        trade_capital = abs(opens1payoff)+trCost*abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0 :
        trade_capital = trCost*abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0 :
        trade_capital = trCost*abs(opens1payoff)+trCost*abs(opens2payoff)
        
    return [Profit, Count,opentime,closetime,trade_capital,Ibeta,CrossTime]

def trade_up(cy, cy_mean, rawS, form_len, CapitalWeight, Maxi, Johansen_std, Cost, OpenS, StopS, 
             FinalOpen, capital, CrossScreeningSet,trCost):
    #[總獲利,平倉獲利,停損獲利,換日強停獲利,換日強停虧損]
    Profit = np.zeros((1,5))
    #[開倉次數,平倉次數,停損次數,換日強停獲利次數,換日強停虧損次數]
    Count = np.zeros((1,5))
    OpenTrend = cy_mean - Johansen_std * OpenS #下開倉
    StopTrend = cy_mean - Johansen_std * StopS
    Position = 0 # 部位控制
    Ibeta = [0, 0]
    IntNum = [0, 0]
    LogTradeTime = np.zeros((1,cy.shape[0])) # 時間紀錄
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
    
    for ti in range(1, form_len):
        if (cy[ti-1] < cy_mean[ti] < cy[ti+1]) or (cy[ti-1] > cy_mean[ti] > cy[ti+1]):
            CrossTime += 1
            
    if CrossTime >= CrossScreeningSet:
        for ti in range(form_len, cy.shape[0]+1):
            #尾盤的強制平倉處理
            if ti == cy.shape[0]:
                #若有倉則強制平倉
                if Position == 1:
                    closes1payoff = LongOrShort * rawS[-1,0] * Ibeta[0]
                    closes2payoff = LongOrShort * rawS[-1,1] * Ibeta[1]
                    ForceP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                
                    if ForceP > 0:
                        Profit[0,3] = ForceP
                        Count[0,3] = Count[0,3] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
                    elif ForceP <= 0:
                        Profit[0,4] = ForceP
                        Count[0,4] = Count[0,4] + 1
                        Position = 0
                        LogTradeTime[0,ti-1] = 3
                        closetime = cy.shape[0]
            #尾盤前的交易
            else:
                if opencount <= 1 :
                    #到期前若碰到平倉門檻且有倉，平倉
                    if Position == 1 and cy[ti] >= cy_mean[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('inf'):
                        closes1payoff = LongOrShort * rawS[ti,0] * Ibeta[0]
                        closes2payoff = LongOrShort * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,1] = CloseP
                        Count[0,1] = Count[0,1] + 1
                        Position = 0 
                        LogTradeTime[0,ti] = -1
                        closetime = ti
                        
                    #到期前若碰到停損門檻且有倉，停損    
                    elif Position == 1 and cy[ti]<=StopTrend[ti] and ti < cy.shape[0]-1 and cy[ti]!=float('-inf'):
                        closes1payoff = LongOrShort * rawS[ti,0] * Ibeta[0]
                        closes2payoff = LongOrShort * rawS[ti,1] * Ibeta[1]
                        CloseP = mt.tax(closes1payoff,Cost) + mt.tax(closes2payoff,Cost) + openP
                        Profit[0,2] = CloseP
                        Count[0,2] = Count[0,2] + 1
                        Position = -1           #強制每配對至多開倉一次
                        LogTradeTime[0,ti] = -2
                        closetime = ti
                        
                    #到期前，若碰到開倉門檻、無倉、之前未開倉過，開倉    
                    elif Position == 0 and cy[ti]<=OpenTrend[ti] and ti<FinalOpen and opencount != 1 and cy[ti]!=float('-inf'):
                        Position = 1
                        Ibeta[0] , Ibeta[1] = mt.num_weight(CapitalWeight[0,0],CapitalWeight[0,1],rawS[ti,0],rawS[ti,1],Maxi,capital)
                        opens1payoff = -LongOrShort * rawS[ti, 0] * Ibeta[0]
                        opens2payoff = -LongOrShort * rawS[ti, 1] * Ibeta[1]
                        openP = mt.tax(opens1payoff,Cost) + mt.tax(opens2payoff,Cost)
                        Count[0,0] = Count[0,0] + 1
                        LogTradeTime[0,ti] = 1
                        opencount += 1
                        opentime = ti
                else:
                    break
    Profit[0,0]=sum(Profit[0,1:5])
    trade_capital = 0
    if opens1payoff > 0 and  opens2payoff > 0:
        trade_capital = abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff > 0 and opens2payoff < 0 :
        trade_capital = abs(opens1payoff)+trCost*abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff > 0 :
        trade_capital = trCost*abs(opens1payoff)+abs( opens2payoff)
    elif opens1payoff < 0 and opens2payoff < 0 :
        trade_capital = trCost*abs(opens1payoff)+trCost*abs(opens2payoff)

    return [Profit, Count,opentime,closetime,trade_capital,Ibeta,CrossTime]

def fixdatedata(FilesList_a, FilesList_b):
    #把檔名整理一下，準備把不要的檔案篩出來刪掉
    delFilesList_a , delFilesList_b = [] , []
    for di in range(len(FilesList_a)):
        delFilesList_a.append(FilesList_a[di][0:8])
        if di < len(FilesList_b):
            delFilesList_b.append(FilesList_b[di][0:8])
        else:
            delFilesList_b.append('nan')
    #對照a表，b表中填nan
    for di in range(len(FilesList_a)):
        if delFilesList_a[di] != delFilesList_b[di]:
            delFilesList_b.insert(di,'nan')
            del delFilesList_b[-1]
    #找出nan的index
    delFilesList = [i for i in range(len(delFilesList_b)) if delFilesList_b[i] == 'nan']
    
    #依據nan的index刪除a表
    for i in range(len(delFilesList)-1,-1,-1):
        FilesList_a.pop(delFilesList[i])

def Binal_comb(pool):
    # 排列組合不重複的數列，專門處理Cn取2
    if not(isinstance(pool,list)):
        return 'Error:must be list'
    n = len(pool)
    stemp = []
    if 2 > n:
        return 'Error:Beasue input less 2 iter'
    else:
        stepi = 0
        stepj = 0
        for stepi in range(n):
            for stepj in range(stepi):
                stemp.append([pool[stepj],pool[stepi]])
        stemp.sort()
        result = np.array(stemp)
        return result

def VAR_model( y , p ):    
    k = len(y.T)     # 幾檔股票
    n = len(y)       # 資料長度
    
    xt = np.ones( ( n-p , (k*p)+1 ) )
    for i in range(n-p):
        a = 1
        for j in range(p):
            a = np.hstack( (a,y[i+p-j-1]) )
        a = a.reshape([1,(k*p)+1])
        xt[i] = a
    
    zt = np.delete(y,np.s_[0:p],axis=0)
    xt = np.mat(xt)
    zt = np.mat(zt)

    beta = ( xt.T * xt ).I * xt.T * zt                      # 計算VAR的參數
    
    A = zt - xt * beta                                      # 計算殘差
    sigma = ( (A.T) * A ) / (n-p)                           # 計算殘差的共變異數矩陣
        
    return [ sigma , beta ]

# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數-----
def order_select( raw_y , max_p ):
    k = len(raw_y.T)     # 幾檔股票
    n = len(raw_y)       # 資料長度
    
    bic = np.zeros((max_p,1))
    for p in range(1,max_p+1):
        sigma = VAR_model( raw_y , p )[0]
        bic[p-1] = np.log( np.linalg.det(sigma) ) + np.log(n) * p * (k*k) / n
        
    bic_order = int(np.where(bic == np.min(bic))[0] + 1)        # 因為期數p從1開始，因此需要加1
    
    return bic_order

def JCI_AutoSelection(Row_Y,opt_q):
    #論文中的BIC model selection
    [NumObs, k] = Row_Y.shape
    opt_p = opt_q + 1
    Tl = NumObs - opt_p
    
    TraceTest_table = np.zeros([5, k])
    BIC_table = np.zeros([5, 1])
    BIC_List = np.ones([5, 1]) * np.Inf
    opt_model_num = 0
    for mr in range(0,5):
        tr_H, _, _, _, _, ut, _, _, _ = JCItest_TraceTest(Row_Y, mr+1, opt_q)
        #把結果存起來，True是拒絕，False是不拒絕，tr_H[0]是Rank0,tr_H[1]是Rank1
        TraceTest_table[mr,:] = tr_H
        #以下計算BIC，僅計算Rank1
        eps = np.mat(ut)
        sq_Res_r1 = eps.T * eps / Tl
        errorRes_r1 = eps * sq_Res_r1.I * eps.T
        trRes_r1 = np.trace(errorRes_r1)
        L = (-k*Tl*0.5)*np.log(2*np.pi) - (Tl*0.5)*np.log(np.linalg.det(sq_Res_r1)) -0.5*trRes_r1
        
        if mr==0:
            #alpha(k,1) + beta(k,1) + q*Gamma(k,k)
            deg_Fred = 2*k + opt_q*(k*k)
        elif mr==1:
            #alpha(k,1) + beta(k,1) + C0(1,1) + q*Gamma(k,k)
            deg_Fred = 2*k + 1 + opt_q*(k*k)
        elif mr==2:
            #alpha(k,1) + beta(k,1) + C0(1,1) + C1(k,1) + q*Gamma(k,k)
            deg_Fred = 3*k + 1 + opt_q*(k*k)
        elif mr==3:
            #alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + q*Gamma(k,k)
            deg_Fred = 3*k + 2 + opt_q*(k*k)
        elif mr==4:
            #alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + D1(k,1) + q*Gamma(k,k)
            deg_Fred = 4*k + 2 + opt_q*(k*k)
        #把Rank1各模型的BIC存起來
        BIC_table[mr] = -2*np.log(L) + deg_Fred*np.log(NumObs*k)
        
        #挑出被選的Rank1模型
        if TraceTest_table[mr,0] == 1 and TraceTest_table[mr,1] == 0 :
            #拒絕R0，不拒絕R1，該模型的最適Rank為R1，並把該模型與Rank1的BIC值存起來
            BIC_List[mr] = BIC_table[mr]
            opt_model_num += 1
        elif TraceTest_table[mr,0] == 0 and TraceTest_table[mr,1] == 0:
            #不拒絕R0，那R1應該是不用測，該模型的最適Rank為R0，紀錄為NaN
            continue
        elif TraceTest_table[mr,0] == 0 and TraceTest_table[mr,1] == 1:
            #不拒絕R0，那R1應該是不用測，該模型的最適Rank為R0，紀錄為NaN
            continue
        elif TraceTest_table[mr,0] == 1 and TraceTest_table[mr,1] == 1:
            #拒絕R0且拒絕R1，該模型的最適Rank為R2，紀錄為NaN
            continue
    
    BIC_List = BIC_List.tolist()
    #找出有紀錄的BIC中最小值，即為Opt_model，且Opt_model+1就對應我們的模型編號
    Opt_model = BIC_List.index(min(BIC_List))
    '''
    #分model1~3/4~5討論
    #model1~3
    if BIC_List[0] != [float('Infinity')] : 
        Opt_model_no_trend = 0
    if BIC_List[1] != [float('Infinity')] : 
        Opt_model_no_trend = 1
    if BIC_List[2] != [float('Infinity')]:
        Opt_model_no_trend = 2
    #model4~5
    Opt_model_trend = 0    
    if BIC_List[3] != [float('Infinity')] :
        Opt_model_trend = 3
    if BIC_List[4] != [float('Infinity')] :
        Opt_model_trend = 4
    if (Opt_model_trend  == 3 or Opt_model_trend  == 4):
        Opt_model = Opt_model_trend
    else:
        Opt_model =  Opt_model_no_trend
    '''
    if opt_model_num == 0:
        #如果opt_model_num是0，代表沒有最適模型或最適模型為Rank0
        return  0
    else:
        #如果opt_model_num不是0，則Opt_model+1模型的Rank1即為我們最適模型
        return Opt_model+1


def JCItest_TraceTest(X_data,model_type,lag_q):
    #trace test
    [NumObs,NumDim] = X_data.shape

    dY_ALL = X_data[1:, :] - X_data[0:-1, :] 
    dY = dY_ALL[lag_q:, :] #DY
    Ys = X_data[lag_q:-1, :] #Lag_Y
    
    #底下開始處理估計前的截距項與時間趨勢項
    if lag_q == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_q-1,1)) ) )#Lag_Y
        elif model_type == 3:
            dX = np.ones((NumObs-lag_q-1,1)) #DLag_Y
        elif model_type == 4:
            dX = np.ones((NumObs-lag_q-1,1)) #DLag_Y
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_q,1).reshape(NumObs-lag_q-1,1) ) )#Lag_Y
        elif model_type == 5:
            dX = np.hstack( ( np.ones((NumObs-lag_q-1,1)) , np.arange(1,NumObs-lag_q,1).reshape(NumObs-lag_q-1,1) ) )
    
    elif lag_q>0:
        dX = np.zeros([NumObs-lag_q-1, NumDim * lag_q]) #DLag_Y
        for xi in range(lag_q):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_q - xi -1 :NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_q-1,1)) ) )
        elif model_type == 3:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_q-1,1)) ) )
        elif model_type == 4:
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_q,1).reshape(NumObs-lag_q-1,1) ) )
            dX = np.hstack( ( dX, np.ones((NumObs-lag_q-1,1)) ) )
        elif model_type == 5:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_q-1,1)) , np.arange(1,NumObs-lag_q,1).reshape(NumObs-lag_q-1,1) ) )
    
    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    dX_2 = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    #python無法計算0矩陣的inverse，用判斷式處理
    if  np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_q-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_q-1) - dX * dX_2.I * dX.T
    
    R0, R1 = dY.T * M, Ys.T * M
    
    S00 = R0 * R0.T / (NumObs-lag_q-1)
    S01 = R0 * R1.T / (NumObs-lag_q-1)
    S10 = R1 * R0.T / (NumObs-lag_q-1)
    S11 = R1 * R1.T / (NumObs-lag_q-1)
    
    #計算廣義特徵值與廣義特徵向量
    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    
    # 排序特徵向量Eig_vector與特徵值lambda
    sort_ind = np.argsort(-eigValue_lambda)
    eigValue_lambda = eigValue_lambda[sort_ind]
   
    eigVecs = eigvecs[:, sort_ind]
    #將所有eigenvector同除第一行的總和，這是為了標準化參數
    eigVecs = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2])) 
    eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    jci_beta = eigVecs[:,0][0:2].reshape(NumDim,1)
    
    #Alpha
    a = np.mat(eigVecs[:,0])
    jci_alpha = S01 * a.T
    
    #初始化 c0, d0, c1, d1
    c0 , d0 = 0, 0
    c1 , d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])

    #計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        cvalue = [12.3329, 4.1475]
    elif model_type == 2:
        #c0 = eigVecs_st[-1, 0:1]
        c0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + numpy.matlib.repmat(c0, NumObs-lag_q-1, 1) )* jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        cvalue = [20.3032, 9.1465]
    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        cvalue = [15.4904, 3.8509]
    elif model_type == 4:
        #d0 = eigVecs_st[-1, 0:1]
        d0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + np.arange(1,NumObs-lag_q,1).reshape(NumObs-lag_q-1,1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        cvalue = [25.8863, 12.5142]
    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:,-1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
        cvalue = [18.3837, 3.8395]
    #計算殘差    
    ut = W - dX * P.T
    #VEC加總
    Ct_all = jci_alpha*c0 + c1 + jci_alpha*d0 +d1
    #計算VEC項, VEC滯後項加總, 殘差covariance matrix
    gamma = []
    for bi in range(1,lag_q+1):
        Bq = P[:, (bi-1)*NumDim : bi * NumDim]
        gamma.append(Bq)
    temp1 = np.dot(np.dot(jci_beta.transpose(),S11[0:2,0:2]),jci_beta)
    omega_hat = S00[0:2,0:2] - np.dot(np.dot(jci_alpha,temp1),jci_alpha.transpose())
    #把Ct統整在一起
    Ct=[]
    Ct.append(c0)
    Ct.append(d0)
    Ct.append(c1)
    Ct.append(d1)
    
    TraceTest_H = []
    TraceTest_T = []
    for rn in range(0,NumDim):
        eig_lambda = np.cumprod(1-eigValue_lambda[rn:NumDim,:])
        trace_stat = -2 * np.log(eig_lambda[-1] ** ((NumObs-lag_q-1)/2))
        TraceTest_H.append(cvalue[rn] < trace_stat)
        TraceTest_T.append(trace_stat)
    #回傳[H=0(拒絕共整合) ,stat值 , alpha, beat(cointegration matrix), VEC各參數, 殘差, VEC滯後項加總, VEC加總, covariance matrix]
    return TraceTest_H, TraceTest_T, jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat


def JCItestpara_spilCt(X_data,model_type,lag_p):
    if model_type == 'model1':
        model_type = 1
    elif model_type == 'model2':
        model_type = 2
    elif model_type == 'model3':
        model_type = 3
    [NumObs,NumDim] = X_data.shape

    dY_ALL = X_data[1:, :] - X_data[0:-1, :] 
    dY = dY_ALL[lag_p:, :] #DY
    Ys = X_data[lag_p:-1, :] #Lag_Y
    
    #底下開始處理估計前的截距項與時間趨勢項
    if lag_p == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )#Lag_Y
        elif model_type == 3:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
        elif model_type == 4:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )#Lag_Y
        elif model_type == 5:
            dX = np.hstack( ( np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    elif lag_p>0:
        dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p]) #DLag_Y
        for xi in range(lag_p):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p - xi -1 :NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 3:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 4:
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 5:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    dX_2 = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    #python無法計算0矩陣的inverse，用判斷式處理
    if  np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_p-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T
    
    R0, R1 = dY.T * M, Ys.T * M
    
    S00 = R0 * R0.T / (NumObs-lag_p-1)
    S01 = R0 * R1.T / (NumObs-lag_p-1)
    S10 = R1 * R0.T / (NumObs-lag_p-1)
    S11 = R1 * R1.T / (NumObs-lag_p-1)
    
    #計算廣義特徵值與廣義特徵向量
    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    
    # 排序特徵向量Eig_vector與特徵值lambda
    sort_ind = np.argsort(-eigValue_lambda)
    #eigValue_lambda = eigValue_lambda[sort_ind]
   
    eigVecs = eigvecs[:, sort_ind]
    #將所有eigenvector同除第一行的總和
    eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2])) 
   
    #eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    jci_beta = eigVecs_st[:,0][0:2].reshape(NumDim,1)

    #Alpha
    a = np.mat(eigVecs_st[:,0])
    jci_a = S01 * a.T
    jci_alpha = jci_a/np.sum(np.absolute(jci_a)) 
    
    #初始化 c0, d0, c1, d1
    c0 , d0 = 0, 0
    c1 , d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])
    #計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        
    elif model_type == 2:
        c0 = eigVecs_st[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + numpy.matlib.repmat(c0, NumObs-lag_p-1, 1) )* jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
    
    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
    
    elif model_type == 4:
        d0 = eigVecs_st[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        
    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:,-1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
    #計算殘差    
    ut = W - dX * P.T
    Ct_all = jci_alpha*c0 + c1 + jci_alpha*d0 +d1

    #計算VEC項
    gamma = []
    for bi in range(1,lag_p+1):
        Bq = P[:, (bi-1)*NumDim : bi * NumDim]
        gamma.append(Bq)
    temp1 = np.dot(np.dot(jci_beta.transpose(),S11[0:2,0:2]),jci_beta)
    omega_hat = S00[0:2,0:2] - np.dot(np.dot(jci_alpha,temp1),jci_alpha.transpose())
    #把Ct統整在一起
    Ct=[]
    Ct.append(c0)
    Ct.append(d0)
    Ct.append(c1)
    Ct.append(d1)
    return jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat

def Johansen_mean(alpha,beta,gamma,mu,lagp,NumDim=2): 
    #論文中的closed form mean
    #lagp指的是VECM的LAG期數
    sumgamma = np.zeros([NumDim, NumDim])
    for i in range(0,lagp):
        sumgamma =sumgamma+gamma[i]
    GAMMA = np.eye(NumDim) - sumgamma 
    #計算正交化的alpha,beta
    alpha_orthogonal = alpha.copy()  
    alpha_t = alpha.transpose()
    alpha_orthogonal[1,0] = (-(alpha_t[0,0]*alpha_orthogonal[0,0])) / alpha_t[0,1]    
    alpha_orthogonal = alpha_orthogonal/sum(abs(alpha_orthogonal))
    beta_orthogonal = beta.copy()  
    beta_t = beta.transpose()
    beta_orthogonal[1,0] = -((beta_t[0,0]*beta_orthogonal[0,0])) / beta_t[0,1]    
    beta_orthogonal = beta_orthogonal/sum(abs(beta_orthogonal)) 
    #計算MEAN
    temp1 = np.linalg.inv(np.dot(np.dot(alpha_orthogonal.transpose(), GAMMA),beta_orthogonal))
    C = np.dot(np.dot(beta_orthogonal,temp1),alpha_orthogonal.transpose())
    temp2 = np.linalg.inv(np.dot(alpha.transpose(),alpha))
    alpha_hat = np.dot(alpha,temp2)
    temp3 = np.dot(GAMMA,C) - np.eye(NumDim)
    C0 = np.mat(mu[0])
    C1 = np.mat(mu[2])
    D0 = np.mat(mu[1])
    D1 = np.mat(mu[3])
    C0 = alpha*C0 + C1 + alpha*D0 + D1
    Ct = alpha*D0 + D1
    expect_intcept = np.dot(np.dot(alpha_hat.transpose(),temp3),C0)
    expect_slope = np.dot(np.dot(alpha_hat.transpose(),temp3),Ct)
    return expect_intcept, expect_slope

def Johansen_std(alpha,beta,ut,rank=1):
    temp1 = np.eye(rank)+np.dot(beta.transpose(),alpha)
    temp2 = np.kron(temp1,temp1)
    temp3 = np.linalg.inv(np.eye(rank)-temp2)
    omega = np.dot(ut.transpose(),ut)/(len(ut)-1)
    temp4 = np.dot(np.dot(beta.transpose(),omega),beta)
    var = np.dot(temp3,temp4)
    #std = np.sqrt(var)
    return var

def Johansen_std_correct(alpha,beta,ut,mod_gamma,lag_p,rank=1):
    #論文中的closed form std
    NumDim = 2
    if lag_p > 0:
        #建立～A
        tilde_A_11 = alpha
        tilde_A_21 = np.zeros([NumDim*lag_p, 1])
        tilde_A_12 = np.zeros([NumDim, NumDim*lag_p])
        
        #建立～B
        tilde_B_11 = beta
        #tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        tilde_B_3 = np.zeros([NumDim + NumDim*lag_p, NumDim*lag_p])
        
        #用同一個迴圈同時處理～A與～B
        for qi in range(lag_p):
            tilde_A_12[0:NumDim,qi*NumDim:(qi+1)*NumDim] = mod_gamma[qi]
            tilde_B_3[qi*NumDim:NumDim*(2+qi),qi*NumDim:(qi+1)*NumDim] = np.vstack([ np.eye(NumDim), -np.eye(NumDim)])
        tilde_A_22 = np.eye(NumDim*lag_p)
        tilde_A = np.hstack([ np.vstack([tilde_A_11,tilde_A_21]),  np.vstack([tilde_A_12,tilde_A_22 ])])
        tilde_B = np.hstack([ np.vstack([tilde_B_11,tilde_A_21]), tilde_B_3])
    
    elif lag_p == 0:
        tilde_A = alpha
        tilde_B = beta
    tilde_Sigma = np.zeros([NumDim*(lag_p+1), NumDim*(lag_p+1)])
    tilde_Sigma[0:NumDim, 0:NumDim] = np.dot(ut.transpose(),ut)/(len(ut)-1)
    tilde_J = np.zeros([1, 1+NumDim*(lag_p)])
    tilde_J[0,0] = 1
    if lag_p == 0  :
        temp1 = np.eye(rank)+np.dot(beta.transpose(),alpha)
        temp2 = np.kron(temp1,temp1)
        temp3 = np.linalg.inv(np.eye(rank)-temp2)
        omega = np.dot(ut.transpose(),ut)/(len(ut)-1)
        temp4 = np.dot(np.dot(beta.transpose(),omega),beta)
        var = np.dot(temp3,temp4)
    else:   
        temp1 =np.eye(NumDim*(lag_p+1)-1)+np.dot(tilde_B.transpose(),tilde_A)
        temp2 = np.kron(temp1,temp1)
        k = (NumDim*(lag_p+1)-1)*(NumDim*(lag_p+1)-1)
        temp3 = np.linalg.inv(np.eye(k)-temp2)
        temp4 = np.dot(np.dot(tilde_B.transpose(),tilde_Sigma),tilde_B)
        temp4 = temp4.flatten('F')
        temp5 = np.dot(temp3,temp4)
        sigma_telta_beta = np.zeros([NumDim*(lag_p+1)-1, NumDim*(lag_p+1)-1])
        for i in range(NumDim*(lag_p+1)-1):
            for j in range(NumDim*(lag_p+1)-1):
                sigma_telta_beta[i][j]= temp5[0,i+j*(NumDim*(lag_p+1)-1)]
        var = np.dot(np.dot(tilde_J, sigma_telta_beta), tilde_J.transpose())
    return var


def TradeCost(SSt, InitB, c ,LS):
    TC = 0
    if LS == 'S':
        TC = max(SSt[0]*InitB[0]*c,0)
        TC = TC + max(SSt[1]*InitB[1]*c,0)
        return TC
    elif LS == 'L':
        TC = abs(min(SSt[0]*InitB[0]*c,0))
        TC = TC + abs(min(SSt[1]*InitB[1]*c,0))
        return TC

def simp_frac(x,y,Range):
    #輸入兩個實數，可以找到最接近的整數比例
    #Range是限制，回傳的值再不考慮正負號的情況下只會在[1,Range]到[Range,1]之間
    
    #紀錄正負號
    PNlog1 , PNlog2 = (x>0)*2-1 , (y>0)*2-1
    #將兩個數字都變為正數
    intx , inty = x * PNlog1 , y * PNlog2
    
    #建構xy數列
    intrange = [i for i in range(1,Range+1)]  
    #依照xy數列建構空的atan夾角空間
    anglespace = np.zeros([ len(intrange) , len(intrange) ])
    
    for inti in range(len(intrange)):
        for intj in range(len(intrange)):
            #填充atan夾角空間，單位為弧度
            anglespace[inti,intj] = math.atan( intrange[intj] / intrange[inti] ) 
    
    # 兩個度數相減得到夾角弧度，degspace為夾角弧度的絕對值
    degspace = abs(anglespace - math.atan(inty/intx))
    # 找出最小夾角並回傳空間內座標
    optij = np.where(degspace == np.min(degspace) )
    
    # 依據空間內座標找出最適的xy，如果有重複答案選取最小，最後把前面的正負號乘回去
    SFx1 = intrange[optij[0][0]] * PNlog1
    SFy2 = intrange[optij[1][0]] * PNlog2
    
    return [SFx1,SFy2]

@njit(cache=True)
def tax(payoff, rate):
    return payoff * (1.0 - rate * (payoff > 0))

def tax_two_side(payoff,rate):
    #定義cashflow為payoff，若payoff為正，代表獲得現金（建空倉or平多倉）
    #若payoff為負，代表損失現金（空倉回補or建多倉）
    #若payoff為負時課rate的稅
    if payoff > 0:
        tax_price = payoff * (1 - rate * (payoff > 0))
    else:
        tax_price = payoff * (1 + rate * (payoff < 0))
    return tax_price    
        

def min_integer( w1 , w2 , stock1_max , stock2_max ):
    # 最小化整數比
    y = abs(w2 / w1)
    #    print("y:",y)
    #if (y>10) or (y<(1/10)):
    #    return [ 0, 0]
    theta = np.arctan(y)

    sq = []
    for i in range(1,stock1_max+1):
        for j in range(1,stock2_max+1):
            sq.append( [ i , j , abs(theta - np.arctan(j/i)) ]) 
    
    sq = np.array(sq)
    min_pos = np.array(np.where(sq[:,2] == np.min(sq[:,2])) )  # 挑出角度差最小的權重

    if len(min_pos.T) > 1:                                     # 如果有重複，則挑第一個
        min_pos = min_pos[0,0]
    else:
        min_pos = int(min_pos)
        
    #回傳值依原始正負調整-------------------------------------------------------------------------------
    if w1 > 0 and w2 > 0:
        w1 = sq[min_pos,0]
        w2 = sq[min_pos,1]
    elif w1 < 0 and w2 > 0:
        w1 = -sq[min_pos,0]
        w2 =  sq[min_pos,1]
    elif w1 > 0 and w2 < 0:
        w1 =  sq[min_pos,0]
        w2 = -sq[min_pos,1]
    else:
        w1 = -sq[min_pos,0]
        w2 = -sq[min_pos,1]
    return [ w1 , w2 ]

def num_weight_old( w1 , w2 , price1 , price2 , maxi , initial_capital):
    #將資金權重換成股票張數權重，並進行整數化 ; maxi為最大張數。
    #initial_capital = 3000      # 總資產300萬台幣
    #print("w1:",w1,",w2:",w2)
    stock1_num = (w1 * initial_capital)/price1
    stock2_num = (w2 * initial_capital)/price2
    #print("stw1:",stock1_num,"stw2",stock2_num)
    #if price1>1000 or price2 >1000:
    #return [ 0 , 0 ]
    if abs(stock1_num) > maxi or abs(stock2_num) > maxi :
        
        stock1_maxi = maxi
        stock2_maxi = maxi
        
    elif abs(stock1_num) > maxi or abs(stock2_num) < maxi :
        
        stock1_maxi = maxi
        stock2_maxi = abs(int(round(stock2_num)))
        
    elif abs(stock1_num) < maxi or abs(stock2_num) > maxi :
        
        stock1_maxi = abs(int(round(stock1_num)))
        stock2_maxi = maxi
        
    else:
        
        stock1_maxi = abs(int(round(stock1_num)))
        stock2_maxi = abs(int(round(stock2_num)))
        
    if (abs(stock1_num)<0.5) or (abs(stock2_num) <0.5) :
        return [0,0]
    w1 , w2 = min_integer( stock1_num , stock2_num , stock1_maxi , stock2_maxi )
    return [ w1 , w2 ]

@njit(cache=True)
def num_weight(w1, w2, price1, price2, maxi, initial_capital):
    stock1_num = w1 / price1
    stock2_num = w2 / price2
    y = abs(stock2_num / stock1_num)
    theta = np.arctan(y)

    # 預先分配 numpy 陣列代替 Python list（Numba 不支援動態 list）
    n = maxi * maxi
    sq = np.empty((n, 3), dtype=np.float64)
    idx = 0
    for i in range(1, maxi + 1):
        for j in range(1, maxi + 1):
            sq[idx, 0] = float(i)
            sq[idx, 1] = float(j)
            sq[idx, 2] = abs(theta - np.arctan(float(j) / float(i)))
            idx += 1

    # 手動找最小值位置（np.where 在 njit 裡的 flat 屬性不支援）
    min_val = sq[0, 2]
    min_pos = 0
    for k in range(1, n):
        if sq[k, 2] < min_val:
            min_val = sq[k, 2]
            min_pos = k

    if w1 > 0 and w2 > 0:
        rw1 = sq[min_pos, 0]
        rw2 = sq[min_pos, 1]
    elif w1 < 0 and w2 > 0:
        rw1 = -sq[min_pos, 0]
        rw2 =  sq[min_pos, 1]
    elif w1 > 0 and w2 < 0:
        rw1 =  sq[min_pos, 0]
        rw2 = -sq[min_pos, 1]
    else:
        rw1 = -sq[min_pos, 0]
        rw2 = -sq[min_pos, 1]

    size = int(maxi / max(abs(rw1), abs(rw2)))
    rw1 = rw1 * size
    rw2 = rw2 * size
    return rw1, rw2
    
    
def Formation(max_p, rawS):
    # VAR model 最大lag期數: max_p
    # rawS: 兩檔股價資料
    ind = np.zeros([1,9])
    stocka = rawS.columns[0]
    stockb = rawS.columns[1]
    rawS = np.array(rawS)
    opt_p = order_select(rawS, max_p) # 配適 VAR(P) 模型 ，並利用BIC選擇落後期數
        
    if opt_p <1:
        return pd.DataFrame(np.zeros([1,9]))
    
    # 殘差檢定，若殘差非whiteness 或  非normal 跳過該配對
    VAR_model_set = VAR(rawS)
    # 檢定VAR殘差是否為whiteness,StatPValue小於0.05代表非whiteness
    if VAR_model_set.fit(opt_p).test_whiteness( nlags = max_p ).pvalue < 0.05:
        return pd.DataFrame(np.zeros([1,9]))
    
    # 檢定VAR殘差是否為normal,StatPValue小於0.05代表非normal
    if VAR_model_set.fit(opt_p).test_normality().pvalue < 0.05:
        return pd.DataFrame(np.zeros([1,9]))
    try:
        opt_model = JCI_AutoSelection(rawS,opt_p-1)  #bic based model selection
        #如果有共整合，紀錄下Model與opt_q
        ind[0,2] = opt_p-1
        ind[0,6] = opt_model
        F_a, F_b, F_ct, F_ut, F_gam, ct, omega_hat = JCItestpara_spilCt(rawS,opt_model,opt_p-1)
        #把arrary.shape(2,1) 的數字放進 shape(2,) 的Serires 
        #取出共整合係數
        Beta =  pd.DataFrame(F_b).stack()
        #將共整合係數標準化，此為資金權重Capital Weight
        CapitW =  Beta / np.sum( np.absolute(Beta) )
        ind[0,7] = CapitW[0,0]
        ind[0,8] = CapitW[1,0]
        #計算Spread的時間趨勢均值與標準差
        Johansen_intcept, Johansen_slope = Johansen_mean(F_a,F_b,F_gam,F_ct,opt_p-1)
        Johansen_var_correct = Johansen_std_correct(F_a,F_b,F_ut, F_gam,opt_p-1)
        if Johansen_var_correct >= 0:
            Johansen_std = np.sqrt(Johansen_var_correct)
        elif Johansen_var_correct <= 0:
            return pd.DataFrame(np.zeros([1,9]))
        ind[0,3] = Johansen_intcept
        ind[0,4] = Johansen_slope
        ind[0,5] = Johansen_std
    except:
        return pd.DataFrame(np.zeros([1,9]))
    
    ind = pd.DataFrame(ind)
    ind.columns = ['S1','S2','VECM(q)','Johansen_intercept','Johansen_slope',
                   'Johansen_std','VECM_Model_Type','w1','w2']
    ind.loc[0,'S1'] = stocka
    ind.loc[0,'S2'] = stockb
    return ind

    