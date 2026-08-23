import numpy as np
import pandas as pd
from .integer import num_weight
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from .vecm import para_vecm
from scipy.stats import f
import os
from sklearn.linear_model import LinearRegression


def spread_mean(stock1, stock2, i, table):
    if table.model_type.iloc[i] == 'model1':
        model = 'H2'
    elif table.model_type.iloc[i] == 'model2':
        model = 'H1*'
    elif table.model_type.iloc[i] == 'model3':
        model = 'H1'
    else:
        raise ValueError(f'Please check {table.model_type.iloc[i]} !')
    stock1 = stock1[i, :150]
    stock2 = stock2[i, :150]
    b1 = table.w1.iloc[i]
    b2 = table.w2.iloc[i]
    y = np.vstack([stock1, stock2]).T
    logy = np.log(y)
    # print(logy)
    lyc = logy.copy()
    p = order_select(logy,5)
    # print('p:',p)
    _, _, para = para_vecm(logy, model, p)
    logy = np.mat(logy)
    y_1 = np.mat(logy[p:])
    dy = np.mat(np.diff(logy,axis=0))
    for j in range(len(stock1)-p-1):
        if model == 'H1':
            if p != 1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.hstack([dy[j:(j+p-1)].flatten(),np.mat([1])]).T
            else:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.mat([1])
        elif model == 'H1*':
            if p != 1:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T
        elif model == 'H2':
            if p != 1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * y_1[j].T
        else:
            raise ValueError(f'Please check {model} !')
        dy[j+p, :] = delta.T
        y_1[j+1] = y_1[j] + delta.T
    b = np.mat([[b1], [b2]])
    spread = b.T*lyc[p:].T
    spread_m = np.array(b.T*y_1.T).flatten()
    return spread_m, spread


def get_Estd(stock1, stock2, i, table, dy=True, D=16):
    if table.model_type.iloc[i] == 'model1':
        model = 'H2'
    elif table.model_type.iloc[i] == 'model2':
        model = 'H1*'
    elif table.model_type.iloc[i] == 'model3':
        model = 'H1'
    else:
        raise ValueError(f'Please check {table.model_type.iloc[i]} !')
    stock1 = stock1[i, :150]
    stock2 = stock2[i, :150]
    b1 = table.w1.iloc[i]
    b2 = table.w2.iloc[i]
    b = np.mat([[b1], [b2]])
    y = np.vstack([stock1, stock2]).T
    logy = np.log(y)  # np.log(y)
    p = order_select(logy, 5)
    u, A, _ = para_vecm(logy, model, p)
    constant = np.mat(A[:, 0])
    A = A[:, 1:]
    l = A.shape[1]
    extend = np.hstack([np.identity(l-2), np.zeros([l-2, 2])])
    newA = np.vstack([A, extend])
    if not dy:
        lagy = logy[p-1:-1, :]
        for i in range(1, p):
            lagy = np.hstack([lagy, logy[p-1-i:-i-1, :]])
        MatrixA = np.mat(A)
        MatrixLagy = np.mat(lagy)
        Estimate_logy = MatrixA * MatrixLagy.T + constant
        e = logy[p:, :].T-Estimate_logy
        var = e*e.T/e.shape[1]
    else:
        var = u*u.T/u.shape[1]
    NowCoef = np.mat(np.eye(len(newA)))
    Evar = var.copy()
    for i in range(149):
        NowCoef = newA * NowCoef
        Evar = Evar + NowCoef[:2, :2]*var*NowCoef[:2, :2].T
    Evar = b.T * Evar * b
    return np.sqrt(Evar)


def VAR_model(y, p):
    k = len(y.T)  # 幾檔股票
    n = len(y)  # 資料長度
    xt = np.ones((n - p, (k * p) + 1))
    for i in range(n - p):
        a = 1
        for j in range(p):
            a = np.hstack((a, y[i + p - j - 1]))
        a = a.reshape([1, (k * p) + 1])
        xt[i] = a
    zt = np.delete(y, np.s_[0:p], axis=0)
    xt = np.mat(xt)
    zt = np.mat(zt)
    beta = (xt.T * xt).I * xt.T * zt  # 計算VAR的參數
    A = zt - xt * beta  # 計算殘差
    sigma = (A.T * A) / (n - p)  # 計算殘差的共變異數矩陣
    return [sigma, beta]


def order_select(y, max_p):
    k = len(y.T)  # 幾檔股票
    n = len(y)  # 資料長度
    bic = np.zeros((max_p, 1))
    for p in range(1, max_p + 1):
        sigma = VAR_model(y, p)[0]
        bic[p - 1] = np.log(np.linalg.det(sigma)) + np.log(n) * p * (k * k) / n
    bic_order = int(np.where(bic == np.min(bic))[0] + 1)  # 因為期數p從1開始，因此需要加1
    return bic_order


def fore_chow(stock1, stock2, stock1_trade, stock2_trade, model):
    if model == 'model1':
        model_name = 'H2'
    elif model == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'
    y = np.vstack([stock1, stock2]).T
    day1 = np.vstack([stock1_trade, stock2_trade]).T
    k = len(y.T)  # 幾檔股票
    n = len(y)  # formation period 資料長度
    y = np.log(y)
    day1 = np.log(day1)
    h = len(day1) - n
    p = order_select(y, 5)  # 計算最佳落後期數
    # ut , A = VAR_model(y , p)                                                                 # 計算VAR殘差共變異數與參數
    at, A = para_vecm(y, model_name, p)
    ut = np.dot(at, at.T) / len(at.T)
    # A = pd.DataFrame(A)
    A = A.T
    phi_0 = np.eye(k)
    A1 = np.delete(A, 0, axis=0).T
    phi = np.hstack((np.zeros([k, 2 * (p - 1)]), phi_0))
    sigma_t = np.dot(np.dot(phi_0, ut), phi_0.T)  # sigma hat
    ut_h = []
    for i in range(1, h + 1):
        lag_mat = day1[len(day1) - i - p - 1:  len(day1) - i, :]
        lag_mat = np.array(lag_mat[::-1])
        if p == 1:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:2].T)).T)
        else:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:k * p - 1].reshape([k * p, 1]))).T)
    for i in range(h - 1):
        a = phi[:, i * 2:len(phi.T)]
        phi_i = np.dot(A1, a.T)
        sigma_t = sigma_t + np.dot(np.dot(phi_i, ut), phi_i.T)
        phi = np.hstack((phi, phi_i))
    phi = phi[:, ((p - 1) * k):len(phi.T)]
    ut_h = np.array(ut_h).reshape([1, h * 2])
    e_t = np.dot(phi, ut_h.T)
    # 程式防呆，如果 sigma_t inverse 發散，則回傳有結構性斷裂。
    # noinspection PyBroadException
    try:
        tau_h = np.dot(np.dot(e_t.T, np.linalg.inv(sigma_t)), e_t) / k
    except:
        return 1
    else:
        if tau_h > float(f.ppf(0.99, k, n - k * p + 1)):  # tau_h > float(chi2.ppf(0.99,k)):
            return 1  # 有結構性斷裂
        else:
            return 0


def fore_chow_jordan(stock1, stock2, model, Flen, give=False, p=0, A=0, ut=0, maxp=5):
    if model == 'model1':
        model_name = 'H2'
    elif model == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'

    day1 = np.vstack([stock1, stock2]).T
    day1 = np.log(day1)
    h = len(day1) - Flen
    k = 2  # 幾檔股票
    n = Flen  # formation period 資料長度

    if not give:
        y = np.vstack([stock1[0:Flen], stock2[0:Flen]]).T
        y = np.log(y)
        p = order_select(y, maxp)
        at, A = para_vecm(y, model_name, p)
        ut = np.dot(at, at.T) / len(at.T)

    Remain_A = A.copy()
    Remain_ut = ut.copy()
    Remain_p = p

    A = A.T
    phi_0 = np.eye(k)
    A1 = np.delete(A, 0, axis=0).T
    phi = np.hstack((np.zeros([k, 2 * (p - 1)]), phi_0))
    sigma_t = np.dot(np.dot(phi_0, ut), phi_0.T)  # sigma hat
    ut_h = []

    for i in range(1, h + 1):
        lag_mat = day1[len(day1) - i - p - 1:  len(day1) - i, :]
        lag_mat = np.array(lag_mat[::-1])
        if p == 1:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:2].T)).T)
        else:
            ut_h.append(lag_mat[0].T - (A[0].T + np.dot(A[1:k * p + 1].T, lag_mat[1:k * p - 1].reshape([k * p, 1]))).T)

    for i in range(h - 1):
        a = phi[:, i * 2:len(phi.T)]
        phi_i = np.dot(A1, a.T)
        sigma_t = sigma_t + np.dot(np.dot(phi_i, ut), phi_i.T)
        phi = np.hstack((phi, phi_i))
    phi = phi[:, ((p - 1) * k):len(phi.T)]
    ut_h = np.array(ut_h).reshape([1, h * 2])
    e_t = np.dot(phi, ut_h.T)

    # 程式防呆，如果 sigma_t inverse 發散，則回傳有結構性斷裂。
    # noinspection PyBroadException
    try:
        tau_h = np.dot(np.dot(e_t.T, np.linalg.inv(sigma_t)), e_t) / k
    except:
        return Remain_p, Remain_A, Remain_ut, 1
    else:
        if tau_h > float(f.ppf(0.99, k, n - k * p + 1)):  # tau_h > float(chi2.ppf(0.99,k)):
            return Remain_p, Remain_A, Remain_ut, 1  # 有結構性斷裂
        else:
            return Remain_p, Remain_A, Remain_ut, 0


def spread_cross_threshold(trigger_spread, threshold, add_num):
    # initialize array
    check = np.zeros(trigger_spread.shape)
    # put on the condition
    check[(trigger_spread - threshold) > 0] = add_num
    check[:, 0] = check[:, 1]
    # Open_trigger_array
    check = check[:, 1:] - check[:, :-1]
    return check


# up_down = true, if focus on breaking upward
# trigger_spread = spreads time series of (pairs x time) in trading period
# add_num is a flag meaning origin position is buy pair(3) or sale pair(1), or do not break(0)
# return 1 or 3 or 0 , shape = (pairs x time)
def spread_up_down_threshold(trigger_spread, threshold, add_num, up_down):
    check = np.zeros(trigger_spread.shape)
    if up_down:     # focus on breaking upward
        check[(trigger_spread - threshold) > 0] = add_num
    else:           # focus on breaking downward
        check[(trigger_spread - threshold) < 0] = add_num
    check[:, 0] = 0 # force first minute do nothing
    return check



def tax(payoff, rate):
    tax_price = payoff * (1 - rate * (payoff > 0))
    return tax_price


'''
tick = False
table = formation table (Result of Johenson test)
formation_period = 150
trading_period = 100
avg_min_data = average minute data
trigger_data = last tick minute data of trading period
all_trigger_data = all last tick minute data
open_times = open threshold
close_times = 0
stop_loss_times = stop loss threshold
maxi = 5
tax_cost = 0.0015
cost_gate = 0.0015
capital = 50000000
cross_threshold=False
cost_gate_type=0
dump=False
folder_name='folder_name'
method=None
reward_type=None
output_stock_price=False
trend_stationary=False
'''

class Trading(object):
    def __init__(self, tick, table, formation_period, trading_period, open_delete, close_delete, avg_min_data,
                 trigger_data, all_trigger_data, open_times, close_times, stop_loss_times, maxi, tax_cost, cost_gate,
                 capital, cross_threshold=False, cost_gate_type=0, dump=False, folder_name='folder_name', method=None,
                 reward_type=None, output_stock_price=False, trend_stationary=False
                 ):
        self.tick = tick
        self.table = table  # formation period table
        self.formation_period = formation_period  # 建模時間
        self.trading_period = trading_period  # 交易時間
        self.open_delete = open_delete  # 開盤刪除
        self.close_delete = close_delete  # 尾盤刪除
        self.avg_min_data = avg_min_data  # 每分鐘平均股價
        self.trigger_data = trigger_data  # 測試期間每五秒股價
        self.all_trigger_data = all_trigger_data  # 建模期間+測試期間每五秒股價
        self.open_times = open_times  # 開倉倍數
        self.close_times = close_times  # 平倉倍數
        self.stop_loss_times = stop_loss_times  # 停損倍數
        self.maxi = maxi  # 最大股票持有張數
        self.tax_cost = tax_cost  # 交易成本
        self.cost_gate = cost_gate  # 交易門檻
        self.capital = capital  # 每組配對最大資金上限
        self.dump = dump
        self.cross_threshold = cross_threshold
        self.cost_gate_type = cost_gate_type
        self.output_stock_price = output_stock_price
        '''
        cost gate type: 
            0 for cost gate is upper bound minus mean > cost gate
            1 for cost gate is spread minus mean > cost gate
        '''
        self.folder_name = folder_name
        self.use_spread_max_min = False
        self.method = method
        self.reward_type = reward_type
        self.trend_stationary = trend_stationary

    def check_fore_lag5_timing(self, x, open_timing):
        stock1_name = self.table.stock1.astype('str', copy=False)
        stock2_name = self.table.stock2.astype('str', copy=False)
        avg_min_stock1 = np.array(self.avg_min_data[stock1_name].T)[x, :]
        avg_min_stock2 = np.array(self.avg_min_data[stock2_name].T)[x, :]
        model_type = self.table.model_type[x]
        count = 0
        p, A, ut, _ = fore_chow_jordan(avg_min_stock1[:self.formation_period + 1],
                                       avg_min_stock2[:self.formation_period + 1],
                                       model_type, self.formation_period)
        for i in range(open_timing // (1 + self.tick * 11), self.trading_period):
            p, A, ut, num = fore_chow_jordan(avg_min_stock1[:self.formation_period + i + 1],
                                             avg_min_stock2[:self.formation_period + i + 1],
                                             model_type, self.formation_period, True, p, A, ut)
            if num == 0:
                count = 0
            else:
                count += num
            if count == 5:
                return i * (1 + self.tick * 11)
        return self.trading_period * (1 + self.tick * 11)

    def check_exit_timing(self, check_close, check_stop_loss):
        sec_trading_period = self.trading_period * (1 + self.tick * 11)
        normal_close_timing = np.argmax(check_close != 0)
        stop_loss_timing = np.argmax(check_stop_loss != 0)
        if normal_close_timing >= sec_trading_period - 2:
            normal_close_timing = 0
        if stop_loss_timing >= sec_trading_period - 2:
            stop_loss_timing = 0
        if normal_close_timing == 0 and stop_loss_timing == 0:      #一天結束前都還沒停損or平倉, 就回傳結束時間, record = -4
            return -4, sec_trading_period - 2
        else:
            if normal_close_timing == 0:        # 有停損沒平倉, 就回傳停損時間, record = -2
                return -2, stop_loss_timing
            elif stop_loss_timing == 0:         # 有平倉沒停損 , 就回傳平倉時間, record = 666
                return 666, normal_close_timing
            else:
                if normal_close_timing < stop_loss_timing:  # 平倉先出現 , record = 666
                    return 666, normal_close_timing
                else:
                    return -2, stop_loss_timing             # 停損先出現, record = -2

    @staticmethod
    def initialize_performance_dict():
        performance_dict = dict()
        performance_dict['profit'] = 0
        performance_dict['half_tax_profit'] = 0
        performance_dict['zero_tax_profit'] = 0
        performance_dict['reward'] = 0
        performance_dict['record'] = 0
        performance_dict['close_timing'] = -999
        performance_dict['open_timing'] = -999
        performance_dict['capital'] = 0
        performance_dict['stock1_num'] = 0
        performance_dict['stock2_num'] = 0
        performance_dict['w1'] = 0
        performance_dict['w2'] = 0
        return performance_dict

    def pairs_trading_back_test(self, stock_date, folder_name, open40=False, use_adf=True, use_fore_lag5=True, new_std=True):

        self.table.reset_index(drop=True, inplace=True)

        stock1_name = self.table.stock1.astype('str', copy=False)
        stock2_name = self.table.stock2.astype('str', copy=False)

    # get stock1, stock2 time series price data
        model_type = self.table.model   # model = Johenson test model(1~5)
        trigger_stock1 = np.array(self.trigger_data[stock1_name].T)
        trigger_stock2 = np.array(self.trigger_data[stock2_name].T)
        all_trigger_stock1 = np.array(self.all_trigger_data[stock1_name].T)
        all_trigger_stock2 = np.array(self.all_trigger_data[stock2_name].T)
        avg_min_stock1 = np.array(self.avg_min_data[stock1_name].T)
        avg_min_stock2 = np.array(self.avg_min_data[stock2_name].T)

    # get variables of formation table
        if self.method is None:
            w1 = np.expand_dims(np.array(self.table.w1), axis=1)
            w2 = np.expand_dims(np.array(self.table.w2), axis=1)
            trigger_spread = w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2)
            all_trigger_spread = w1 * np.log(all_trigger_stock1) + w2 * np.log(all_trigger_stock2)
            # Note that : Estd = 'std' in csv file ; Emu = 'Johansen_intercept' in csv file
            if self.trend_stationary:
                std = np.full((101, len(self.table)), self.table['Estd']).T
                slope = np.full((101, len(self.table)), self.table.Johansen_slope).T
                tmp = slope * np.array((list(range(149, 250, 1))) * len(self.table)).reshape(len(self.table), 101)
                intercept = np.full((101, len(self.table)), self.table.Emu).T
                mu = tmp + intercept
                slope = np.full((250, len(self.table)), self.table.Johansen_slope).T
                tmp = slope * np.array((list(range(0, 250, 1))) * len(self.table)).reshape(len(self.table), 250)
                intercept = np.full((250, len(self.table)), self.table.Emu).T
                output_json_mu = tmp + intercept
            else:
                if new_std:
                    std = np.array(self.table.Estd)
                    mu = np.array(self.table.Emu)
                else:
                    std = np.array(self.table.stdev)
                    mu = np.array(self.table.mu)
        elif self.method == 'soft_computing':
            w1 = np.expand_dims(np.array(self.table.w1), axis=1)
            w2 = np.expand_dims(np.array(self.table.w2), axis=1)
            trigger_spread = w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2)
            all_trigger_spread = w1 * np.log(all_trigger_stock1) + w2 * np.log(all_trigger_stock2)
            if new_std:
                std = np.array(self.table.Estd).reshape(-1, 1)
                mu = np.array(self.table.Emu).reshape(-1, 1)
            else:
                std = np.array(self.table.stdev).reshape(-1, 1)
                mu = np.array(self.table.mu).reshape(-1, 1)
            trigger_spread = (trigger_spread - mu) / std
            all_trigger_spread = (all_trigger_spread - mu) / std
            mu = np.array([0] * len(trigger_stock1))
            std = np.array([1] * len(trigger_stock1))
        elif self.method == 'eg_ols':
            w1 = None
            w2 = None
            alpha = None
            for i in range(len(trigger_stock1)):
                reg = LinearRegression().fit(np.log(all_trigger_stock1)[i][16:166].reshape(-1, 1),
                                             np.log(all_trigger_stock2)[i][16:166].reshape(-1, 1))
                if w1 is None:
                    w1 = reg.coef_
                else:
                    w1 = np.vstack((w1, reg.coef_))
                if w2 is None:
                    w2 = [1]
                else:
                    w2 = np.vstack((w2, [1]))
                if alpha is None:
                    alpha = reg.intercept_
                else:
                    alpha = np.vstack((alpha, reg.intercept_))
            w1 = w1 * (-1)
            all_trigger_spread = w1 * np.log(all_trigger_stock1) + w2 * np.log(all_trigger_stock2) - alpha
            if not new_std:
                std = np.std(all_trigger_spread[:, 16:166], axis=1)
                mu = np.mean(all_trigger_spread[:, 16:166], axis=1)
            else:
                mu = np.zeros(len(trigger_stock1))
                std = np.zeros(len(trigger_stock1))
                for i in range(len(trigger_stock1)):
                    spread_m, spread = spread_mean(all_trigger_stock1[:, 16:], all_trigger_stock2[:, 16:], i, self.table)
                    mu[i] = np.mean(spread_m[-1:])
                    std[i] = get_Estd(all_trigger_stock1[:, 16:], all_trigger_stock2[:, 16:], i, self.table)
            trigger_spread = (w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2) - alpha -
                              np.expand_dims(mu, axis=1)) / np.expand_dims(std, axis=1)
            all_trigger_spread = (all_trigger_spread - np.expand_dims(mu, axis=1)) / np.expand_dims(std, axis=1)
            mu = np.array([0] * len(trigger_stock1))
            std = np.array([1] * len(trigger_stock1))
        elif self.method == 'eg_tls':
            w1 = None
            w2 = None
            alpha = None
            for i in range(len(trigger_stock1)):
                x = np.log(all_trigger_stock1)[i][16:166]
                y = np.log(all_trigger_stock2)[i][16:166]
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                s_xx = sum([(tmp-x_mean) ** 2 for tmp in x]) / (len(x) - 1)
                s_yy = sum([(tmp-y_mean) ** 2 for tmp in y]) / (len(y) - 1)
                s_xy = sum([(a-x_mean) * (b-y_mean) for a, b in zip(x, y)]) / (len(x) - 1)
                delta = 1
                beta = \
                    ((s_yy - delta * s_xx) + ((s_yy - delta * s_xx) ** 2 + 4 * delta * (s_xy ** 2)) ** (1/2)) / 2 * s_xy
                intercept = y_mean - beta * x_mean
                if w1 is None:
                    w1 = beta
                else:
                    w1 = np.vstack((w1, beta))
                if w2 is None:
                    w2 = [1]
                else:
                    w2 = np.vstack((w2, [1]))
                if alpha is None:
                    alpha = intercept
                else:
                    alpha = np.vstack((alpha, intercept))
            w1 = w1 * (-1)
            all_trigger_spread = w1 * np.log(all_trigger_stock1) + w2 * np.log(all_trigger_stock2) - alpha
            if not new_std:
                std = np.std(all_trigger_spread[:, 16:166], axis=1)
                mu = np.mean(all_trigger_spread[:, 16:166], axis=1)
            else:
                mu = np.zeros(len(trigger_stock1))
                std = np.zeros(len(trigger_stock1))
                for i in range(len(trigger_stock1)):
                    spread_m, spread = spread_mean(all_trigger_stock1[:, 16:], all_trigger_stock2[:, 16:], i, self.table)
                    mu[i] = np.mean(spread_m[-1:])
                    std[i] = get_Estd(all_trigger_stock1[:, 16:], all_trigger_stock2[:, 16:], i, self.table)
            trigger_spread = (w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2) - alpha -
                              np.expand_dims(mu, axis=1)) / np.expand_dims(std, axis=1)
            all_trigger_spread = (all_trigger_spread - np.expand_dims(mu, axis=1)) / np.expand_dims(std, axis=1)
            mu = np.array([0] * len(trigger_stock1))
            std = np.array([1] * len(trigger_stock1))
        else:
            raise ValueError('Unknown method! ')
        # tick = false
        if self.tick:
            trigger_stock1 = trigger_stock1[:, :-48]
            trigger_stock2 = trigger_stock2[:, :-48]
            trigger_spread = trigger_spread[:, :-48]


    # get up_baseline and down_baseline = array of mu of spread of stocks
        spread_max = all_trigger_spread[:, :166].max(axis=1)    # get array of max spread of stocks
        spread_min = all_trigger_spread[:, :166].min(axis=1)    # get array of min spread of stocks
        if self.use_spread_max_min:
            up_baseline = spread_max
            down_baseline = spread_min
        else:
            up_baseline = mu
            down_baseline = mu
        # plot_mu = mu.copy()



    # get open/close/stop_loss threshold(upper bond and lower bond)
        if self.trend_stationary:
            up_open = up_baseline + self.open_times * std
            up_close = down_baseline - self.close_times * std
            down_open = down_baseline - self.open_times * std
            down_close = up_baseline + self.close_times * std
            up_stop_loss = up_baseline + self.stop_loss_times * std
            down_stop_loss = down_baseline - self.stop_loss_times * std
            std = std[:, 0]
            mu = [list(m) for m in output_json_mu]
        else:
            up_open = np.expand_dims(up_baseline + self.open_times * std, axis=1)
            up_close = np.expand_dims(down_baseline - self.close_times * std, axis=1)
            down_open = np.expand_dims(down_baseline - self.open_times * std, axis=1)
            down_close = np.expand_dims(up_baseline + self.close_times * std, axis=1)
            up_stop_loss = np.expand_dims(up_baseline + self.stop_loss_times * std, axis=1)
            down_stop_loss = np.expand_dims(down_baseline - self.stop_loss_times * std, axis=1)



    # check when to open and close
    # getting when do all pairs cross threshold or not , value = 0,1,or 3 ; shape = (pair, time)
    # if double cross, then set to 0
        if self.cross_threshold:
            check_up_open = spread_cross_threshold(trigger_spread, up_open, 1)
            check_down_open = spread_cross_threshold(trigger_spread, down_open, 3)
            check_up_close = spread_cross_threshold(trigger_spread, up_close, 1)
            check_down_close = spread_cross_threshold(trigger_spread, down_close, 3)
            check_up_stop_loss = spread_cross_threshold(trigger_spread, up_stop_loss, 1)
            check_down_stop_loss = spread_cross_threshold(trigger_spread, down_stop_loss, 3)

            double_cross_up = np.multiply(check_up_open, check_up_close)
            double_cross_down = np.multiply(check_down_open, check_down_close)
            check_up_open[double_cross_up != 0] = 0
            check_down_open[double_cross_down != 0] = 0

            double_cross_up = np.multiply(check_up_open, check_up_stop_loss)
            double_cross_down = np.multiply(check_down_open, check_down_stop_loss)
            check_up_open[double_cross_up != 0] = 0
            check_down_open[double_cross_down != 0] = 0
        else:
            check_up_open = spread_up_down_threshold(trigger_spread, up_open, 1, True)
            check_down_open = spread_up_down_threshold(trigger_spread, down_open, 3, False)
            check_up_close = spread_up_down_threshold(trigger_spread, up_close, 1, False)
            check_down_close = spread_up_down_threshold(trigger_spread, down_close, 3, True)
            check_up_stop_loss = spread_up_down_threshold(trigger_spread, up_stop_loss, 1, True)
            check_down_stop_loss = spread_up_down_threshold(trigger_spread, down_stop_loss, 3, False)
            # check whether cross open threshold && cross stop loss threshold, if so then sset to 0
            double_cross_up = np.multiply(check_up_open, check_up_stop_loss)
            double_cross_down = np.multiply(check_down_open, check_down_stop_loss)
            check_up_open[double_cross_up != 0] = 0
            check_down_open[double_cross_down != 0] = 0
            if self.trend_stationary:
                check_up_open[self.table.Johansen_slope > 0] = 0
                check_down_open[self.table.Johansen_slope < 0] = 0

        if self.cost_gate_type == 1:
            check_up_open[abs(trigger_spread - np.expand_dims(mu, axis=1)) <= self.cost_gate] = 0
            check_down_open[abs(trigger_spread - np.expand_dims(mu, axis=1)) <= self.cost_gate] = 0
        # Combine open trigger array
        check_open = check_up_open + check_down_open
        # 40分鐘後不開倉
        if open40 is False:
            check_open[:, 41 * (1 + self.tick * 11):] = 0
        open_timing = np.argmax(check_open != 0, axis=1)        # array of index of first minute that check_open != 0; shape = len(pairs)
        minute_open_timing = open_timing // (1 + self.tick * 11)




    # get integer stock share pair at opening timing
        int_w = list()      # list of [w1,w2] for each pair, which [w1,w2] means 原Johenson test 得出的資金比重,
                            # 轉換成開倉當時的股價的整數股票張數w1,w2
        i = 0
        while i < len(open_timing):     # len of pairs that pass Johenson in this date
            if open_timing[i] != 0:     # there is a chance to open in trading period
                """
                # 檢查開倉時是否有同時突破停損門檻，如果有即不開倉
                if check_up_stop_loss[i, open_timing[i]] > 0 or check_down_stop_loss[i, open_timing[i]] < 0:
                    check_open[i, open_timing[i]] = 0
                    open_timing = np.argmax(check_open != 0, axis=1)
                    minute_open_timing = open_timing // (1 + self.tick * 11)
                    continue
                # 檢查是否同時破上開倉以及下開倉，如果有即不開倉
                if abs(check_open[i, open_timing[i]]) == 4:
                    check_open[i, open_timing[i]] = 0
                    open_timing = np.argmax(check_open != 0, axis=1)
                    minute_open_timing = open_timing // (1 + self.tick * 11)
                    continue
                
                # 檢查開倉時ADF是否有過，如果沒過即不在此時間點開倉
                w1, w2 = num_weight(self.table.w1.iloc[i], self.table.w2.iloc[i],
                                    trigger_stock1[i, (open_timing[i] + 1)], trigger_stock2[i, (open_timing[i] + 1)],
                                    self.maxi, self.capital)
                """
                w1, w2 = num_weight(self.table.w1.iloc[i], self.table.w2.iloc[i],
                                    trigger_stock1[i, (open_timing[i])], trigger_stock2[i, (open_timing[i])],
                                    self.maxi, self.capital)
                
                adf_spread = w1 * np.log(avg_min_stock1[i, :(self.formation_period + minute_open_timing[i] + 1)]) +\
                             w2 * np.log(avg_min_stock2[i, :(self.formation_period + minute_open_timing[i] + 1)])
                if use_adf:
                    if adfuller(adf_spread, regression='c')[1] > 0.05:
                        check_open[i, open_timing[i]] = 0
                        open_timing = np.argmax(check_open != 0, axis=1)
                        minute_open_timing = open_timing // (1 + self.tick * 11)
                    else:
                        int_w.append([w1, w2])
                        i += 1
                else:
                    int_w.append([w1, w2])
                    i += 1
            else:
                int_w.append([0, 0])
                i += 1


    # print result
        result = list()
        for i in range(len(open_timing)):       # len of pairs that pass Johenson in this date

            if self.cost_gate_type == 0:
                # if 這次開倉的策略(threshold)就算真的開倉了(mu - threshold*std), 所獲得的收益也比手續費還低
                if (self.open_times + self.close_times) * std[i] <= self.cost_gate:
                    performance_dict = self.initialize_performance_dict()   # init performance(profit = 0)
                    if self.output_stock_price:
                        result.append([stock1_name.iloc[i], stock2_name.iloc[i], model_type[i], mu[i], std[i],
                                       list(all_trigger_spread[i, 16:]), list(all_trigger_stock1[i, 16:]),
                                       list(all_trigger_stock2[i, 16:]), performance_dict,
                                       self.table.w1.iloc[i], self.table.w2.iloc[i]])
                    else:
                        result.append([stock1_name.iloc[i], stock2_name.iloc[i], model_type[i], mu[i], std[i],
                                       list(all_trigger_spread[i, 16:]), performance_dict,
                                       self.table.w1.iloc[i], self.table.w2.iloc[i]])
                    continue

            condition = abs(check_open[i, open_timing[i]])
            # long_short = 0
            if condition == 1:      # 上開倉
                long_short = -1
                # 把開倉前的停損flag都關掉
                check_up_close[i, :open_timing[i] + self.cross_threshold] = 0   #cross_threshold == 跨過threshold才開倉, 所以需要再等一分鐘, 如果沒設置,就不用
                check_up_stop_loss[i, :open_timing[i] + self.cross_threshold] = 0
                record, close_timing = self.check_exit_timing(check_up_close[i, :], check_up_stop_loss[i, :])
                if use_fore_lag5:
                    fore_lag5_timing = self.check_fore_lag5_timing(i, open_timing[i] + self.cross_threshold)
                    if fore_lag5_timing < close_timing:
                        close_timing = fore_lag5_timing
                        record = -3
            elif condition == 3:    # 下開倉
                long_short = 1
                # 把開倉前的停損flag都關掉
                check_down_close[i, :open_timing[i] + self.cross_threshold] = 0 #cross_threshold == 跨過threshold才開倉, 所以需要再等一分鐘, 如果沒設置,就不用
                check_down_stop_loss[i, :open_timing[i] + self.cross_threshold] = 0
                record, close_timing = self.check_exit_timing(check_down_close[i, :], check_down_stop_loss[i, :])
                if use_fore_lag5:
                    fore_lag5_timing = self.check_fore_lag5_timing(i, open_timing[i] + self.cross_threshold)
                    if fore_lag5_timing < close_timing:
                        close_timing = fore_lag5_timing
                        record = -3
            elif condition == 0:    # 啥都沒做, 直接輸出default performance
                performance_dict = self.initialize_performance_dict()
                if self.output_stock_price:
                    result.append(
                        [stock1_name.iloc[i], stock2_name.iloc[i], model_type[i], mu[i], std[i], list(all_trigger_spread[i, 16:]),
                         list(all_trigger_stock1[i, 16:]), list(all_trigger_stock2[i, 16:]), performance_dict,
                         self.table.w1.iloc[i], self.table.w2.iloc[i]])
                else:
                    result.append(
                        [stock1_name.iloc[i], stock2_name.iloc[i], model_type[i], mu[i], std[i], list(all_trigger_spread[i, 16:]),
                         performance_dict, self.table.w1.iloc[i], self.table.w2.iloc[i]])
                continue
            else:
                raise ValueError("Error Condition: ", condition)


        #算stock1 及 stock2 的投入資本
            stock1_open_price = trigger_stock1[i, (open_timing[i] + self.cross_threshold)]
            stock2_open_price = trigger_stock2[i, (open_timing[i] + self.cross_threshold)]
            stock1_close_price = trigger_stock1[i, (close_timing + self.cross_threshold)]
            stock2_close_price = trigger_stock2[i, (close_timing + self.cross_threshold)]
            capital = 0  # 投入本金
            if int_w[i][0]*long_short > 0:
                capital += stock1_open_price * int_w[i][0] * long_short
            else:
                capital += 0.9*stock1_open_price * abs(int_w[i][0] * long_short)  # 股票放空須繳交9成保證金
            if int_w[i][1]*long_short > 0:
                capital += stock2_open_price * int_w[i][1] * long_short
            else:
                capital += 0.9*stock2_open_price * abs(int_w[i][1] * long_short)  # 股票放空須繳交9成保證金


        # 算 profit
            open_s1_payoff = -long_short * stock1_open_price * int_w[i][0]
            open_s2_payoff = -long_short * stock2_open_price * int_w[i][1]
            close_s1_payoff = long_short * stock1_close_price * int_w[i][0]
            close_s2_payoff = long_short * stock2_close_price * int_w[i][1]
            profit = \
                tax(open_s1_payoff, 0.003) + tax(open_s2_payoff, 0.003) + tax(close_s1_payoff, 0.003) \
                + tax(close_s2_payoff, 0.003)
            half_tax_profit = \
                tax(open_s1_payoff, 0.0015) + tax(open_s2_payoff, 0.0015) + tax(close_s1_payoff, 0.0015) \
                + tax(close_s2_payoff, 0.0015)
            zero_tax_profit = \
                tax(open_s1_payoff, 0) + tax(open_s2_payoff, 0) + tax(close_s1_payoff, 0) \
                + tax(close_s2_payoff, 0)
            #if stock_date == '20160218' and stock1_name[i] == '2002' and stock2_name[i] == '2823':
            #    print(open_s1_payoff)
            if self.reward_type is None:
                if self.tax_cost == 0.0015:
                    reward = half_tax_profit    # 我們用這個
                elif self.tax_cost == 0.003:
                    reward = profit
                elif self.tax_cost == 0:
                    reward = zero_tax_profit
                else:
                    raise ValueError('tax cost error!')
            elif self.reward_type == 'complexity_reward':
                reward = int_w[i][0] * long_short * ((stock1_close_price-stock1_open_price) / stock1_open_price) + \
                         int_w[i][1] * long_short * ((stock2_close_price-stock2_open_price) / stock2_open_price)
                print(f'{self.reward_type}: {reward}')
            else:
                raise ValueError('reward type error!')


        # 把output整理出來
            performance_dict = dict()
            performance_dict['profit'] = profit
            performance_dict['half_tax_profit'] = half_tax_profit
            performance_dict['zero_tax_profit'] = zero_tax_profit
            performance_dict['reward'] = reward
            performance_dict['record'] = record
            performance_dict['close_timing'] = int(close_timing)
            performance_dict['open_timing'] = int(open_timing[i])
            performance_dict['stock1_num'] = w1
            performance_dict['stock2_num'] = w2
            performance_dict['w1'] = self.table.w1.iloc[i]
            performance_dict['w2'] = self.table.w2.iloc[i]
            performance_dict['capital'] = capital
            if self.output_stock_price:     # 要顯示當天該股票所有的價格波動
                result.append([stock1_name.iloc[i], stock2_name.iloc[i], model_type[i], mu[i], std[i],
                               list(all_trigger_spread[i, 16:]), list(all_trigger_stock1[i, 16:]),
                               list(all_trigger_stock2[i, 16:]), performance_dict,
                               self.table.w1.iloc[i], self.table.w2.iloc[i]])
            else:   # 不顯示當天該股票所有的價格波動 , 我們在這裡
                result.append([stock1_name.iloc[i], stock2_name.iloc[i], model_type[i], mu[i], std[i],
                               list(all_trigger_spread[i, 16:]), performance_dict,
                               self.table.w1.iloc[i], self.table.w2.iloc[i]])
            if self.dump:   # dump = false, 不理
                price = [stock1_open_price, stock1_close_price, stock2_open_price, stock2_close_price]
                plot_spread(self.tick, stock_date, stock1_name[i], stock2_name[i], all_trigger_spread[i], mu[i], std[i],
                            self.open_times, self.open_times, self.stop_loss_times, folder_name,
                            open_timing[i] + self.cross_threshold, close_timing + self.cross_threshold, record, reward,
                            self.trend_stationary, price, int_w[i])

    # 使用外部給的策略, 回傳當天全部pair的績效
        result = pd.DataFrame(result)
        return result


def plot_spread(tick, stock_date, stock1, stock2, spread, mu, std, up_open_time, down_open_time, stop_loss_time,
                folder_name, open_timing, close_timing, status, reward, trend_stationary, price, int_w):
    plt.rcParams.update({'font.size': 22})
    spread = spread[16:]
    fig, ax = plt.subplots(figsize=(30, 15))
    if trend_stationary:
        mu = np.array(mu)
    else:
        mu = np.array([mu] * len(spread))
    up_open = mu + up_open_time * std
    down_open = mu - down_open_time * std
    up_stop_loss = mu + stop_loss_time * std
    down_stop_loss = mu - stop_loss_time * std
    ax.plot(spread)
    ax.plot(up_open, 'b')
    ax.plot(down_open, 'b')
    ax.plot(up_stop_loss, 'r')
    ax.plot(down_stop_loss, 'r')
    ax.plot(mu, 'g')
    offset = 149
    ax.vlines(offset, min(down_stop_loss), max(up_stop_loss), 'b', linestyles='dashed')
    ax.scatter(open_timing+offset, spread[open_timing+offset], edgecolors='b', marker='o', linewidth=3,
               zorder=99, facecolors='none')
    ax.scatter(close_timing+offset, spread[close_timing+offset], edgecolors='r', marker='o', linewidth=3,
               zorder=99, facecolors='none')
    if status == 666:
        status_comment = '正常平倉'
    elif status == -2:
        status_comment = '碰到停損門檻平倉'
    elif status == -3:
        status_comment = '結構性斷裂平倉(fore_lag5)'
    elif status == -4:
        status_comment = '時間結束，強迫平倉'
    else:
        status_comment = 'Error'
    plt.title(stock_date + ' s1:' + stock1 + ' s2:' + stock2 + ' 上開倉門檻:' + str(up_open_time) + ' 下開倉門檻:'
              + str(down_open_time) + ' 停損門檻:' + str(stop_loss_time))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.85,
             f'profit:{reward:.4f} {status_comment}'
             f'\nopen price: s1={price[0]},s2={price[2]}'
             f'\nclose price: s1={price[1]},s2={price[3]}'
             f'\nint_w: s1={int_w[0]},s2={int_w[1]}'
             f'\ntiming index: open={open_timing}, close={close_timing}', bbox=props, transform=ax.transAxes)
    if tick:
        file_comment = 'matrix_tick'
    else:
        file_comment = 'matrix'
    if open_timing == close_timing:
        file_comment += 'same_time'
    if not os.path.exists(f'{PTDQN_DATA_FOLDER}/' + folder_name + '/jpg'):
        os.makedirs(f'{PTDQN_DATA_FOLDER}/' + folder_name + '/jpg')
    plt.tight_layout()
    plt.savefig(f'{PTDQN_DATA_FOLDER}/' + folder_name + '/jpg/' + stock_date + '_' + stock1 + '_' + stock2 + '_' + str(
        up_open_time) + '_' + str(down_open_time) + '_' + str(stop_loss_time) + '_' + file_comment + '.jpg')
    plt.close('all')
