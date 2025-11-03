import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import math
from scipy.stats import norm, gumbel_r
import time
import datetime

from numba import njit


@njit
def polevl(x, coefs, N):
    ans = 0
    power = len(coefs) - 1
    for coef in coefs[:N]:
        ans += coef * x**power
        power -= 1

    return ans
@njit
def p1evl(x, coefs, N):
    return polevl(x, [1] + coefs, N)
@njit
def my_inv_erf(z):
    if z < -1 or z > 1:
        raise ValueError("`z` must be between -1 and 1 inclusive")

    if z == 0:
        return 0
    if z == 1:
        return math.inf
    if z == -1:
        return -math.inf

    # From scipy special/cephes/ndrti.c
    def ndtri(y):
        # approximation for 0 <= abs(z - 0.5) <= 3/8
        P0 = [
            -5.99633501014107895267E1,
            9.80010754185999661536E1,
            -5.66762857469070293439E1,
            1.39312609387279679503E1,
            -1.23916583867381258016E0,
        ]

        Q0 = [
            1.95448858338141759834E0,
            4.67627912898881538453E0,
            8.63602421390890590575E1,
            -2.25462687854119370527E2,
            2.00260212380060660359E2,
            -8.20372256168333339912E1,
            1.59056225126211695515E1,
            -1.18331621121330003142E0,
        ]

        # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
        # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
        P1 = [
            4.05544892305962419923E0,
            3.15251094599893866154E1,
            5.71628192246421288162E1,
            4.40805073893200834700E1,
            1.46849561928858024014E1,
            2.18663306850790267539E0,
            -1.40256079171354495875E-1,
            -3.50424626827848203418E-2,
            -8.57456785154685413611E-4,
        ]

        Q1 = [
            1.57799883256466749731E1,
            4.53907635128879210584E1,
            4.13172038254672030440E1,
            1.50425385692907503408E1,
            2.50464946208309415979E0,
            -1.42182922854787788574E-1,
            -3.80806407691578277194E-2,
            -9.33259480895457427372E-4,
        ]

        # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
        # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
        P2 = [
            3.23774891776946035970E0,
            6.91522889068984211695E0,
            3.93881025292474443415E0,
            1.33303460815807542389E0,
            2.01485389549179081538E-1,
            1.23716634817820021358E-2,
            3.01581553508235416007E-4,
            2.65806974686737550832E-6,
            6.23974539184983293730E-9,
        ]

        Q2 = [
            6.02427039364742014255E0,
            3.67983563856160859403E0,
            1.37702099489081330271E0,
            2.16236993594496635890E-1,
            1.34204006088543189037E-2,
            3.28014464682127739104E-4,
            2.89247864745380683936E-6,
            6.79019408009981274425E-9,
        ]

        s2pi = 2.50662827463100050242
        code = 1

        if y > (1.0 - 0.13533528323661269189):      # 0.135... = exp(-2)
            y = 1.0 - y
            code = 0

        if y > 0.13533528323661269189:
            y = y - 0.5
            y2 = y * y
            x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8))
            x = x * s2pi
            return x

        x = math.sqrt(-2.0 * math.log(y))
        x0 = x - math.log(x) / x

        z = 1.0 / x
        if x < 8.0:                 # y > exp(-32) = 1.2664165549e-14
            x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8)
        else:
            x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8)

        x = x0 - x1
        if code != 0:
            x = -x

        return x

    result = ndtri((z + 1) / 2.0) / math.sqrt(2)

    return result

@njit
def my_3x3_det(m1, m2, m3, m4, m5, m6, m7, m8, m9):
    return m1 * m5 * m9 + m4 * m8 * m3 + m7 * m2 * m6 - m1 * m6 * m8 - m3 * m5 * m7 - m2 * m4 * m9

@njit
def my_3x3_inv(m1, m2, m3, m4, m5, m6, m7, m8, m9):
    determinant = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9
    return np.array([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                     [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                     [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]])/determinant


'''For Gumble distribution assumption'''

@njit
def GumbelPDF_jit(x, u=0, s=1, eps=1e-4):
    if eps>s:
        s=eps
    z = (x - u) / s
    return 1/s * np.exp(-(z + np.exp(-z)) )

def GumbelPDF(x, u=0, s=1, eps=1e-4, do_torch=False):
    if do_torch:
        z = (x - u) / s
        return 1/torch.max(s,eps) * torch.exp(-(z + torch.exp(-z)) )
    else:
        if eps > s:
            s = eps
        z = (x - u) / s
        return 1/np.max(np.array([s,eps])) * np.exp(-(z + np.exp(-z)) )


@njit
def Gumbel_distribution_q_jit(u=0, s=1, q=0.95):
    return u - s*(np.log(-np.log(q)))

def Gumbel_distribution_q(u=0, s=1, q=0.95, do_torch=False):
    if do_torch:
        return torch.tensor(u) - torch.tensor(s)*(torch.log(-torch.log(torch.tensor(q))))
    else:
        gumbel_dist = gumbel_r(loc=u, scale=s)
        gumbel_dist.ppf(q)
        return gumbel_dist.ppf(q)


@njit
def GumbelCDF_jit(x, u=0, s=1, eps=1e-4):
    if eps > s:
        s = eps
    return np.exp(-np.exp(-(x - u) / s))

def GumbelCDF(x, u=0, s=1, eps=1e-4, do_torch=False):
    if do_torch:
        return torch.exp(-torch.exp(-(x - u) / torch.max(s,eps)))
    else:
        if eps > s:
            s=eps
        return np.exp(-np.exp(-(x - u) / s))


def expect_profit_1st_term_Gumbel(x, rtop_u, rtop_b, c):
    return (1-GumbelCDF(x, rtop_u, rtop_b))*(x-c)

def neg_expect_profit_Gumbel(x, rtop_u, rtop_b, top_u, top_b, f, c,
                                 adj=True):
    x=x[0]
    CDFrx = GumbelCDF(x, rtop_u, rtop_b)
    CDFtx = GumbelCDF(x, top_u, top_b)
    if adj:
        return -((1 - CDFrx) * (x - c) + (1 - CDFtx) * CDFrx * (-c - f))
    else:
        return -((1 - CDFrx) * (x - c) + (1 - CDFtx) * CDFrx * (x - c - f))


'''For Normal distribution assumption'''

@njit
def NormalPDF_jit(x, u=0, s=1, eps=1e-4):
    if eps > s:
        s = eps
    return 1/np.sqrt(2 * np.pi * np.square(s)) * np.exp( -1/2 * np.square((x-u)/s) )


def NormalPDF(x, u=0, s=1, eps=1e-4, do_torch=False):
    if do_torch:
        return 1/torch.max(torch.sqrt(2 * torch.pi * s**2),eps) * torch.exp(-1/2*torch.square((x-u)/torch.max(s,eps)) )
    else:
        if eps > s:
            s = eps
        return 1/np.sqrt(2 * np.pi * np.square(s)) * np.exp( -1/2 * np.square((x-u)/s) )


@njit
def Normal_distribution_q_jit(u, s, q=0.95):
    return u+s*np.sqrt(2)*my_inv_erf(2*q-1)

def Normal_distribution_q(u, s, q=0.95, do_torch=False):
    if do_torch:
        q=torch.clip(q,0.01,0.99)
        result = u + s*torch.sqrt(torch.tensor(2))*torch.erfinv(2*q-1)
        return result

    else:
        normal_dist = norm(loc=u, scale=s)
        normal_dist.ppf(q)
        return normal_dist.ppf(q)


@njit
def NormalCDF_jit(x, u, s, eps=1e-4):
    if eps > s:
        s = eps
    return (1.0 + math.erf((x - u) / np.sqrt(2.0) / s)) / 2.0

def NormalCDF(x, u, s, eps=1e-4, do_torch=False):
    if do_torch:
        return (torch.tensor(1.0) + torch.erf((x - u) / torch.max(torch.sqrt(torch.tensor(2.0))*s, eps))) / torch.tensor(2.0)
    else:
        if eps > s:
            s = eps
        return (1.0 + scipy.special.erf((x - u) / np.sqrt(2.0) / s)) / 2.0


def neg_expect_profit_Normal(x, rtop_u, rtop_b, top_u, top_b, f, c,
                                 adj=True):
    CDFrx = NormalCDF(x, rtop_u, rtop_b)
    CDFtx = NormalCDF(x, top_u, top_b)
    if adj:
        return -((1 - CDFrx) * (x - c) + (1 - CDFtx) * CDFrx * (-c - f))
    else:
        return -((1 - CDFrx) * (x - c) + (1 - CDFtx) * CDFrx * (x - c - f))




@njit
def Gaussian_Copula_Density(m1, m2, m3, m4, m5, m6, m7, m8, m9, q,
                            eps=1e-4):
    detR = my_3x3_det(m1, m2, m3, m4, m5, m6, m7, m8, m9)

    quantile_vector = np.array([[Normal_distribution_q_jit(0, 1, q[0])],[Normal_distribution_q_jit(0, 1, q[1])],[Normal_distribution_q_jit(0, 1, q[2])]])

    exp_term = -1/2 * (np.dot(np.dot(quantile_vector.T, ((my_3x3_inv(m1, m2, m3, m4, m5, m6, m7, m8, m9)-np.eye(3)))),quantile_vector))

    if np.sqrt(detR) < eps:
        cGaussian = 1/eps * np.exp(exp_term)
    else:
        cGaussian = 1 /np.sqrt(detR) * np.exp(exp_term)
    return cGaussian[0,0]

@njit
def joint_Gumbel_pdf(m1, m2, m3, m4, m5, m6, m7, m8, m9, x1, u1, s1, x2, u2, s2, x3, u3, s3,
                     eps=1e-4):

    q1 = GumbelCDF_jit(x1, u=u1, s=s1, eps=eps)
    q2 = GumbelCDF_jit(x2, u=u2, s=s2, eps=eps)
    q3 = NormalCDF_jit(x3, u=u3, s=s3, eps=eps)
    q = np.array([q1, q2, q3])

    return (Gaussian_Copula_Density(m1, m2, m3, m4, m5, m6, m7, m8, m9, q, eps=eps)
                * GumbelPDF_jit(x1, u=u1, s=s1, eps=eps)
                * GumbelPDF_jit(x2, u=u2, s=s2, eps=eps)
                * NormalPDF_jit(x3, u=u3, s=s3, eps=eps))

@njit
def joint_Normal_pdf(m1, m2, m3, m4, m5, m6, m7, m8, m9, x1, u1, s1, x2, u2, s2, x3, u3, s3,
                     eps=1e-4):

    q1 = NormalCDF_jit(x1, u=u1, s=s1, eps=eps)
    q2 = NormalCDF_jit(x2, u=u2, s=s2, eps=eps)
    q3 = NormalCDF_jit(x3, u=u3, s=s3, eps=eps)
    q = np.array([q1, q2, q3])

    return (Gaussian_Copula_Density(m1, m2, m3, m4, m5, m6, m7, m8, m9, q, eps=eps)
                * NormalPDF_jit(x1, u=u1, s=s1, eps=eps)
                * NormalPDF_jit(x2, u=u2, s=s2, eps=eps)
                * NormalPDF_jit(x3, u=u3, s=s3, eps=eps))


@njit
def profit_2nd_term_Gumbel(T, x, y, z, ux, sx, uy, sy, uz, sz, c, m1, m2, m3, m4, m5, m6, m7, m8, m9,
                           adj=True):
    j = joint_Gumbel_pdf(m1, m2, m3, m4, m5, m6, m7, m8, m9, x, ux, sx, y, uy, sy, z, uz, sz, eps=1e-4)

    if adj:
        return j*(-z-c)
    else:
        return j * (T - z - c)


@njit
def profit_2nd_term_Normal(T, x, y, z, ux, sx, uy, sy, uz, sz, c, m1, m2, m3, m4, m5, m6, m7, m8, m9,
                           adj=True):
    j = joint_Normal_pdf(m1, m2, m3, m4, m5, m6, m7, m8, m9, x, ux, sx, y, uy, sy, z, uz, sz, eps=1e-4)

    if adj:
        return j * (-z - c)
    else:
        return j * (T - z - c)

@njit
def expect_profit_Gumbel_integral(T, m1, m2, m3, m4, m5, m6, m7, m8, m9, ux, sx, uy, sy, uz, sz, c,
                                  adj=True, grain=100):
    x_min, x_max = Gumbel_distribution_q_jit(u=ux, s=sx, q=0.01), T
    y_min, y_max = T, Gumbel_distribution_q_jit(u=uy, s=sy, q=0.99)
    z_min, z_max = Normal_distribution_q_jit(u=uz, s=sz, q=0.01), Normal_distribution_q_jit(u=uz, s=sz, q=0.99)

    if (x_min > x_max) or (y_min > y_max):
        # the probability of second term is too small, return the first term
        return (1 - GumbelCDF_jit(T, ux, sx)) * (T - c)

    dx = (x_max - x_min)/grain
    dy = (y_max - y_min)/grain
    dz = (z_max - z_min)/grain
    riemann_sum = 0

    for i in range(grain):
        for j in range(grain):
            for k in range(grain):
                x = x_min + i*dx + dx/2
                y = y_min + j*dy + dy/2
                z = z_min + k*dz + dz/2
                riemann_sum += profit_2nd_term_Gumbel(T, x, y, z, ux, sx, uy, sy, uz, sz, c, m1, m2, m3, m4, m5, m6, m7, m8, m9, adj)
    riemann_sum *= dx*dy*dz

    return (1-GumbelCDF_jit(T, ux, sx))*(T - c) + riemann_sum

@njit
def expect_profit_Normal_integral(T, m1, m2, m3, m4, m5, m6, m7, m8, m9, ux, sx, uy, sy, uz, sz, c,
                                  adj=True, grain=100):
    x_min, x_max = Normal_distribution_q_jit(u=ux, s=sx, q=0.01), T
    y_min, y_max = T, Normal_distribution_q_jit(u=uy, s=sy, q=0.99)
    z_min, z_max = Normal_distribution_q_jit(u=uz, s=sz, q=0.01), Normal_distribution_q_jit(u=uz, s=sz, q=0.99)

    if (x_min > x_max) or (y_min > y_max):
        # the probability of second term is too small, return the first term
        return (1 - NormalCDF_jit(T, ux, sx)) * (T - c)

    dx = (x_max - x_min)/grain
    dy = (y_max - y_min)/grain
    dz = (z_max - z_min)/grain
    riemann_sum = 0

    for i in range(grain):
        for j in range(grain):
            for k in range(grain):
                x = x_min + i*dx + dx/2
                y = y_min + j*dy + dy/2
                z = z_min + k*dz + dz/2
                riemann_sum += profit_2nd_term_Normal(T, x,y,z, ux, sx, uy, sy, uz, sz, c, m1, m2, m3, m4, m5, m6, m7, m8, m9, adj=adj)
    riemann_sum *= dx*dy*dz

    return (1-NormalCDF_jit(T, ux, sx))*(T - c) + riemann_sum

@njit
def find_opt_open_threshold(m1, m2, m3, m4, m5, m6, m7, m8, m9, ux, sx, uy, sy, uz, sz, c,
                            mode="Gumbel", adj=True, opt_grain=100, integral_grain=10, early_stop=5):
    X = np.linspace(0, ux, opt_grain)
    max_val = -1000000
    max_x = -1
    flag = 0

    for x in X:
        if mode == "Gumbel":
            result = expect_profit_Gumbel_integral(x,
                                                   m1, m2, m3,
                                                   m4, m5, m6,
                                                   m7, m8, m9,
                                                   ux, sx,
                                                   uy, sy,
                                                   uz, sz,
                                                   c,
                                                   adj,
                                                   grain=integral_grain)
        elif mode == "Normal":
            result = expect_profit_Normal_integral(x,
                                                   m1, m2, m3,
                                                   m4, m5, m6,
                                                   m7, m8, m9,
                                                   ux, sx,
                                                   uy, sy,
                                                   uz, sz,
                                                   c,
                                                   adj,
                                                   grain=integral_grain)
        else:
            raise ValueError("mode sould be either 'Gumbel' or 'Normal'.")

        if result > max_val:
            max_val = result
            max_x = x
            flag = 0
        else:
            flag +=1
        if flag >= early_stop:
            return max_x, max_val
    return max_x, max_val



'''--------------------------------------------------------------------------------'''

def check_nan(df):
    if df.isnull().values.any():
        count_null = 0
        for i in range(len(df)):
            if df.iloc[i].isnull().values.any():
                count_null+=1
        print("Before delete nan, len:",len(df))
        df = df.dropna()
        print("After delete nan, len:",len(df))
        raise ValueError("Count null entries:", count_null)

def get_time():
    main_time = time.time()
    main_time = datetime.datetime.fromtimestamp(main_time)
    main_time = main_time.strftime("%Y-%m-%d %H-%M-%S")
    return main_time


def formation_spread_touch_mean_once(formation_period_spread, spread_mean):
    sign = formation_period_spread[0] - spread_mean
    for j in range(len(formation_period_spread)):
        if (formation_period_spread[j] - spread_mean) * sign <= 0:
            return True
    return False


def price_to_return(price):
    stock_return = price.copy()
    for j in range(1, len(stock_return)):
        stock_return[j] = (stock_return[j] / stock_return[0]) - 1
    stock_return[0] = 0
    return stock_return


def find_PRtop(trading_period_norm_spread):

    # find tr
    tr = -1
    if trading_period_norm_spread[-1] < 0:
        sign = -1
    elif trading_period_norm_spread[-1] > 0:
        sign = 1
    else: #trading_period_norm_spread[-1] == 0
        return np.max(np.abs(trading_period_norm_spread))

    for j in range(2, len(trading_period_norm_spread) + 1):
        if trading_period_norm_spread[-j] * sign < 0:  # The last time to cross zero was at time len(trading_period_norm_spread)-j+1
            tr = len(trading_period_norm_spread) - j + 1
            break

    if tr == -1:  # no return to mean
        PRtop = -1  # if Prtop = 0 then might get undefined behavior of reward function at threshold = 0
    else:
        PRtop = np.max(np.abs(trading_period_norm_spread[:tr]))
    return PRtop


def timer(i, data_size, start, precision=10000):
    if i == 0:
        cond1 = False
    elif (data_size // precision) == 0:
        cond1 = True
    else:
        cond1 = (i % (data_size // precision) == 0 and i != 0)

    if cond1:
        current_time = time.time() - start
        progress = i / data_size * 100
        estimated_total_time = current_time * 100 / progress if progress > 0 else 0

        current_min, current_sec = divmod(int(current_time), 60)
        total_min, total_sec = divmod(int(estimated_total_time), 60)
        remain_min, remain_sec = divmod(int(estimated_total_time - current_time), 60)

        print(f"Back testing... {progress:.{len(str(precision)) - 3}f}%",
              f"    {current_min:02}:{current_sec:02} /",
              f"{total_min:02}:{total_sec:02}",
              f"(Remaining: {remain_min:02}:{remain_sec:02})")


def calculate_MDD(cumulative_profit, percent=False):
    if percent:
        MDD = 0
        for i in range(len(cumulative_profit)):
            for j in range(i):
                if (cumulative_profit[j] - cumulative_profit[i]) / cumulative_profit[j] > MDD:
                    MDD = (cumulative_profit[j] - cumulative_profit[i]) / cumulative_profit[j]
    else:
        max_profit = -10000000
        min_profit = 10000000
        MDD = 0
        for p in cumulative_profit:
            if p > max_profit:
                max_profit = p
                min_profit = 10000000
            elif p < min_profit:
                min_profit = p
            if min_profit - max_profit < MDD:
                MDD = min_profit - max_profit
    return MDD


def calculate_Sharpe_Ratio(daily_return):
    return np.mean(daily_return) / np.std(daily_return) * np.sqrt(252)


def calculate_Sortino_Ratio(daily_return):
    negative_returns = np.array([ret for ret in daily_return if ret < 0])
    DD = np.sqrt(np.sum(np.square(negative_returns)) / len(daily_return))
    if DD == 0:
        return 0
    return np.mean(daily_return) / DD * np.sqrt(252)

if __name__ == '__main__':
    pass
