# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 23:57:28 2020

@author: John Hao Han
"""

import pandas as pd
import numpy as np
import numpy.matlib
from scipy.linalg import eigh
import math


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



def Binal_comb(pool):
    # Binal_comb(range(3)) = [[0,1],[0,2],[1,2]] dytp=narray
    # 排列組合不重複的數列，專門處理Cn取2
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
    xt = np.asmatrix(xt)
    zt = np.asmatrix(zt)

    beta = ( xt.T * xt ).I * xt.T * zt                      # 計算VAR的參數
    
    A = zt - xt * beta                                      # 計算殘差
    sigma = ( (A.T) * A ) / (n-p)                           # 計算殘差的共變異數矩陣
        
    return [ sigma , beta ]

# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數--------------------------------------------------------------
def order_select( y , max_p ):
    
    k = len(y.T)     # 幾檔股票
    n = len(y)       # 資料長度
    
    bic = np.zeros((max_p,1))
    for p in range(1,max_p+1):
        sigma = VAR_model( y , p )[0]
        bic[p-1] = np.log( np.linalg.det(sigma) ) + np.log(n) * p * (k*k) / n
    bic_order = int(np.where(bic == np.min(bic))[0][0] + 1)     # 因為期數p從1開始，因此需要加1
    
    return bic_order


# Johanse test與VECM模型參數-------------
def JCItestModel5( Xt , opt_q , alpha ):
    # 防呆檢查
    if alpha == 0.05:
        testCi = 0
    elif alpha == 0.01:
        testCi = 1
    else:
        return 'alpha should be 0.05 or 0.01.'
    
    [NumObs,NumDim] = Xt.shape
    
    if NumObs < NumDim:
        return 'Xt must be a T*NumDim matrix and T>NumDim. '
    elif NumDim > 2:
        return 'Only 2 NumDim can work.'
    elif opt_q < 1:
        return 'Set x higher than 1 '
    
    #最適落後期數VAR(p)應該要先用order_select 的BIC找最適p，且VEC(q)，q=p-1
    #Xt=NumObs*NumDim的矩陣，NumObs為觀察值（t=1,2,...,NumObs），NumDim為股票個數

    #設定計算用參數
    p = opt_q+1
    T = NumObs-p
    dY_ALL = Xt[1:,:]-Xt[0:-1,:]
    dY = dY_ALL[p-1:,:]
    Ys = Xt[p-1:-1,:]
    dX = np.zeros([T,NumDim*(p-1)])
    
    for xi in range(p-1):
        dX[:,xi*NumDim:(xi+1)*NumDim] = dY_ALL[p-xi-2:NumObs-xi-2,:]

    #H5增加截距與時間趨勢
    dX = np.hstack( ( dX, np.ones((T,1)) , np.arange(1,T+1,1).reshape(T,1) ) )
    '''
    #H4增加截距與時間趨勢
    dX = np.hstack( ( dX, np.ones((T,1)) ) )
    Ys = np.hstack( ( Ys, np.arange(1,T+1,1).reshape(T,1)))
    '''
    #轉成matrix，計算比較直觀
    dX , dY , Ys = np.asmatrix(dX) , np.asmatrix(dY), np.asmatrix(Ys)
    
    #先求dX'*dX 方便下面做inverse
    DX = dX.T * dX
    #I-dX * (dX'*dX)^-1 * dX'
    M = np.identity(T)-dX*DX.I*dX.T
    
    #matrix 跟numpy 都可以用下面這行，跟上面是同樣的計算
    #M = np.identity(T) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )
    
    R0 , R1= dY.T * M , Ys.T * M
    S00 = R0 * R0.T / T
    S01 = R0 * R1.T / T
    S10 = R1 * R0.T / T
    S11 = R1 * R1.T / T
    
    eigVals, eigvecs = eigh(S10 * S00.I * S01 , S11 , eigvals_only=False)
    '''
    #如果要驗算特徵值與特徵矩陣，MustBeZero越接近0越好
    eig_d = np.zeros((NumDim,NumDim))
    for e in range(NumDim):
        eig_d[e,e] = eig_D[e]
    
    MustBeZero = np.dot(Sxx , eig_V)-np.dot(np.dot(S11 , eig_V), eig_d)
    '''
    #排序特徵向量與特徵值
    sort_ind = np.argsort(-eigVals)
    eigVals = eigVals[sort_ind] 
    eigVecs = eigvecs[:,sort_ind]
    eigVals = eigVals.reshape(NumDim,1)
    
    #VECM各項參數
    JCIstat	= [[0]*11 for i in range(NumDim)]
    JCIstat[0][:]= ['','A','B','c0','d0','c1','d1','Bq','eigValue','eigVector','testStat']
    
    #是否通過檢定，True=通過，False=沒通過
    H = [[0]*2 for i in range(NumDim)]
    
    #CVTableRow 跟CVTable 都是算pValue的表格，但是因為CVTable表不精確，Pvalue等之後比較精確再來做
    #CVTableRow = [95, 99]  95=95% , 99=99%
    #CVTable = [ (k-r) , criterion percentage]
    CVTable = [[3.8415, 6.6349],
               [18.3969, 23.1574],
               [35.0131, 41.0722]]
    
    for rn in range(1,NumDim):
        B = np.asmatrix(eigVecs[:,0:rn])
        A = S01*B
        W = dY-Ys*B*A.T
        P = dX.I * W # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = A.I * c
        c1 = c - A * c0
        d = P[:,-1]
        d0 = A.I * d
        d1 = d - A * d0
        
        Bq = [[0]*2 for i in range(opt_q)]
        for bi in range(1,opt_q+1):
            Bq[bi-1][0] = 'B'+str(bi)
            Bq[bi-1][1] = P[:,((bi-1)*NumDim):bi*NumDim]
            
        JCIstat[rn][0] = ['r'+str(rn)]
        JCIstat[rn][1] = A[:,0]
        JCIstat[rn][2] = B[:,0]
        JCIstat[rn][3] = c0
        JCIstat[rn][4] = d0
        JCIstat[rn][5] = c1
        JCIstat[rn][6] = d1
        JCIstat[rn][7] = Bq
        JCIstat[rn][8] = eigVals[rn,:]
        JCIstat[rn][9] = eigVecs[:,rn]
        #如果需要Residual = W - dX * P.T
        eig_lambda = np.cumprod(1-eigVals[rn:NumDim,:])
        JCIstat[rn][10] = -2 * np.log(eig_lambda[-1] ** (T/2))
        
        H[rn][0] = ['h'+str(rn)]
        H[rn][1] = CVTable[NumDim-rn-1][testCi] < JCIstat[rn][10]
        '''
        #因為統計檢定量的分配數字還不夠精確，Pvalue的計算暫緩
        CVraw = CVTable[NumDim-rn-1] 
        pvalue = np.interp(JCIstat[rn][11], CVraw , CVTableRow)
        pvalue = 1-(pvalue/100)
        '''
        return [H, JCIstat]


# Johanse test與VECM模型參數-------------
def JCItestModel1(Xt, opt_q, alpha):
    # 防呆檢查
    if alpha == 0.05:
        testCi = 0
    elif alpha == 0.01:
        testCi = 1
    else:
        return 'alpha should be 0.05 or 0.01.'

    [NumObs, NumDim] = Xt.shape

    if NumObs < NumDim:
        return 'Xt must be a T*NumDim matrix and T>NumDim. '
    elif NumDim > 2:
        return 'Only 2 NumDim can work.'
    elif opt_q < 1:
        return 'Set x higher than 1 '

    # 最適落後期數VAR(p)應該要先用order_select 的BIC找最適p，且VEC(q)，q=p-1
    # Xt=NumObs*NumDim的矩陣，NumObs為觀察值（t=1,2,...,NumObs），NumDim為股票個數

    # 設定計算用參數
    p = opt_q + 1
    T = NumObs - p
    dY_ALL = Xt[1:, :] - Xt[0:-1, :]
    dY = dY_ALL[p - 1:, :]
    Ys = Xt[p - 1:-1, :]
    dX = np.zeros([T, NumDim * (p - 1)])

    for xi in range(p - 1):
        dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[p - xi - 2:NumObs - xi - 2, :]

    # 轉成matrix，計算比較直觀
    dX, dY, Ys = np.asmatrix(dX), np.asmatrix(dY), np.asmatrix(Ys)

    # 先求dX'*dX 方便下面做inverse
    DX = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    M = np.identity(T) - dX * DX.I * dX.T

    # matrix 跟numpy 都可以用下面這行，跟上面是同樣的計算
    # M = np.identity(T) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )

    R0, R1 = dY.T * M, Ys.T * M
    S00 = R0 * R0.T / T
    S01 = R0 * R1.T / T
    S10 = R1 * R0.T / T
    S11 = R1 * R1.T / T

    eigVals, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    '''
    #如果要驗算特徵值與特徵矩陣，MustBeZero越接近0越好
    eig_d = np.zeros((NumDim,NumDim))
    for e in range(NumDim):
        eig_d[e,e] = eig_D[e]

    MustBeZero = np.dot(Sxx , eig_V)-np.dot(np.dot(S11 , eig_V), eig_d)
    '''
    # 排序特徵向量與特徵值
    sort_ind = np.argsort(-eigVals)
    eigVals = eigVals[sort_ind]
    eigVecs = eigvecs[:, sort_ind]
    eigVals = eigVals.reshape(NumDim, 1)

    # VECM各項參數
    JCIstat = [[0] * 7 for i in range(NumDim)]
    JCIstat[0][:] = ['', 'A', 'B', 'Bq', 'eigValue', 'eigVector', 'testStat']

    # 是否通過檢定，True=通過，False=沒通過
    H = [[0] * 2 for i in range(NumDim)]

    # CVTableRow 跟CVTable 都是算pValue的表格，但是因為CVTable表不精確，Pvalue等之後比較精確再來做
    # CVTableRow = [95, 99]  95=95% , 99=99%
    # CVTable = [ (k-r) , criterion percentage]
    CVTable = [[4.1475, 6.9701],
               [12.3329, 16.2917],
               [24.3168, 29.6712]]

    for rn in range(1, NumDim):
        B = np.asmatrix(eigVecs[:, 0:rn])
        A = S01 * B
        W = dY - Ys * B * A.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T

        Bq = [[0] * 2 for i in range(opt_q)]
        for bi in range(1, opt_q + 1):
            Bq[bi - 1][0] = 'B' + str(bi)
            Bq[bi - 1][1] = P[:, ((bi - 1) * NumDim):bi * NumDim]

        JCIstat[rn][0] = ['r' + str(rn)]
        JCIstat[rn][1] = A[:, 0]
        JCIstat[rn][2] = B[:, 0]
        JCIstat[rn][3] = Bq
        JCIstat[rn][4] = eigVals[rn, :]
        JCIstat[rn][5] = eigVecs[:, rn]
        # 如果需要Residual = W - dX * P.T
        eig_lambda = np.cumprod(1 - eigVals[rn:NumDim, :])
        JCIstat[rn][6] = -2 * np.log(eig_lambda[-1] ** (T / 2))

        H[rn][0] = ['h' + str(rn)]
        H[rn][1] = CVTable[NumDim - rn - 1][testCi] < JCIstat[rn][6]
        '''
        #因為統計檢定量的分配數字還不夠精確，Pvalue的計算暫緩
        CVraw = CVTable[NumDim-rn-1] 
        pvalue = np.interp(JCIstat[rn][11], CVraw , CVTableRow)
        pvalue = 1-(pvalue/100)
        '''
        return [H, JCIstat]


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
    dX, dY, Ys = np.asmatrix(dX), np.asmatrix(dY), np.asmatrix(Ys)

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

    # ── 符號校正 ──────────────────────────────────────────────────
    # 廣義特徵值問題的特徵向量方向不唯一（v和-v都是合法解），不同平台的
    # LAPACK/BLAS實作可能選到正負號相反、但數學上同樣正確的特徵向量，
    # 導致後面算出來的Johansen intercept/slope/W1/W2跨平台不可重現
    # （VECM_q/Model不受影響，因為那兩者只依賴特徵值，不依賴特徵向量）。
    # 用「最大絕對值分量為正」這個確定性規則固定主特徵向量的符號，
    # 比「強制第一個分量為正」更穩健：不會因為第一個分量剛好接近0而不穩定。
    #
    # 注意：eigVecs[:,0] 不一定只有2維——model_type為2~5時，Ys會被多接一個
    # 截距/趨勢項欄位，讓這裡解出來的特徵向量維度變成NumDim+1（甚至更多），
    # 但後面jci_beta只取前NumDim個分量（對應股票權重，見下方[0:2]）。符號
    # 校正的錨點必須跟jci_beta取一樣的[0:NumDim]範圍，否則常常會錨定到那個
    # 多出來的截距/趨勢分量（量值通常遠大於beta本身），沒有真正固定住beta的符號。
    dominant_vec = eigVecs[:, 0]
    beta_part = dominant_vec[0:NumDim]
    sign_anchor_idx = np.argmax(np.abs(beta_part))
    if beta_part[sign_anchor_idx] < 0:
        eigVecs[:, 0] = -dominant_vec

    #將所有eigenvector同除第一行的總和
    eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2]))
   
    #eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    jci_beta = eigVecs_st[:,0][0:2].reshape(NumDim,1)

    #Alpha
    a = np.asmatrix(eigVecs_st[:,0])
    jci_a = S01 * a.T
    jci_alpha = jci_a/np.sum(np.absolute(jci_a)) 
    
    #初始化 c0, d0, c1, d1
    c0 , d0 = 0, 0
    c1 , d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])
    '''
    #標準化eigvecs以利計算co,do
    if len(eigVecs_s) ==3:
        for i in range(0,len(eigVecs_s)):
            a1 = eigVecs_s[0,i]
            a2 = eigVecs_s[1,i]
            a3 = eigVecs_s[2,i]
            total = abs(a1)+abs(a2)+abs(a3)
            eigVecs_s[0,i] = a1/total
            eigVecs_s[1,i] = a2/total
            eigVecs_s[2,i] = a3/total
    else:
         for i in range(0,len(eigVecs_s)):
            a1 = eigVecs_s[0,i]
            a2 = eigVecs_s[1,i]
            total = abs(a1)+abs(a2)
            eigVecs_s[0,i] = a1/total
            eigVecs_s[1,i] = a2/total
    '''
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

def JCItest_withTrace(X_data,model_type,lag_p):
    #trace test
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
    dX, dY, Ys = np.asmatrix(dX), np.asmatrix(dY), np.asmatrix(Ys)

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
    eigValue_lambda = eigValue_lambda[sort_ind]
   
    eigVecs = eigvecs[:, sort_ind]
    #將所有eigenvector同除第一行的總和
    #eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2])) 
   
    eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    #jci_beta = eigVecs_st[:,0][0:2].reshape(NumDim,1)
    jci_beta = eigVecs[:,0][0:2].reshape(NumDim,1)
    
    '''
    #Alpha
    a = np.asmatrix(eigVecs_st[:,0])
    jci_a = S01 * a.T
    jci_alpha = jci_a/np.sum(np.absolute(jci_a)) 
    '''
    #Alpha
    a = np.asmatrix(eigVecs[:,0])
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
        W = dY - (Ys[:,0:2] * jci_beta + numpy.matlib.repmat(c0, NumObs-lag_p-1, 1) )* jci_alpha.T
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
        W = dY - (Ys[:,0:2] * jci_beta + np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) * d0) * jci_alpha.T
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
    
    TraceTest_H = []
    TraceTest_T = []
    for rn in range(0,NumDim):
        eig_lambda = np.cumprod(1-eigValue_lambda[rn:NumDim,:])
        trace_stat = -2 * np.log(eig_lambda[-1] ** ((NumObs-lag_p-1)/2))
        TraceTest_H.append(cvalue[rn] < trace_stat)
        TraceTest_T.append(trace_stat)
    return TraceTest_H, TraceTest_T, jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat


def JCI_AutoSelection(Row_Y,opt_q):
    #論文中的BIC model selection
    [NumObs, k] = Row_Y.shape
    opt_p = opt_q + 1
    Tl = NumObs - opt_p
    
    TraceTest_table = np.zeros([5, k])
    BIC_table = np.zeros([5, 1])
    BIC_List = np.ones([5, 1]) * np.inf
    opt_model_num = 0
    for mr in range(0,5):
        tr_H, _, _, _, _, ut, _, _, _ = JCItest_withTrace(Row_Y, mr+1, opt_q)
        #把結果存起來，True是拒絕，False是不拒絕，tr_H[0]是Rank0,tr_H[1]是Rank1
        TraceTest_table[mr,:] = tr_H
        #以下計算BIC，僅計算Rank1
        eps = np.asmatrix(ut)
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



def TimeSpreadMeanStd(St, JCIpara, cw, opt_q , OpenDel):
    '''
    #Debug用的參數
    St, JCIpara, cw= rowLS  , JCIstat , np.asmatrix(CapitW[:,mi])
    opt_q,   OpenDel = opt_p-1 , OpenDrop
    '''
    cw = np.asmatrix(cw)
    
    A = JCIpara[1][1]
    Beta = JCIpara[1][2]
    C0 = JCIpara[1][3]
    D0 = JCIpara[1][4]
    C1 = JCIpara[1][5]
    D1 = JCIpara[1][6]
    Bq = JCIpara[1][7]
    
    S_hat = np.zeros([St.shape[0],2])
    S_hat[0:opt_q+1,:] = St[0:opt_q+1,:]
    S_hat = np.asmatrix(S_hat)
    
    # tf = 下面計算用的t
    tf = np.arange(-OpenDel+1,St.shape[0]-OpenDel+1)
    # 轉成matrix 方便計算
    tf = np.asmatrix(tf)
    
    for si in range(opt_q+1,St.shape[0]):
        '''
        yt = yt-1 + dy
        dy = A(Beta'*yt-1 +  c0  + d0t ) + c1 + d1t + DX 
        dy = A*Beta'*yt-1 + A*c0 + A*d0t + c1 + d1t + DX 
        '''
        # m1 = yt-1 * Beta * A'
        m1 = S_hat[si-1,:] * Beta * A.T
        # m2 = A*c0 + A*d0t
        m2 = C0.T * A.T + tf[:,si] * (D0 * A.T)
        # m3 = c1 + d1t
        m3 = C1.T + tf[:,si] * D1.T
        
        #這個迴圈是在算DX
        dXt = np.matrix([0,0])
        for bi in range(opt_q):
            dXt = dXt + (S_hat[si-bi-1,:] - S_hat[si-bi-2,:] ) * Bq[bi][1]
        
        # yt = yt-1 + dy
        S_hat[si,:] = S_hat[si-1,:] + m1 + m2 + m3 + dXt
    
    # 迭代後的均值  
    Mean_Trend = S_hat * cw.T
    # 共整合序列
    Cointer_Spread = St * cw.T
    
    '''
    #畫個圖檢查一下
    firstP = 0
    plotx = np.arange(firstP,indataNum)
    plt.plot(plotx,Cointer_Spread,Mean_Trend)
    
    '''
    
    # 將均值兩次差分
    Mean_Trend_d2 = np.zeros([St.shape[0],1])
    Mean_Trend_d2[2:] = np.diff(Mean_Trend,n=2,axis=0)
    Mean_Trend_d2[0:2] = Mean_Trend_d2[2]
    smsp = np.full([1,1], np.nan)
    # 兩次差分後夠接近0的就視為收斂
    for si in range(St.shape[0]):
        if abs(Mean_Trend_d2[si]) < 0.000001:
            smsp = si
            break
    
    if np.isnan(smsp):
        StMean = np.zeros([St.shape[0],1])
        St_Std = 0
        smsp = St.shape[0]
    else:
        StMean = Mean_Trend
        ErrorT = Mean_Trend[smsp: ,]-Cointer_Spread[smsp: ,]
        St_Std = np.sqrt(ErrorT.T * ErrorT / len(Mean_Trend[smsp: ,]))

    return [StMean, St_Std, smsp]


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
    C0 = np.asmatrix(mu[0])
    C1 = np.asmatrix(mu[2])
    D0 = np.asmatrix(mu[1])
    D1 = np.asmatrix(mu[3])
    C0 = alpha*C0 + C1 + alpha*D0 + D1
    Ct = alpha*D0 + D1
    expect_intcept = np.dot(np.dot(alpha_hat.transpose(),temp3),C0)
    expect_slope = np.dot(np.dot(alpha_hat.transpose(),temp3),Ct)
    return expect_intcept, expect_slope


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

