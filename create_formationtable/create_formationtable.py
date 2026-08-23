# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:19:53 2026

@author: Hao Han Chang
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 確保跟 mt.py 同層時可以直接 import
import mt
import time
from tqdm.auto import tqdm


form_del_min = 16 #開盤捨棄
inNum=150   #建模期間

dat_path = "/content/GGWP_US_LLTL/create_formationtable/data/"
save_path = "/content/GGWP_US_LLTL/create_formationtable/check_table/"
os.makedirs(save_path, exist_ok=True)
data_list = sorted(os.listdir(dat_path))

startime = time.time()

#data_i = 0#第data_i個檔案
for data_i in tqdm(range(len(data_list)), desc="檔案", unit="file"):
    Smin = pd.read_csv(dat_path+data_list[data_i])
    maxcompanynu = Smin.shape[1] #找出有多少檔
    ind = mt.Binal_comb(range(maxcompanynu))
    ind = np.hstack((ind,np.zeros([ind.shape[0], 8])))
    #創建ind時指定使用object，因為此表格會是文字與數字混合
    ind = pd.DataFrame(ind, columns=['S1', 'S2', 'VECM_q', 'Johansen intercept', 'Johansen slope', 'Johansen std', 'Model', 'W1', 'W2', 'Del_min'], dtype=object)
    col_name = {i : name for i, name in enumerate(Smin.columns.values)}
    LSmin = np.log(Smin.iloc[form_del_min:,:]) #捨棄前16分鐘股價，然後取log
    LSmin = LSmin.reset_index(drop=True)
    ObsNum=Smin.shape[1]

    # 第mi對股價的相關資料
    for mi in tqdm(range(ind.shape[0]), desc=data_list[data_i][:10], unit="pair", leave=False):
        rowS = LSmin.iloc[0:inNum,[int(ind.iloc[mi,0]),int(ind.iloc[mi,1])]] #前150分鐘的一對股價
        rowAS = np.array(rowS)
        p = mt.order_select( rowAS , max_p=5 )
    
        # portmanteau test
        model = VAR(rowAS)
        if model.fit(p).test_whiteness( nlags = p + 1 ).pvalue < 0.05: 
            continue #如果沒通過portmanteau test，捨棄該配對
        # Normality test
        if model.fit(p).test_normality().pvalue < 0.05:
            continue #如果沒通過Normality test，捨棄該配對
        
        opt_model = mt.JCI_AutoSelection(rowAS,p-1)  #bic based model selection
        if opt_model <=0:
            continue #最適模型等於0代表該配對無共整合關係，捨棄該配對
        
        ind.iloc[mi,2] = p-1
        ind.iloc[mi,6] = opt_model  
        
        F_a, F_b, F_ct, F_ut, F_gam,ct,omega_hat = mt.JCItestpara_spilCt(rowAS,opt_model,p-1)
        
        CW=F_b/np.sum( np.absolute(F_b))
        ind.iloc[mi,7] = CW[0]
        ind.iloc[mi,8] = CW[1]
        
        Johansen_intcept, Johansen_slope = mt.Johansen_mean(F_a,F_b,F_gam,F_ct,p-1)
        Johansen_var_correct = mt.Johansen_std_correct(F_a,F_b,F_ut, F_gam,p-1)
    
        if Johansen_var_correct < 0:
            continue #如果Johansen_var小於0，捨棄該配對
        Johansen_std = np.sqrt(Johansen_var_correct)
        
        ind.iloc[mi,3] = Johansen_intcept[0,0] #後面加[0,0]是希望只存數值，否則存檔中會有中括號
        ind.iloc[mi,4] = Johansen_slope[0,0]
        ind.iloc[mi,5] = Johansen_std[0,0]
        
        ind.iloc[mi,9] = form_del_min
        ind.iloc[mi,0] = col_name.get(int(ind.iloc[mi,0]))
        ind.iloc[mi,1] = col_name.get(int(ind.iloc[mi,1]))
        
    Table = ind.loc[ind['Model']>0] #排除沒有共整合關係的配對
    
    saveCSVfilename=data_list[data_i][:10]+"_CheckTable.csv"
    full_path = os.path.join(save_path, saveCSVfilename)
    Table.to_csv(full_path, index=False)
    
    now = time.time()
    tqdm.write("已完成第"+str(data_i+1)+"個檔案"+"總共耗時"+str(int(now - startime))+"秒")

