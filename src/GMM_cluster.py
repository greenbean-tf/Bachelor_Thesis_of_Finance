import pickle
import utils
import preprocess
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


def main():
    print("START PREPROCESSING...")
    PathRoot = r"C:\Users\nycu_dev1\Desktop\YEN-HSU Code\preprocess_data_forLLTL\\"
    output_path = r"C:\Users\nycu_dev1\Desktop\YEN-HSU Code\clustered_data_forLLTL\\"

    preprocess_data1 = pd.read_pickle(PathRoot +  f'preprocess_data_{2020}_{10}_to_{2021}_{9}.pickle')
    preprocess_data2 = pd.read_pickle(PathRoot + f'preprocess_data_{2021}_{10}_to_{2022}_{9}.pickle')
    preprocess_data3 = pd.read_pickle(PathRoot + f'preprocess_data_{2022}_{10}_to_{2023}_{10}.pickle')
    preprocess_data1.reset_index(drop=True, inplace=True)
    preprocess_data2.reset_index(drop=True, inplace=True)
    preprocess_data3.reset_index(drop=True, inplace=True)
    preprocess_data1 = preprocess_data1.dropna()
    preprocess_data2 = preprocess_data2.dropna()
    preprocess_data3 = preprocess_data3.dropna()


    train=np.column_stack((preprocess_data1['Johensen_intercept'], preprocess_data1['Johensen_std']))
    t1=np.column_stack((preprocess_data2['Johensen_intercept'], preprocess_data2['Johensen_std']))
    t2=np.column_stack((preprocess_data3['Johensen_intercept'], preprocess_data3['Johensen_std']))
    print("DONE PREPROCESSING...")


    #train
    print("START TRANING...")
    n_components = 3  #3 clusters
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(train)

    train_labels = gmm.predict(train)
    preprocess_data1['GMM_label'] = train_labels
    for cluster_id in range(n_components):
        cluster_data = preprocess_data1[preprocess_data1['GMM_label'] == cluster_id]
        cluster_data.to_pickle(output_path +  f'preprocess_data_{2020}_{10}_to_{2021}_{9}_{cluster_id}.pickle')
        print(output_path +  f'preprocess_data_{2020}_{10}_to_{2021}_{9}_{cluster_id}.pickle saved') 

    #test1
    test_scores = gmm.score(t1)  
    test_labels = gmm.predict(t1)
    test_silhouette = silhouette_score(t1, test_labels)
    print("TEST1 SCORE")
    print(f"Log Likelyhood: {test_scores}")
    print(f"Silhoutte score: {test_silhouette}")
    
    preprocess_data2['GMM_label'] = test_labels
    for cluster_id in range(n_components):
        cluster_data = preprocess_data2[preprocess_data2['GMM_label'] == cluster_id]
        cluster_data.to_pickle(output_path +  f'preprocess_data_{2021}_{10}_to_{2022}_{9}_{cluster_id}.pickle')
        print(output_path +  f'preprocess_data_{2021}_{10}_to_{2022}_{9}_{cluster_id}.pickle saved') 

    #test2
    test_scores = gmm.score(t2)  
    test_labels = gmm.predict(t2)  
    test_silhouette = silhouette_score(t2, test_labels)
    print("TEST2 SCORE")
    print(f"Log Likelyhood: {test_scores}")
    print(f"Silhoutte score: {test_silhouette}")

    preprocess_data3['GMM_label'] = test_labels
    for cluster_id in range(n_components):
        cluster_data = preprocess_data3[preprocess_data3['GMM_label'] == cluster_id]
        cluster_data.to_pickle(output_path +  f'preprocess_data_{2022}_{10}_to_{2023}_{10}_{cluster_id}.pickle')
        print(output_path +  f'preprocess_data_{2022}_{10}_to_{2023}_{10}_{cluster_id}.pickle saved') 

if __name__=="__main__":
    main()


"""
TEST1 SCORE (training 2018/10, testing 2019/10)
Log Likelyhood: 5.670668796843494
Silhoutte score: 0.6972950775064698
TEST2 SCORE (training 2018/10, testing 2020/10)
Log Likelyhood: 6.141572155629402
Silhoutte score: 0.7435604537522517


"""