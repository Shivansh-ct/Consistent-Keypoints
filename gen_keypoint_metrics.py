#!/usr/bin/env python
# coding: utf-8

# In[95]:


import argparse
import pandas as pd
import os
import itertools
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats
import math


# In[96]:


def avg_max_t(X, meu=0):
    # rows of x are number of subjects and columns keypoints
    # meu represents the population mean assumed under some hypothesis
    # m is the number of samples, x_bar is sample mean, sigma is sample variance
    t_arr = np.zeros((X.shape[1]))
    p_arr = np.zeros((X.shape[1]))
    for j in range(X.shape[1]):  
        # print(stats.ttest_1samp(X[:, j], popmean=meu))
        # print(X[:, j])
        t_arr[j] = stats.ttest_1samp(X[:, j], popmean=meu).statistic
        p_arr[j] = stats.ttest_1samp(X[:, j], popmean=meu).pvalue
        if math.isnan(t_arr[j]):
            print(t_arr[j])
            t_arr[j] = 0
            p_arr[j] = 1
        # print(t_arr[j])
    # t_arr = np.nan_to_num(t_arr)
    t_arr = np.absolute(t_arr)
    return [np.average(t_arr), np.amax(t_arr)]


# In[97]:


def HotellingTest(X):
    statistic = 0
    try:
        n, p = X.shape
        delta = np.mean(X, axis=0).reshape(1, -1)
        Sx = sample_covariance(X)
        S_pooled = Sx/n
        t_squared = delta@np.linalg.inv(S_pooled)@delta.T
        t1 = delta@np.linalg.inv(Sx)@delta.T
        t2 = t1/np.linalg.det(np.linalg.inv(Sx))
        t3 = np.linalg.det(Sx)
        t4 = delta@delta.T
        statistic = t_squared[0,0]*(n-p)/(p*(n-1))
    except Exception as e:
        print(str(e))

    return statistic


# In[98]:


def sample_covariance(X):
    temp = np.zeros((X.shape[1], X.shape[1]))
    n = X.shape[0]
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            temp[i, j] = (1/(n-1))*(X[:, i] - np.mean(X[:, i]))@(X[:, j] - np.mean(X[:, j]))
    return temp


# In[99]:


def var_explained(X, X_approx):
    # X and X_approx need to be numpy array
    return 100 - 100*(np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2)


# In[108]:



def t_data_creation(X, save_path, var_threshold):
    
    columns = ['Frame_No', 'Avg_t', 'Max_t', 'Avg_t2', 'Max_t2', 't2_with_PCA']

    cols_x = [s for s in X.columns if 'delta_x' in s]
    cols_y = [s for s in X.columns if 'delta_y' in s]
    cols_xy = cols_x+cols_y

    final_data = []

    X_temp = X[cols_xy].copy()

    pca_X_temp = X_temp.to_numpy()

    for n_comp in range(1, pca_X_temp.shape[1]):
        pca = PCA(n_components=n_comp)
        pca.fit(pca_X_temp)

        X_temp_transform = pca_X_temp@pca.components_.T
        ve = var_explained(pca_X_temp, X_temp_transform@pca.components_)

        if ve>=var_threshold:
            print(n_comp, ve)
            break

    for frame_no in sorted(X['Frame_No'].unique()):
        print("Frame no: ", frame_no)
        bivar_HT2 = []
        for col_x, col_y in zip(cols_x, cols_y):
            bivar_HT2.append(HotellingTest(X[X['Frame_No']==frame_no][[col_x]+[col_y]].to_numpy()))
        avg_bivar_HT2 = np.mean(bivar_HT2)
        max_bivar_HT2 = np.amax(bivar_HT2)

        X_temp_fno = X_temp[X['Frame_No']==frame_no].to_numpy()
        temp = avg_max_t(X_temp_fno)
        avg_t, max_t = temp[0], temp[1]

        t2_with_PCA = HotellingTest(X_temp_transform[X['Frame_No']==frame_no])

        res = [frame_no, avg_t, max_t, avg_bivar_HT2, max_bivar_HT2, t2_with_PCA]

        final_data.append(res)

    df = pd.DataFrame(final_data, columns=columns)
    df.to_csv(save_path)
    return df


# In[69]:


df_disfa = pd.read_csv("Data/DISFA_KPMs.csv", index_col=0)


# In[70]:


#Dropping the registered keypoints in data

drop_keypoints = [0, 16, 27, 33, 39, 42]

drop_columns = []
for kp in drop_keypoints:
    drop_columns.append('delta_x'+str(kp)+'_Normalized')
for kp in drop_keypoints:
    drop_columns.append('delta_y'+str(kp)+'_Normalized')

df_disfa.drop(columns=drop_columns, inplace=True)


# In[110]:


df_disfa['Subject_No'].unique()


# In[109]:


df_t_data = t_data_creation(df_disfa.copy(), "Data/all_keypoint_metrics.csv", var_threshold=90)


# In[ ]:





# In[ ]:




