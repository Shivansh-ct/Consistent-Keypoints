#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# In[2]:


all_keypoint_metrics = pd.read_csv("Data/all_keypoint_metrics.csv").iloc[:4845, :]

ori_AU_consistency = np.loadtxt("Data/AU_consistency.txt") 


# In[3]:


Avg_t = np.array(all_keypoint_metrics['Avg_t'])
Max_t = np.array(all_keypoint_metrics['Max_t'])
Avg_t2 = np.array(all_keypoint_metrics['Avg_t2'])
Max_t2 = np.array(all_keypoint_metrics['Max_t2'])
t2_with_PCA = np.array(all_keypoint_metrics['t2_with_PCA'])


# ### Computing all the regression plots

# In[37]:


AU_consistency = 100*ori_AU_consistency/27
df_temp = pd.DataFrame(AU_consistency, columns=['AU_consistency'])
df_temp = pd.concat([df_temp, all_keypoint_metrics], axis=1)
  
metric_dict = {'Avg_t':"Average $\it{t}$-statistic", 'Max_t':"Maximum $\it{t}$-statistic", 'Avg_t2':"Average $\it{t}^{2}$-statistic", 'Max_t2':"Maximum $\it{t}^{2}$-statistic", 't2_with_PCA':"$t^{2}$-statistic with PCA"}

for metric in metric_dict.keys():
    
    t_arr = np.array(all_keypoint_metrics[metric])
    
    reg = LinearRegression().fit(t_arr.reshape(-1,1), AU_consistency.reshape(-1,1))
    r2_score = reg.score(t_arr.reshape(-1,1), AU_consistency.reshape(-1,1))

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # all the plt features will work if placed after lmplot function
    sb.lmplot(x = metric,
                y = "AU_consistency", 
                ci = None,
                data = df_temp, height=10, aspect=1, markers=".", line_kws={'color': 'red'})
    plt.ylim(0,110)
#     plt.xlim(0,5.5)
    plt.xlabel(metric_dict[metric], fontsize=28, fontweight='bold')
    plt.ylabel("AU Consistency", fontsize=28, fontweight='bold')
#     plt.rcParams["axes.edgecolor"] = "black"
#     plt.rcParams["figure.autolayout"] = True
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    
    # Set font and weight for xticks and yticks
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

#     plt.legend([f'$R^2$ Score: {round(r2_score, 2)}'], loc='upper right', fontsize=20, title='Legend Title')
#     plt.text(0.95, 2.97, r'($R^2$ Score: {0:.2f})'.format(round(r2_score, 2)), ha='center', va='center', transform=ax.transAxes, fontsize=20, fontweight='bold')
    plt.text(0.95, 2.97, f'($R^2$ Score: {round(r2_score, 2)})', ha='center', va='center', transform=ax.transAxes, fontsize=30, fontweight='bold')    
    
    # Increase padding between xlabel and xticks
#     fig.subplots_adjust(bottom=-0.4)
    # Increase padding between xlabel and xticks
#     ax.xaxis.labelpad = 80  # Adjust the value as needed

    # Increase padding between xlabel and xticks
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed


    plt.tight_layout()  # Add this line to adjust layout
    print("Plotting")
    plt.show()
#     plt.savefig("Results/Regression/"+metric+".svg", dpi=300)
#     plt.close()