import glob
import os
import pandas as pd
import numpy as np

path = "../../datasets/macro/Spontaneous/DISFA/DISFA/ActionUnit_Labels/"
sub_list = glob.glob(path+"*/")
sub_list.sort()

for sub in sub_list:
    sub = sub.split("/")[-2]
    print(sub)
    au_list = glob.glob(path+sub+"/"+"*.txt", recursive=True)
    au_list.sort()
    # print(au_list)
    cols_list = ['Subject_No', 'Frame_No']
    flag = 0
    for txt in au_list:

        cols_list.append("AU"+txt.split("/")[-1][8:-4])
        # print(cols_list)
        # exit()
        if flag == 0:
            au = np.loadtxt(txt, delimiter=',')
            sub_arr = np.ones((au.shape[0], 1))*int(sub[-3:])
            arr = np.concatenate((sub_arr, au), axis=1)
            # arr = au
            flag = 1
        else:
            # print("else wala hello", "AU No is ","AU"+txt[8:-4] )
            au = np.loadtxt(txt, delimiter=',')
            arr = np.concatenate((arr, au[:, 1].reshape(au.shape[0],-1)), axis=1)
    df_temp = pd.DataFrame(arr, columns=cols_list)
    # df_temp.to_csv(path+sub+"/"+txt[18:23]+"_all_au.csv")

    if "001" in sub:
        df_final = df_temp.copy()
    else:
        df_final = pd.concat([df_final, df_temp], axis=0)
    # break

df_final.to_csv("Data/all_au_labelings.csv")

