
import pandas
import json
import numpy as np
import sys
import os

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

map_names = ['3s5z_vs_3s6z']

all_final_max_step = []
for map_name in map_names:
    print("########################MAP##########################")
    print(map_name)
    final_max_step = 25e6
    exp_names = ['final_mappo_long']

    median_value = []
    std_value = []
    for exp_name in exp_names:
        print(exp_name)
        data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']

        all_step = np.array(df[key_step])
        all_win_rate = np.array(df[key_win_rate])

        df_final = df[key_cols].dropna()
        step = df_final[key_step]
        win_rate = df_final[key_win_rate]

        max_step = step.max()['Step']

        if "ppo" in exp_name and max_step < 4.96e6:
            print("error: broken data! double check!")
            print("drop one run!")

        df_final = df_final.loc[df_final['Step'] <= final_max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        y_seed_last = np.array(y_seed)[-10:]

        median_seed = np.median(np.median(y_seed_last, axis=0))
        std_seed = np.std(np.median(y_seed_last, axis=0))
        median_value.append(str(format(median_seed*100, '.1f')) + "(" + str(format(std_seed*100, '.1f')) + ")")

        print(median_seed)
        print(std_seed)
