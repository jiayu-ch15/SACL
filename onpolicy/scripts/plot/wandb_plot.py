
import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

plt.style.use('ggplot')

save_dir = './football/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

scenario_names = ['3v1']
title_names = ['3v1']
exp_names = ['test', 'test1'] 
label_names = ["5 epochs", "10 epochs"]
color_names = ['red','blue','limegreen']

for scenario_name, title_name in zip(scenario_names, title_names):
    plt.figure()
    ###################################PPO###################################

    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        data_dir =  './football/' + exp_name + '/goal.csv'

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
        max_steps.append(max_step)

        df_final = df_final.loc[df_final['Step'] <= max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        median_seed = np.median(y_seed, axis=1)
        std_seed = np.std(y_seed, axis=1)
        plt.plot(x_step, median_seed, label = label_name, color=color_name)
        plt.fill_between(x_step,
            median_seed - std_seed,
            median_seed + std_seed,
            color=color_name,
            alpha=0.1)

    plt.tick_params(axis='both',which='major') 
    final_max_step = np.min(max_steps)
    print("final max step is {}".format(final_max_step))
    x_major_locator = MultipleLocator(int(final_max_step/5))
    x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
    y_major_locator = MultipleLocator(0.2)
    y_minor_Locator = MultipleLocator(0.1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    tx = ax.xaxis.get_offset_text() 
    tx.set_fontsize(18) 
    #ax.xaxis.grid(True, which='minor')
    plt.xlim(0, final_max_step)
    plt.ylim([0, 1.1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Timesteps', fontsize=20)
    plt.ylabel('Win Rate', fontsize=20)
    plt.title(title_name, fontsize=20)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

    plt.savefig(save_dir + scenario_name + ".png", bbox_inches="tight")