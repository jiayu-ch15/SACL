
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

map_names = ['hanabi']

title_names = ['2 Player Hanabi']

for map_name,title_name in zip(map_names,title_names):
    plt.figure()
    ###################################PPO###################################
    exp_names = ['seed1']  
    label_names = ["seed 3"]
    color_names = ['red']

    save_dir = './hanabi_new/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        cut_max_step = 0
        cut_x_step = None
        cut_y_seed = None

        if exp_name in ['seed1']:

            if exp_name == 'seed1':
                data_dir =  './hanabi_new/365.csv'

            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            key_step = [n for n in key_cols if n == 'Step']
            key_win_rate = [n for n in key_cols if n != 'Step']

            df_final = df[key_cols].dropna()
            step = df_final[key_step]
            cut_max_step = step.max()['Step']
            
            cut_x_step = np.array(df_final[key_step]).squeeze(-1)
            cut_y_seed = np.array(df_final[key_win_rate])

        data_dir =  './hanabi_new/' + exp_name + '.csv'

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

        df_final = df_final.loc[df_final['Step'] <= max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        max_step += cut_max_step
        x_step += cut_max_step
        if cut_y_seed is not None:
            print(cut_x_step)
            print(x_step)
            x_step = np.concatenate((cut_x_step,x_step),axis=0)
            y_seed = np.concatenate((cut_y_seed,y_seed),axis=0)

        max_steps.append(max_step) 
        # mean_seed = np.mean(y_seed, axis=1)
        # std_seed = np.std(y_seed, axis=1)
        plt.plot(x_step, y_seed, label = label_name, color=color_name)
        # plt.fill_between(x_step,
        #     mean_seed - std_seed,
        #     mean_seed + std_seed,
        #     color=color_name,
        #     alpha=0.1)

    plt.tick_params(axis='both',which='major') 
    
    final_max_step = np.max(max_steps)
    final_max_step = 20e9
    x_major_locator = MultipleLocator(int(final_max_step/4))
    x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
    y_major_locator = MultipleLocator(1)
    y_minor_Locator = MultipleLocator(0.5)
    plt.xlim(0.7, 20)
    plt.ylim(20, 25)
    ax=plt.gca()
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    tx = ax.xaxis.get_offset_text() 
    tx.set_fontsize(18) 
    #ax.xaxis.grid(True, which='minor')
    plt.xlim(0, final_max_step)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)

    plt.xlabel('Timesteps', fontsize=25)
    plt.ylabel('Score', fontsize=20)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)
    plt.title(title_name,fontsize=25)
    plt.savefig(save_dir + map_name + "_score_long.png", bbox_inches="tight")