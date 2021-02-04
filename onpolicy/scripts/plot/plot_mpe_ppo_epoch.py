
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

map_names = ['spread','speaker_listener','reference']
title_names = ['Spread','Comm','Reference']

for map_name,title_name in zip(map_names,title_names):
    plt.figure()
    ###################################PPO###################################
    exp_names = ['ppo5', 'ppo10', 'ppo15'] 
    label_names = ["5 epochs", "10 epochs", "15 epochs"]
    color_names = ['red','blue','limegreen']

    save_dir = './ppo_epoch/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        data_dir =  './ppo_epoch/' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']

        all_step = np.array(df[key_step])
        all_win_rate = np.array(df[key_win_rate])

        print("original shape is ")
        print(all_step.shape)
        print(all_win_rate.shape)

        df_final = df[key_cols].dropna()
        step = df_final[key_step]
        win_rate = df_final[key_win_rate]

        print("drop nan shape is")
        print(np.array(step).shape)
        print(np.array(win_rate).shape)

        max_step = step.max()['Step']
        print("max step is {}".format(max_step))

        if max_step < 2.5e6:
            max_step = 2e6
        elif max_step < 4e6:
            max_step = 3e6
        else:
            max_step = 20e6

        print("final step is {}".format(max_step))
        max_steps.append(max_step)

        df_final = df_final.loc[df_final['Step'] <= max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        mean_seed = np.mean(y_seed, axis=1)
        std_seed = np.std(y_seed, axis=1)
        plt.plot(x_step, mean_seed, label = label_name, color=color_name)
        plt.fill_between(x_step,
            mean_seed - std_seed,
            mean_seed + std_seed,
            color=color_name,
            alpha=0.1)

    plt.tick_params(axis='both',which='major') 
    final_max_step = np.min(max_steps)
    print("final max step is {}".format(final_max_step))
    if map_name == "spread": 
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
        y_major_locator = MultipleLocator(10)
        y_minor_Locator = MultipleLocator(2)
        plt.ylim(-150, -105)
    elif map_name == "speaker_listener":
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
        y_major_locator = MultipleLocator(10)
        y_minor_Locator = MultipleLocator(2)
        plt.ylim(-40, -5)
    else:
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/4)) 
        y_major_locator = MultipleLocator(5)
        y_minor_Locator = MultipleLocator(1)
        plt.ylim(-30, -5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    #ax.xaxis.grid(True, which='minor')
    plt.xlim(0, final_max_step)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=15)
    plt.xlabel('Timesteps', fontsize=25)
    plt.ylabel('Episode Rewards', fontsize=20)
    plt.title(title_name, fontsize=25)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=25)

    plt.savefig(save_dir + map_name + "_ppo_epoch.png", bbox_inches="tight")