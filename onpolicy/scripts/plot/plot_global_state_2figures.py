
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

map_names = ['MMM2','6h_vs_8z','corridor','10m_vs_11m','3s5z_vs_3s6z','8m_vs_9m','5m_vs_6m']
title_names = [name.replace("_vs_"," vs. ") for name in map_names]
#########figure1

for map_name, title_name in zip(map_names,title_names):
    plt.figure()
    ###################################PPO###################################
    #exp_names = ['final_mappo', 'mappo_nomustalive', 'final_mappo_original', 'mappo_original_mustalive'] 
    exp_names = ['final_mappo', 'mappo_nomustalive']
    label_names = ["with death mask", "without death mask"]
    color_names = ['red','blue','limegreen','saddlebrown']

    save_dir = './global_state/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        if exp_name in ['mappo_nomustalive', 'mappo_original_mustalive', 'mappo_catobs']:
            data_dir =  './global_state/' + map_name + '/' + map_name + '_' + exp_name + '.csv'
        else:
            data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'
        
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

        if "ppo" in exp_name and max_step < 4.96e6:
            print("error: broken data! double check!")

        if max_step < 4e6:
            max_step = 2e6
        elif max_step < 9e6:
            max_step = 5e6
        else:
            max_step = 10e6

        print("final step is {}".format(max_step))
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

    plt.savefig(save_dir + map_name + "_global_state_mask.png", bbox_inches="tight")

########## figure 2
map_names = ['MMM2','6h_vs_8z','corridor','10m_vs_11m','3s5z_vs_3s6z','5m_vs_6m']
for map_name in map_names:
    plt.figure()
    ###################################PPO###################################
    exp_names = ['final_mappo', 'mappo_original_mustalive', 'mappo_catobs'] 
    label_names = ["agent-specific", "original", "concated"]
    color_names = ['red','blue','limegreen']

    save_dir = './global_state/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        if exp_name in ['mappo_nomustalive', 'mappo_original_mustalive', 'mappo_catobs']:
            data_dir =  './global_state/' + map_name + '/' + map_name + '_' + exp_name + '.csv'
        else:
            data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'
        
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

        if "ppo" in exp_name and max_step < 4.96e6:
            print("error: broken data! double check!")

        if max_step < 4e6:
            max_step = 2e6
        elif max_step < 9e6:
            max_step = 5e6
        else:
            max_step = 10e6

        print("final step is {}".format(max_step))
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

    plt.savefig(save_dir + map_name + "_global_state.png", bbox_inches="tight")