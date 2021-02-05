
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
    exp_names = ['mappo', 'ippo'] 
    label_names = ["MAPPO", "IPPO"]
    color_names = ['red','limegreen']

    save_dir = './mpe/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        print(exp_name)
        data_dir =  './mpe/' + map_name + '/' + map_name + '_' + exp_name + '.csv'

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

        if max_step < 2.5e6:
            max_step = 2e6
        elif max_step < 4e6:
            max_step = 3e6
        else:
            max_step = 20e6

        max_steps.append(max_step)

        df_final = df_final.loc[df_final['Step'] <= max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)[::2]
        y_seed = np.array(df_final[key_win_rate])[::2]

        mean_seed = np.mean(y_seed, axis=1)
        std_seed = np.std(y_seed, axis=1)
        plt.plot(x_step, mean_seed, label = label_name, color=color_name)
        plt.fill_between(x_step,
            mean_seed - std_seed,
            mean_seed + std_seed,
            color=color_name,
            alpha=0.1)

    print("########################MADDPG##########################")
    exp_names = ['maddpg', 'qmix']
    label_names = ["MADDPG", "QMix"]
    color_names = ['blue','saddlebrown']

    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        data_dir =  './mpe/' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)

        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']

        one_run_max_step = []
        qmix_x_step = []
        qmix_y_seed = []
        for k in key_win_rate:

            df_final = df[[k, 'Step']].dropna()
            step = df_final[key_step]
            win_rate = df_final[k]

            max_step = step.max()['Step']

            if max_step < 2.5e6:
                max_step = 2e6
            elif max_step < 4e6:
                max_step = 3e6
            else:
                max_step = 20e6
            
            one_run_max_step.append(max_step)
            df_final = df_final.loc[df_final['Step'] <= max_step] 
            qmix_x_step.append(np.array(df_final[key_step]).squeeze(-1))
            qmix_y_seed.append(np.array(df_final[k]))
        # pick max qmix step
        qmix_max_step = np.min(one_run_max_step)
        max_steps.append(qmix_max_step)

        # adapt sample frequency
        sample_qmix_s_step = []
        sample_qmix_y_seed = []
        final_max_length = []
        for x, y in zip(qmix_x_step, qmix_y_seed):
            eval_interval = x[10] - x[9]
            print(eval_interval)
            if eval_interval == 1000:
                final_max_length.append(len(x[::32]))
                sample_qmix_s_step.append(x[::32])
                sample_qmix_y_seed.append(y[::32])
            elif eval_interval == 3200:
                final_max_length.append(len(x[::10]))
                sample_qmix_s_step.append(x[::10])
                sample_qmix_y_seed.append(y[::10])

        # truncate numpy
        max_common_length = np.min(final_max_length)
        print("max common qmix length is {}".format(max_common_length))
        final_qmix_x_step = []
        final_qmix_y_seed = []
        for x, y in zip(sample_qmix_s_step, sample_qmix_y_seed):
            final_qmix_x_step.append(x[:max_common_length])
            final_qmix_y_seed.append(y[:max_common_length])

        x_step = np.mean(final_qmix_x_step, axis=0)
        y_seed = np.array(final_qmix_y_seed)

        mean_seed = np.mean(y_seed, axis=0)
        std_seed = np.std(y_seed, axis=0)
        plt.plot(x_step, mean_seed, label=label_name, color=color_name)
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
        y_major_locator = MultipleLocator(20)
        y_minor_Locator = MultipleLocator(5)
        plt.ylim(-180, -100)
    elif map_name == "speaker_listener":
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
        y_major_locator = MultipleLocator(20)
        y_minor_Locator = MultipleLocator(5)
        plt.ylim(-100, -5)
    else:
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/4)) 
        y_major_locator = MultipleLocator(10)
        y_minor_Locator = MultipleLocator(2)
        plt.ylim(-40, -5)
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
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=15)

    plt.xlabel('Timesteps', fontsize=25)
    plt.ylabel('Episode Rewards', fontsize=20)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=25)
    plt.title(title_name,fontsize=25)
    plt.savefig(save_dir + map_name + "_episode_rewards.png", bbox_inches="tight")