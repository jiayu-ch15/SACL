
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
    exp_names = ['mappo_walltime', 'maddpg_walltime','qmix_walltime'] 
    label_names = ["MAPPO", "MADDPG","QMix"]
    color_names = ['red','blue','saddlebrown']

    save_dir = './walltime/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        data_dir =  './walltime/' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)

        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c and 'step' not in c]
        key_step = [n for n in key_cols if n == 'Relative Time (Process)']
        key_win_rate = [n for n in key_cols if n != 'Relative Time (Process)']


        one_run_max_step = []
        qmix_x_step = []
        qmix_y_seed = []
        for k in key_win_rate:
            print("one run original shape is ")
            print(np.array(df[k]).shape)

            df_final = df[[k, 'Relative Time (Process)']].dropna()
            step = df_final[key_step]
            win_rate = df_final[k]
            
            print("one run drop nan shape is")
            print(np.array(step).shape)
            print(np.array(win_rate).shape)

            max_step = step.max()['Relative Time (Process)']/3600.0
            print("one run max step is {}".format(max_step))
            
            one_run_max_step.append(max_step)
            print("final step is {}".format(max_step))

            df_final = df_final.loc[df_final['Relative Time (Process)'] <= max_step*3600.0] 
            qmix_x_step.append(np.array(df_final[key_step]/3600.0).squeeze(-1))
            qmix_y_seed.append(np.array(df_final[k]))
            print("data shape is {}".format(np.array(df_final[k]).shape))

        # pick max qmix step
        qmix_max_step = np.min(one_run_max_step)
        max_steps.append(qmix_max_step)

        # adapt sample frequency
        sample_qmix_s_step = []
        sample_qmix_y_seed = []
        final_max_length = []
        for x, y in zip(qmix_x_step, qmix_y_seed):
            if 'ddpg' in exp_name:
                if 'speaker_listener' in map_name:
                    final_max_length.append(len(x[::5]))
                    sample_qmix_s_step.append(x[::5])
                    sample_qmix_y_seed.append(y[::5])
                else:
                    final_max_length.append(len(x[::16]))
                    sample_qmix_s_step.append(x[::16])
                    sample_qmix_y_seed.append(y[::16])
            elif 'qmix' in exp_name:
                if map_name in ['speaker_listener','reference']:
                    final_max_length.append(len(x[::5]))
                    sample_qmix_s_step.append(x[::5])
                    sample_qmix_y_seed.append(y[::5])
                else:
                    final_max_length.append(len(x[::16]))
                    sample_qmix_s_step.append(x[::16])
                    sample_qmix_y_seed.append(y[::16])
            else:
                final_max_length.append(len(x))
                sample_qmix_s_step.append(x)
                sample_qmix_y_seed.append(y)

        # truncate numpy
        max_common_length = np.min(final_max_length)
        print("max common qmix length is {}".format(max_common_length))
        final_qmix_x_step = []
        final_qmix_y_seed = []
        for x, y in zip(sample_qmix_s_step, sample_qmix_y_seed):
            time = x[:max_common_length]
            if "ddpg" in exp_name:
                if map_name == "reference":
                    ratio = 128/214
                if map_name == "speaker_listener":
                    ratio = 42/132
                if map_name == "spread":
                    ratio = 133/180

            if "qmix" in exp_name:
                if map_name == "reference":
                    ratio = 168/500
                if map_name == "speaker_listener":
                    ratio = 88/270
                if map_name == "spread":
                    ratio = 277/460

            if "ppo" in exp_name:
                if map_name == "reference":
                    ratio = 2290/4400
                if map_name == "speaker_listener":
                    ratio = 1220/2469
                if map_name == "spread":
                    ratio = 2810/4617
            
            time =  time * ratio 
            final_qmix_x_step.append(time)
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
    final_max_step = np.max(max_steps) * ratio
    print("final max step is {}".format(final_max_step))
    # x_major_locator = MultipleLocator(int(final_max_step/4))
    # x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
    if map_name == "spread": 
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
        y_major_locator = MultipleLocator(20)
        y_minor_Locator = MultipleLocator(5)
        plt.ylim(-180, -100)
    elif map_name == "speaker_listener":
        x_major_locator = MultipleLocator(int(final_max_step/4))
        x_minor_Locator = MultipleLocator(int(final_max_step/4)/2) 
        y_major_locator = MultipleLocator(20)
        y_minor_Locator = MultipleLocator(5)
        plt.ylim(-80, -5)
    else:
        x_major_locator = MultipleLocator(int(final_max_step/2)/2)
        x_minor_Locator = MultipleLocator(int(final_max_step/2)/4) 
        y_major_locator = MultipleLocator(10)
        y_minor_Locator = MultipleLocator(2)
        plt.ylim(-40, -5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    #ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    #ax.xaxis.grid(True, which='minor')
    plt.xlim(0, final_max_step)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=15)
    plt.xlabel('Wall Time (Hours)', fontsize=25)
    plt.ylabel('Episode Rewards', fontsize=20)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=25)
    plt.title(title_name,fontsize=25)
    plt.savefig(save_dir + map_name + "_walltime.png", bbox_inches="tight")