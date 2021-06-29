
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

map_names = ['2s_vs_1sc','3m']
title_names = [name.replace("_vs_"," vs. ") for name in map_names]

save_dir = './SMAC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for map_name, title_name in zip(map_names, title_names):
    print("########################MAP##########################")
    print(map_name)
    

    exp_names = ['z_loc_loss','z_loss']
    y_label_names = ['Z Loc Loss', 'Z Loss'] 

    for exp_name, y_label_name in zip(exp_names, y_label_names):
        ###################################PPO###################################
        plt.figure()
        label_names = ["VMAPD"]
        color_names = ['red']

        max_steps = []
        for label_name, color_name in zip(label_names, color_names):
            data_dir =  './SMAC/' + map_name + '/' + exp_name + '/' + map_name + '_' + exp_name + '.csv'

            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            all_step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            df_final = df[key_cols].dropna()
            step = df_final[key_step]
            metric = df_final[key_metric]

            max_step = step.max()['Step']

            max_step = 2e6

            max_steps.append(max_step)

            df_final = df_final.loc[df_final['Step'] <= max_step] 

            x_step = np.array(df_final[key_step]).squeeze(-1)
            y_seed = np.array(df_final[key_metric])

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
        if map_name == "2s_vs_1sc":
            plt.ylim([0.4, 1.6])
        else:
            plt.ylim([0.8, 1.7])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Timesteps', fontsize=20)
        plt.ylabel(y_label_name, fontsize=20)
        plt.legend(loc='lower right', numpoints=1, fancybox=True, fontsize=20)
        plt.title(title_name,fontsize=20)
        plt.savefig(save_dir + map_name  + "_" + y_label_name + ".png", bbox_inches="tight")