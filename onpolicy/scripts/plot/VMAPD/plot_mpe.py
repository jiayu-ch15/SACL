
import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
from icecream import ic

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

plt.style.use('ggplot')

map_names = ['reference','spread','comm']
title_names = ['Reference','Spread','Comm']


exp_names = ['z_loc_loss', 'z_loss']
y_label_names = ['Z Loc Loss', 'Z Loss']
#########figure1

for map_name, title_name in zip(map_names, title_names):
    for exp_name, y_label_name in zip(exp_names, y_label_names):
        ic(map_name, exp_name, y_label_name)
        plt.figure()
        ###################################PPO###################################
        
        algo_names = ['diayn','masd','vmapd']
        label_names = ['DIAYN','MASD','VMAPD']
        color_names = ['blue','limegreen','red']

        save_dir = './MPE/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        max_steps = []
        for algo_name, label_name, color_name in zip(algo_names, label_names, color_names):
            ic(algo_name)
            
            data_dir =  './MPE/' + map_name + '/' + label_name + '/' + exp_name + '/' + \
            map_name + '_' + algo_name + '_' + exp_name + '.csv'
            
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            all_step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            print("original shape is ")
            print(all_step.shape)
            print(all_metric.shape)

            df_final = df[key_cols].dropna()
            step = df_final[key_step]
            metric = df_final[key_metric]

            print("drop nan shape is")
            print(np.array(step).shape)
            print(np.array(metric).shape)

            max_step = step.max()['Step']
            print("max step is {}".format(max_step))

            if max_step < 2.5e6:
                max_step = 2e6
            else:
                max_step = 5e6

            print("final step is {}".format(max_step))
            max_steps.append(max_step)

            df_final = df_final.loc[df_final['Step'] <= max_step] 

            x_step = np.array(df_final[key_step]).squeeze(-1)
            y_seed = np.array(df_final[key_metric])

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

        if map_name == "spread": 
            x_major_locator = MultipleLocator(int(final_max_step/4))
            x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
            if exp_name == "idv_rew":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                # plt.ylim(-3.0, 0.2)
            elif exp_name == "z_loc_loss":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.3, 1.7)
            else:
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.3, 1.7)
        elif map_name == "comm":
            x_major_locator = MultipleLocator(int(final_max_step/4))
            x_minor_Locator = MultipleLocator(int(final_max_step/8)) 

            if exp_name == "idv_rew":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                # plt.ylim(-3.0, 0.2)
            elif exp_name == "z_loc_loss":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(0.7, 1.7)
            else:
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(0.0, 2.0)
        else:
            x_major_locator = MultipleLocator(int(final_max_step/5))
            x_minor_Locator = MultipleLocator(int(final_max_step/10))

            if exp_name == "idv_rew":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                # plt.ylim(-3.0, 0.2)
            elif exp_name == "z_loc_loss":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.1, 1.7)
            else:
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.1, 1.7)
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
        plt.ylabel(y_label_name, fontsize=20)
        if exp_name == "idv_rew":
            plt.legend(loc='lower right', numpoints=1, fancybox=True, fontsize=25)
        elif exp_name == "z_loc_loss":
            plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=25)
        else:
            plt.legend(loc='upper right', numpoints=1, fancybox=True, fontsize=25)

        plt.title(title_name, fontsize=25)
        plt.savefig(save_dir + map_name + "_" + exp_name + ".png", bbox_inches="tight")


exp_names = ['idv_rew']
y_label_names = ['Final Distance']

for map_name, title_name in zip(map_names, title_names):
    for exp_name, y_label_name in zip(exp_names, y_label_names):
        ic(map_name, exp_name, y_label_name)
        plt.figure()
        ###################################PPO###################################
        
        algo_names = ['qmix','mappo','diayn','masd','vmapd']
        label_names = ['QMIX','MAPPO','DIAYN','MASD','VMAPD']
        color_names = ['blue','limegreen','saddlebrown','purple','red']

        save_dir = './MPE/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        max_steps = []
        for algo_name, label_name, color_name in zip(algo_names, label_names, color_names):
            ic(algo_name)
            
            data_dir =  './MPE/' + map_name + '/' + label_name + '/' + exp_name + '/' + \
            map_name + '_' + algo_name + '_' + exp_name + '.csv'
            
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            all_step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            print("original shape is ")
            print(all_step.shape)
            print(all_metric.shape)

            df_final = df[key_cols].dropna()
            step = df_final[key_step]
            metric = df_final[key_metric]

            print("drop nan shape is")
            print(np.array(step).shape)
            print(np.array(metric).shape)

            max_step = step.max()['Step']
            print("max step is {}".format(max_step))

            if max_step < 2.5e6:
                max_step = 2e6
            else:
                max_step = 5e6

            print("final step is {}".format(max_step))
            max_steps.append(max_step)

            df_final = df_final.loc[df_final['Step'] <= max_step] 

            if algo_name == "qmix":
                x_step = np.array(df_final[key_step]).squeeze(-1)[::2]
                y_seed = np.array(df_final[key_metric])[::2]
            else:
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

        if map_name == "spread": 
            x_major_locator = MultipleLocator(int(final_max_step/4))
            x_minor_Locator = MultipleLocator(int(final_max_step/8)) 
            if exp_name == "idv_rew":
                y_major_locator = MultipleLocator(1.0)
                y_minor_Locator = MultipleLocator(0.5)
                plt.ylim(-3.3, 0.2)
            elif exp_name == "z_loc_loss":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.3, 1.7)
            else:
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.3, 1.7)
        elif map_name == "comm":
            x_major_locator = MultipleLocator(int(final_max_step/4))
            x_minor_Locator = MultipleLocator(int(final_max_step/8)) 

            if exp_name == "idv_rew":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.25)
                plt.ylim(-2.8, 0.2)
            elif exp_name == "z_loc_loss":
                y_major_locator = MultipleLocator(0.1)
                y_minor_Locator = MultipleLocator(0.05)
                plt.ylim(0.7, 1.7)
            else:
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(0.0, 2.0)
        else:
            x_major_locator = MultipleLocator(int(final_max_step/5))
            x_minor_Locator = MultipleLocator(int(final_max_step/10))

            if exp_name == "idv_rew":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.25)
                plt.ylim(-2.0, 0.2)
            elif exp_name == "z_loc_loss":
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.1, 1.7)
            else:
                y_major_locator = MultipleLocator(0.5)
                y_minor_Locator = MultipleLocator(0.1)
                plt.ylim(-0.1, 1.7)
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
        plt.ylabel(y_label_name, fontsize=20)

        if exp_name == "idv_rew":
            plt.legend(loc='lower right', numpoints=1, fancybox=True, fontsize=25)
        elif exp_name == "z_loc_loss":
            plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=25)
        else:
            plt.legend(loc='upper right', numpoints=1, fancybox=True, fontsize=25)

        plt.title(title_name, fontsize=25)
        plt.savefig(save_dir + map_name + "_" + exp_name + ".png", bbox_inches="tight")
