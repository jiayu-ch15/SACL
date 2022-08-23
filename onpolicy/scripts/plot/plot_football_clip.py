# python cal_habitat.py > latex.txt
import pandas
import json
import numpy as np
import sys
import os
from icecream import ic
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

plt.style.use('ggplot')

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

scenario_names=['academy_3_vs_1_with_keeper',\
                'academy_counterattack_easy',\
                'academy_counterattack_hard',\
                'academy_corner',\
                'academy_pass_and_shoot_with_keeper',\
                'academy_run_pass_and_shoot_with_keeper']
title_names = [name.replace("_"," ") for name in scenario_names]

method_names = ['final_mappo','final_qmix','final_cds_qmix','final_tikick']
color_names = ['red','blue','limegreen', 'saddlebrown','purple','gray','orange']
label_names = ['MAPPO','QMix','CDS','TiKick']

metrics = {}
metrics.update(
    {
    'final_qmix': ['expected_win_rate'],
    'final_qmix_sparse': ['expected_win_rate'],
    'final_cds_qmix': ['test_score_reward_mean'],
    'final_cds_qplex': ['test_score_reward_mean'],
    'final_cds_qmix_denserew': ['test_score_reward_mean'],
    'final_cds_qplex_denserew': ['test_score_reward_mean'],
    'final_tikick': ['eval_win_rate'],
    'final_mappo': ['expected_goal'],
    'final_mappo_sparse': ['expected_goal'],
    'final_mappo_denserew': ['expected_goal'],
    'final_mappo_separated_sparserew': ['expected_goal'],
    'final_mappo_separated_sharedenserew': ['expected_goal'],
    'final_mappo_separated_denserew': ['expected_goal'],
    }
)
project_dir = './football/'


for scenario_name, title_name in zip(scenario_names, title_names):
    plt.figure()

    if scenario_name in ['academy_3_vs_1_with_keeper',\
                'academy_counterattack_easy',\
                'academy_pass_and_shoot_with_keeper',\
                'academy_run_pass_and_shoot_with_keeper']:
        max_step=25e6
    else:
        max_step=50e6

    for method_name, label_name, color_name in zip(method_names, label_names, color_names):

        if method_name == "final_tikick" and scenario_name not in ['academy_3_vs_1_with_keeper',\
                'academy_counterattack_hard',\
                'academy_corner',\
                'academy_run_pass_and_shoot_with_keeper']:
            continue
        
        if "mappo" in method_name and scenario_name == "academy_corner":
            metric_names = ['eval_expected_win_rate']
        else:
            metric_names = metrics[method_name]

        for metric_name in metric_names:
            data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)
            df = df.loc[df['Step'] <= max_step]
            
            if method_name == "final_qmix":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 44e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 36e6]

            if method_name == "final_qmix_sparse":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 31e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 25e6]

            if method_name == "final_cds_qmix_denserew":
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 20e6]

            if method_name == "final_cds_qplex_denserew":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 25e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 20e6]

            if method_name == "final_cds_qmix":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 30e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 17e6]
            
            if method_name == "final_cds_qplex":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 24e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 19e6]
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            if method_name == "final_qmix":
                x_step = np.array(df[key_step])[::10].squeeze(-1)
                y_seed = np.array(df[key_metric])[::10]
            elif method_name in ['final_cds_qmix_denserew','final_cds_qplex_denserew','final_cds_qmix','final_cds_qplex']:
                x_step = np.array(df[key_step])[::8].squeeze(-1)
                y_seed = np.array(df[key_metric])[::8]
            else:
                x_step = np.array(df[key_step]).squeeze(-1)
                y_seed = np.array(df[key_metric])

            mean_seed = np.mean(y_seed, axis=1)
            print(mean_seed.shape)
            std_seed = np.std(y_seed, axis=1)
            plt.plot(x_step, mean_seed, label = label_name, color=color_name)
            plt.fill_between(x_step,
                mean_seed - std_seed,
                mean_seed + std_seed,
                color=color_name,
                alpha=0.1)

    if max_step == 25e6:
        x_major_locator = MultipleLocator(int(max_step/5))
        x_minor_Locator = MultipleLocator(int(max_step/2.5)) 
    else:
        x_major_locator = MultipleLocator(int(max_step/5))
        x_minor_Locator = MultipleLocator(int(max_step/2.5)) 

    y_major_locator = MultipleLocator(0.2)
    y_minor_Locator = MultipleLocator(0.1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    tx = ax.xaxis.get_offset_text() 
    tx.set_fontsize(15) 
    #ax.xaxis.grid(True, which='minor')
    plt.ylim(0, 1.0)
    plt.xlim(0, max_step)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Timesteps', fontsize=15)
    plt.ylabel('Eval Win Rate', fontsize=15)
    plt.title(title_name, fontsize=15)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=15)

    plt.savefig(project_dir + scenario_name + ".png", bbox_inches="tight")



method_names = ['final_mappo','final_mappo_denserew','final_mappo_sparse','final_qmix','final_qmix_sparse']
label_names = ['MAPPO(s-d)','MAPPO(d)','MAPPO(s)','QMix(s-d)','QMix(s)']



for scenario_name, title_name in zip(scenario_names, title_names):
    plt.figure()

    if scenario_name in ['academy_3_vs_1_with_keeper',\
                'academy_counterattack_easy',\
                'academy_pass_and_shoot_with_keeper',\
                'academy_run_pass_and_shoot_with_keeper']:
        max_step=25e6
    else:
        max_step=50e6

    for method_name, label_name, color_name in zip(method_names, label_names, color_names):

        if method_name == "final_tikick" and scenario_name not in ['academy_3_vs_1_with_keeper',\
                'academy_counterattack_hard',\
                'academy_corner',\
                'academy_run_pass_and_shoot_with_keeper']:
            continue
        
        if "mappo" in method_name and scenario_name == "academy_corner":
            metric_names = ['eval_expected_win_rate']
        else:
            metric_names = metrics[method_name]

        for metric_name in metric_names:
            data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)
            df = df.loc[df['Step'] <= max_step]
            
            if method_name == "final_qmix":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 44e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 36e6]

            if method_name == "final_qmix_sparse":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 31e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 25e6]

            if method_name == "final_cds_qmix_denserew":
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 20e6]

            if method_name == "final_cds_qplex_denserew":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 25e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 20e6]

            if method_name == "final_cds_qmix":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 30e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 17e6]
            
            if method_name == "final_cds_qplex":
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 24e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 19e6]
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            if method_name in ['final_qmix','final_qmix_sparse']:
                x_step = np.array(df[key_step])[::10].squeeze(-1)
                y_seed = np.array(df[key_metric])[::10]
            elif method_name in ['final_cds_qmix_denserew','final_cds_qplex_denserew','final_cds_qmix','final_cds_qplex']:
                x_step = np.array(df[key_step])[::8].squeeze(-1)
                y_seed = np.array(df[key_metric])[::8]
            else:
                x_step = np.array(df[key_step]).squeeze(-1)
                y_seed = np.array(df[key_metric])

            mean_seed = np.mean(y_seed, axis=1)
            print(mean_seed.shape)
            std_seed = np.std(y_seed, axis=1)
            plt.plot(x_step, mean_seed, label = label_name, color=color_name)
            plt.fill_between(x_step,
                mean_seed - std_seed,
                mean_seed + std_seed,
                color=color_name,
                alpha=0.1)

    if max_step == 25e6:
        x_major_locator = MultipleLocator(int(max_step/5))
        x_minor_Locator = MultipleLocator(int(max_step/2.5)) 
    else:
        x_major_locator = MultipleLocator(int(max_step/5))
        x_minor_Locator = MultipleLocator(int(max_step/2.5)) 

    y_major_locator = MultipleLocator(0.2)
    y_minor_Locator = MultipleLocator(0.1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(x_minor_Locator)
    ax.yaxis.set_minor_locator(y_minor_Locator)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    tx = ax.xaxis.get_offset_text() 
    tx.set_fontsize(15) 
    #ax.xaxis.grid(True, which='minor')
    plt.ylim(0, 1.0)
    plt.xlim(0, max_step)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Timesteps', fontsize=15)
    plt.ylabel('Eval Win Rate', fontsize=15)
    plt.title(title_name, fontsize=15)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=15)

    plt.savefig(project_dir + scenario_name + "_sparse.png", bbox_inches="tight")
