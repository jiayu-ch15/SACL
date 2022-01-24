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

map_names=['MMM2']#, '10m_vs_11m','5m_vs_6m','6h_vs_8z','3s5z_vs_3s6z']
map_50_names=[ m + "-50%" for m in map_names]
title_names = [name.replace("_vs_"," vs. ") for name in map_names]

rollout_threads = ['1','2','4','8','16']

method_names = ['final_mappo_rollout' + rt for rt in rollout_threads] + ['final_mappo_rollout16_length2000']
algo_names = ['0.125x','0.25x','0.5x','1x','2x','10x']

metric_names = ['eval_win_rate']

step_names = {
    'MMM2':         [9.6, 0.0, 3.1, 2.1, 2.6, 4.5],
    '3s5z_vs_3s6z': [ 0.0, 4.0, 9.9, 3.0, 4.3, 0.0],
    '5m_vs_6m': [ 0.0, 4.1, 3.9, 3.9, 7.8, 9.9],
    '6h_vs_8z': [ 4.3, 2.6, 3.5, 3.3, 4.6, 0.0],
    '10m_vs_11m': [ 1.7, 1.5, 0.8, 0.7, 1.0, 3.4],
}

step_std_names = {
    'MMM2':         [9.6, 0.0, 3.1, 2.1, 2.6, 4.5],
    '3s5z_vs_3s6z': [ 0.0, 4.0, 9.9, 3.0, 4.3, 0.0],
    '5m_vs_6m': [ 0.0, 4.1, 3.9, 3.9, 7.8, 9.9],
    '6h_vs_8z': [ 4.3, 2.6, 3.5, 3.3, 4.6, 0.0],
    '10m_vs_11m': [ 1.7, 1.5, 0.8, 0.7, 1.0, 3.4],

}

project_dir = './smac/'
max_step = 10e6
bar_width = 1.5
error_params=dict(elinewidth=1, capsize=3, alpha=0.7)

for map_name, map_50_name, title_name in zip(map_names, map_50_names, title_names):
    

    step_names[map_name] = []
    step_std_names[map_name] = []

    for method_name, algo_name in zip(method_names, algo_names):
        print(method_name)
        for metric_name in metric_names:
            data_dir = project_dir + map_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)

            df = df.loc[df['Step'] <= max_step]
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            step = np.array(df[key_step])
            metric = np.array(df[key_metric])

            print(metric)
            need_step = []
            for m in range(metric.shape[1]):
                if map_name == "10m_vs_11m":
                    get_step = np.where(metric[:, m] > 0.9)[0]
                else:
                    get_step = np.where(metric[:, m] > 0.8)[0]
                print(get_step)
                if get_step.shape[0] != 0:
                    need_step.append(step[get_step[0]][0]/1e6)
                else:
                    need_step.append(np.NaN)
            
            need_step = np.array(need_step)

            metric_mean = np.mean(need_step)
            metric_std = np.std(need_step)

            step_names[map_name].append(metric_mean)
            step_std_names[map_name].append(metric_std)

value_dict = defaultdict(list)
for map_name, map_50_name, title_name in zip(map_names, map_50_names, title_names):
    plt.figure()
    metric_mean_list=[]
    metric_std_list = []

    value_dict['Maps'].append(map_name)
    value_dict['Maps'].append(map_50_name)
    for method_name, algo_name, step_name in zip(method_names, algo_names, step_names[map_name]):

        for metric_name in metric_names:
            data_dir = project_dir + map_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)

            df = df.loc[df['Step'] <= max_step]
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            print(all_metric.shape)

            if map_name in ["6h_vs_8z",'3s5z_vs_3s6z'] and method_name == "final_mappo_rollout16_length2000":
                metric = all_metric[-1:,:] * 100
            else:
                metric = all_metric[-10:,:] * 100

            metric_mean = np.mean(np.median(metric, axis=1))
            metric_std = np.std(np.median(metric, axis=1))

            metric_mean_list.append(metric_mean)
            metric_std_list.append(metric_std)

            result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
            value_dict[algo_name].append(result)
            value_dict[algo_name].append(step_name)
    
    plt.tick_params(axis='both',which='major') 
    plt.xticks((np.arange(len(method_names)) + 1) * 4 + bar_width/2, algo_names)#rotation控制倾斜角度
    plt.xticks(fontsize=15)
    ax1=plt.gca()

    y_major_locator = MultipleLocator(20)
    y_minor_Locator = MultipleLocator(10)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.set_minor_locator(y_minor_Locator)
    ax1.set_ylabel('Eval Win Rate (%)', fontsize=15)
    ax1.set_xlabel('Batch Size Scale', fontsize=15)
    if map_name == "MMM2":
        ax1.set_ylim([0, 115])
    elif map_name == "5m_vs_6m":
        ax1.set_ylim([20, 120])
    elif map_name == "3s5z_vs_3s6z":
        ax1.set_ylim([0, 120])
    elif map_name == "10m_vs_11m":
        ax1.set_ylim([40, 110])
    else:
        ax1.set_ylim([0, 105])
    tx = ax1.yaxis.get_offset_text() 
    tx.set_fontsize(18)
    ax1.tick_params(labelsize=15)
    
    X = (np.arange(len(metric_mean_list)) + 1) * 4
    X_final = X
    print(metric_mean_list)
    ax1.bar(X_final, metric_mean_list, alpha=0.8, width=bar_width, label='eval win rate', lw=1, yerr=metric_std_list, error_kw=error_params)
    for x, y in zip(X_final, metric_mean_list):
        plt.text(x, y+0.05, ('%.1f' % y) if y != 0.0 else 'NaN', ha='center', va= 'bottom', fontsize=15)

    ax1.legend(loc='upper left',fontsize=15) 

    ax2 = ax1.twinx()
    y_major_locator = MultipleLocator(2)
    y_minor_Locator = MultipleLocator(1)
    ax2.yaxis.set_major_locator(y_major_locator)
    ax2.yaxis.set_minor_locator(y_minor_Locator)
    tx = ax2.yaxis.get_offset_text() 
    tx.set_fontsize(18)
    ax2.set_ylabel('Timesteps (M)', fontsize=15)
    if map_name == "MMM2":
        ax2.set_ylim([0, 11.5])
    elif map_name == "5m_vs_6m":
        ax2.set_ylim([2, 12])
    elif map_name == "3s5z_vs_3s6z":
        ax2.set_ylim([0, 12])
    elif map_name == "10m_vs_11m":
        y_major_locator = MultipleLocator(3)
        y_minor_Locator = MultipleLocator(1.5)
        ax2.yaxis.set_major_locator(y_major_locator)
        ax2.yaxis.set_minor_locator(y_minor_Locator)
        ax2.set_ylim([0, 10.5])
    else:
        ax2.set_ylim([0, 10.5])
    ax2.grid(False)
    ax2.tick_params(labelsize=15)
    
    

    X_final = X + bar_width
    print(step_names[map_name])
    if map_name == "10m_vs_11m":
        ax2.bar(X_final, step_names[map_name], alpha=0.8, width=bar_width, label='timesteps-90%', lw=1, color='royalblue', yerr=step_std_names[map_name], error_kw=error_params)
    else:
        ax2.bar(X_final, step_names[map_name], alpha=0.8, width=bar_width, label='timesteps-80%', lw=1, color='royalblue', yerr=step_std_names[map_name], error_kw=error_params)
    for x, y in zip(X_final, step_names[map_name]):
        if map_name == "5m_vs_6m":
            if y > 0:
                y_txt = y + 0.05
            else:
                y_txt = 2 + 0.05
        elif map_name == "MMM2":
            if y > 0:
                y_txt = y + 0.05
            else:
                y_txt = 0 + 0.05
        elif map_name == "10m_vs_11m":
            if y > 0:
                y_txt = y + 0.05
            else:
                y_txt = 0 + 0.05
        else:
            y_txt = y + 0.05
        plt.text(x, y_txt, ('%.1f' % y) if y > 0.0 else 'NaN', ha='center', va= 'bottom', fontsize=15)
    ax2.legend(loc='upper right', fontsize=15) 
    
    
    # plt.yticks(fontsize=15)
    plt.title(title_name, fontsize=18)
    # plt.legend(loc='best', fontsize=15, bbox_transform=ax1.transAxes)#,bbox_to_anchor=(0.4, 0.5), numpoints=1, fancybox=True,  handlelength=0.8)

    plt.savefig(project_dir + "/" + map_name + "-batchsize-bar.png", bbox_inches="tight")

    
df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption="smac-batchsize", label='tab:smac-batchsize'))