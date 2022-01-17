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
                'academy_counterattack_easy']

title_names = [name.replace("_"," ") for name in scenario_names]

label_names=['3v1',\
            'CA(easy)']

label_60_names=['3v1-60',\
            'CA(easy)-60']


rollout_threads = ['5','10','25','50','100']

method_names = ['final_mappo_rollout' + rt for rt in rollout_threads] + ['final_mappo_rollout100_length1000']

algo_names = ['0.1x','0.2x','0.5x','1x','2x','10x']

metrics = {'final_mappo_rollout'+rt:['expected_goal'] for rt in rollout_threads}

metrics.update(
    {
    'final_mappo_rollout100_length1000': ['expected_goal'],
    'final_mappo': ['expected_goal'],
    'final_qmix': ['expected_win_rate'],
    'final_cds_qmix': ['test_score_reward_mean'],
    'final_cds_qplex': ['test_score_reward_mean'],
    'final_cds_qmix_denserew': ['test_score_reward_mean'],
    'final_cds_qplex_denserew': ['test_score_reward_mean'],
    'final_tikick': ['eval_win_rate']
    }
)
project_dir = './football/'
step_names = {
    'academy_3_vs_1_with_keeper': [13.3, 7.3, 3.0, 4.6, 6.6, 12.9],
    'academy_counterattack_easy': [0.0, 16.0, 5.4, 5.4, 8.5, 12.5],
}
bar_width = 1.5

value_dict = defaultdict(list)

for scenario_name, label_name, label_60_name, title_name in zip(scenario_names,label_names, label_60_names, title_names):
    
    plt.figure()

    value_dict['scenario_name'].append(label_name)
    value_dict['scenario_name'].append(label_60_name)

    metric_mean_list = []

    for method_name, algo_name, step_name in zip(method_names, algo_names, step_names[scenario_name]):

        metric_names = metrics[method_name]

        for metric_name in metric_names:
            data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)

            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            metric = all_metric[-10:,:] * 100
            metric_mean = np.mean(np.mean(metric, axis=1))
            metric_std = np.std(np.mean(metric, axis=1))

            metric_mean_list.append(metric_mean)

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
    if scenario_name == "academy_3_vs_1_with_keeper":
        ax1.set_ylim([40, 100]) 
    else:
        ax1.set_ylim([30, 110])
    tx = ax1.yaxis.get_offset_text() 
    tx.set_fontsize(18)
    ax1.tick_params(labelsize=15)
    
    X = (np.arange(len(metric_mean_list)) + 1) * 4
    X_final = X
    print(metric_mean_list)
    ax1.bar(X_final, metric_mean_list, alpha=0.8, width=bar_width, label='eval win rate', lw=1)
    for x, y in zip(X_final, metric_mean_list):
        plt.text(x, y+0.05, ('%.1f' % y) if y != 0.0 else 'NaN', ha='center', va= 'bottom', fontsize=15)

    ax1.legend(loc='upper left', fontsize=15) 

    ax2 = ax1.twinx()
    y_major_locator = MultipleLocator(6)
    y_minor_Locator = MultipleLocator(3)
    ax2.yaxis.set_major_locator(y_major_locator)
    ax2.yaxis.set_minor_locator(y_minor_Locator)
    ax2.set_ylabel('Timesteps (M)', fontsize=15)
    if scenario_name == "academy_3_vs_1_with_keeper":
        ax2.set_ylim([0, 18]) 
    else:
        ax2.set_ylim([2, 20])
    # ax2.set_ylim([0, 18])
    ax2.grid(False)
    tx = ax2.yaxis.get_offset_text() 
    tx.set_fontsize(18)
    ax2.tick_params(labelsize=15)
    

    X_final = X + bar_width
    print(step_names[scenario_name])
    ax2.bar(X_final, step_names[scenario_name], alpha=0.8, width=bar_width, label='timesteps-60%', lw=1, color='royalblue')
    for x, y in zip(X_final, step_names[scenario_name]):
        if scenario_name == "academy_counterattack_easy":
            if y == 0.0:
                y_txt = 2 + 0.05
            else:
                y_txt = y+0.05
        else:
            y_txt = y+0.05
        plt.text(x, y_txt, ('%.1f' % y) if y != 0.0 else 'NaN', ha='center', va= 'bottom', fontsize=15)
    ax2.legend(loc='upper right', fontsize=15) 
    
    
    # plt.yticks(fontsize=15)
    plt.title(title_name, fontsize=18)
    # plt.legend(loc='best', fontsize=15, bbox_transform=ax1.transAxes)#,bbox_to_anchor=(0.4, 0.5), numpoints=1, fancybox=True,  handlelength=0.8)

    plt.savefig(project_dir + "/" + scenario_name + "-batchsize-bar.png", bbox_inches="tight")

df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption="football", label='tab:football-batchsize'))


