
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
project_dir = './hanabi/'

agent_nums = ['2agent','3agent','4agent','5agent']
seed_names = ['seed0','seed1','seed2']
title_names = [f'{n} Agents' for n in agent_nums]
label_name = 'MAPPO'

for agent_num, title_name in zip(agent_nums, title_names):
    plt.figure()
    ###################################PPO###################################
    all_step = []
    all_score = []
    all_shape = []
    for seed_name in seed_names:
        data_dir =  project_dir + agent_num + '/' + seed_name + '.csv'
        print(data_dir)
        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_score = [n for n in key_cols if n != 'Step']

        seed_step = np.array(df[key_step])
        seed_score = np.array(df[key_score])

        print("original shape is ")
        print(seed_step.shape)
        print(seed_score.shape)

        all_step.append(seed_step)
        all_score.append(seed_score)
        all_shape.append(seed_score.shape[0])

    all_shape = np.array(all_shape)
    min_all_shape = np.min(all_shape)

    step = np.array(all_step[0][:min_all_shape,0]) * 20000
    score = np.array([ss[:min_all_shape,0] for ss in all_score])

    print(step.shape)
    print(score.shape)

    mean_seed = np.mean(score, axis=0)
    std_seed = np.std(score, axis=0)
    plt.plot(step, mean_seed, label = label_name)
    plt.fill_between(step,
        mean_seed - std_seed,
        mean_seed + std_seed,
        alpha=0.1)

    plt.tick_params(axis='both',which='major') 
    final_max_step = 10e9
    print("final max step is {}".format(final_max_step))
    x_major_locator = MultipleLocator(int(final_max_step/5))
    x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
    y_major_locator = MultipleLocator(5)
    y_minor_Locator = MultipleLocator(2.5)
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
    plt.ylim([0.0, 25.0])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Timesteps', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.title(title_name, fontsize=20)
    plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

    plt.savefig(project_dir + agent_num + ".png", bbox_inches="tight")