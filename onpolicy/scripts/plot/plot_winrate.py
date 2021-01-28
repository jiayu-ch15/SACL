
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

######################################################################

map_name = '2m_vs_1z'
exp_names = ['final_mappo', 'final_ippo', 'final_mappo_original', 'final_qmix'] 
label_names = ["MAPPO", "IPPO", "MAPPO_original", "QMIX"]
save_dir = './win_rate/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for exp_name, label_name in zip(exp_names, label_names):
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

    if max_step < 5e6:
        max_step = 2e6
    elif max_step < 10e6:
        max_step = 5e6
    else:
        max_step = 10e6

    print("final step is {}".format(max_step))
    

    df_final = df_final.loc[df_final['Step'] <= max_step] 

    x_step = np.array(df_final[key_step]).squeeze(-1)
    y_seed = np.array(df_final[key_win_rate])

    median_seed = np.median(y_seed, axis=1)
    std_seed = np.std(y_seed, axis=1)
    plt.plot(x_step, median_seed, label = exp_name)
    plt.fill_between(x_step,
        median_seed - std_seed,
        median_seed + std_seed,
        alpha=0.1)

plt.tick_params(axis='both',which='major') 

x_major_locator = MultipleLocator(int(max_step/5))
x_minor_Locator = MultipleLocator(int(max_step/10)) 
y_major_locator = MultipleLocator(0.2)
y_minor_Locator = MultipleLocator(0.1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
ax.xaxis.get_major_formatter().set_powerlimits((0,1))
#ax.xaxis.grid(True, which='minor')
plt.xlim(0, max_step)
plt.ylim([0, 1.1])
plt.xlabel('Timesteps')
plt.ylabel('Win Rate')
plt.legend(loc='best', numpoints=1, fancybox=True)

plt.savefig(save_dir + map_name + "_win_rate.png", bbox_inches="tight")