
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

avg = 30
plt.style.use('ggplot')

map_names = ['16','21','22','36','43','48','61']

method_names = ['nearest','apf','utility', 'rrt', 'global_stack']
label_names = ['Nearest','APF','Utility', 'RRT', 'MAANS']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    metric_map = []
    for map_name in map_names:
        data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
        # data_dir =  save_dir + map_name + '/' + method_name + "/auc/100step.csv"

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        
        key_step = [n for n in key_cols if n == 'Step']
        key_metric = [n for n in key_cols if n != 'Step']

        x_step = np.array(df[key_step]).squeeze(-1)
        metric = np.array(df[key_metric])

        metric_map.append(metric)

    # [map, step, seed] -- metric_map
    metric_map = np.array(metric_map)
    # first map mean
    y_seed = np.mean(metric_map, axis=0)

    mean_seed = np.mean(y_seed, axis=1)
    std_seed = np.std(y_seed, axis=1)

    plt.plot(x_step, mean_seed, label = label_name, color=color_name)
    plt.fill_between(x_step,
        mean_seed - std_seed,
        mean_seed + std_seed,
        color=color_name,
        alpha=0.1)

plt.tick_params(axis='both',which='major') 
final_max_step = 300
x_major_locator = MultipleLocator(50)
x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(0.25)
y_minor_Locator = MultipleLocator(0.125)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
tx.set_fontsize(18) 
#ax.xaxis.grid(True, which='minor')
plt.xlim(0, final_max_step)
plt.ylim([0, 1.1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Coverage Ratio', fontsize=20)
plt.title('Middle Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

plt.savefig(save_dir + "middle_AUC.png", bbox_inches="tight")


map_names = ['2','49']

plt.figure()

for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    metric_map = []
    for map_name in map_names:
        data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
        # data_dir =  save_dir + map_name + '/' + method_name + "/auc/100step.csv"

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        
        key_step = [n for n in key_cols if n == 'Step']
        key_metric = [n for n in key_cols if n != 'Step']

        x_step = np.array(df[key_step]).squeeze(-1)
        metric = np.array(df[key_metric])

        metric_map.append(metric)

    # [map, step, seed] -- metric_map
    metric_map = np.array(metric_map)
    # first map mean
    y_seed = np.mean(metric_map, axis=0)

    mean_seed = np.mean(y_seed, axis=1)
    std_seed = np.std(y_seed, axis=1)

    plt.plot(x_step, mean_seed, label = label_name, color=color_name)
    plt.fill_between(x_step,
        mean_seed - std_seed,
        mean_seed + std_seed,
        color=color_name,
        alpha=0.1)

plt.tick_params(axis='both',which='major') 
final_max_step = 300
x_major_locator = MultipleLocator(50)
x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(0.25)
y_minor_Locator = MultipleLocator(0.125)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
tx.set_fontsize(18) 
#ax.xaxis.grid(True, which='minor')
plt.xlim(0, final_max_step)
plt.ylim([0, 1.1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Coverage Ratio', fontsize=20)
plt.title('Large Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

plt.savefig(save_dir + "Large_AUC.png", bbox_inches="tight")
