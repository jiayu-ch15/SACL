
import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
from icecream import ic

plt.style.use('ggplot')

map_names = ['16','21','22','36','43','48','61']
agent_names = ['2 Agents','2 Agents']
method_names = ['nearest','apf','utility', 'rrt', 'global_stack']
label_names = ['Nearest','APF','Utility', 'RRT', 'MAANS']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.12

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    metric_agent = []
    for agent_name in agent_names:
        metric_map = []
        for map_name in map_names:
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/250step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/auc/250step.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/3agents/auc/250step.csv"
            
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])

            metric_map.append(metric)

        # [map, episode, seed] -- metric_map
        metric_map = np.array(metric_map)

        # [episode, seed]
        metric_episode = np.mean(metric_map, axis=0)

        # [seed]
        metric_seed = np.mean(metric_episode, axis=0)

        metric_agent.append(metric_seed)
    
    metric_agent = np.array(metric_agent)
    mean_metric = np.mean(metric_agent, axis=1)
    std_metric = np.mean(metric_agent, axis=1)

    X = np.arange(len(agent_names)) + 1
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.6, width=bar_width, color=color_name, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%d' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(50)
y_minor_Locator = MultipleLocator(25)
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
tx.set_fontsize(18) 
#ax.xaxis.grid(True, which='minor')
# plt.xlim(0, final_max_step)
# plt.ylim([0, 1.1])
plt.xticks(np.arange(len(agent_names)) + 1 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('AUC at 250 steps', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "bar_middle_AUC.png", bbox_inches="tight")








map_names = ['20','49']
agent_names = ['2 Agents','2 Agents']
method_names = ['nearest','apf','utility', 'rrt', 'global_stack']
label_names = ['Nearest','APF','Utility', 'RRT', 'MAANS']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.12

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    metric_agent = []
    for agent_name in agent_names:
        metric_map = []
        for map_name in map_names:
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/250step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/auc/250step.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/3agents/auc/250step.csv"
            
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])

            metric_map.append(metric)

        # [map, episode, seed] -- metric_map
        metric_map = np.array(metric_map)

        # [episode, seed]
        metric_episode = np.mean(metric_map, axis=0)

        # [seed]
        metric_seed = np.mean(metric_episode, axis=0)

        metric_agent.append(metric_seed)
    
    metric_agent = np.array(metric_agent)
    mean_metric = np.mean(metric_agent, axis=1)
    std_metric = np.mean(metric_agent, axis=1)

    X = np.arange(len(agent_names)) + 1
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.6, width=bar_width, color=color_name, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%d' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(50)
y_minor_Locator = MultipleLocator(25)
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
tx.set_fontsize(18) 
#ax.xaxis.grid(True, which='minor')
# plt.xlim(0, final_max_step)
# plt.ylim([0, 1.1])
plt.xticks(np.arange(len(agent_names)) + 1 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('AUC at 250 steps', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "bar_large_AUC.png", bbox_inches="tight")