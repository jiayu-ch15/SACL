
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

map_names = ['16','20','21','22','36','43','48','49','61']
agent_names = ['2 Agents','3 Agents','3-2 Agents','4-2 Agents','4-3 Agents']
method_names = ['rrt'] #['global_stack','rrt','utility','nearest','apf']
label_names = ['RRT'] #['MAANS','RRT','Utility','Nearest','APF']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.45

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    print(method_name)
    metric_agent = []
    for agent_name in agent_names:
        print(agent_name)
        metric_map = []
        for map_name in map_names:
            print(map_name)
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/180step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/auc/180step.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat_3agents/' + map_name + '/' + method_name + "/auc/180step.csv"
            if agent_name == "3-2 Agent":
                data_dir =  './habitat_90stop_3_2agents/' + map_name + '/' + method_name + "/auc/180step.csv"
            if agent_name == "4-2 Agents":
                data_dir =  './habitat_90stop_4_2agents/' + map_name + '/' + method_name + "/auc/180step.csv"
            if agent_name == "4-3 Agents":
                data_dir =  './habitat_90stop_4_3agents/' + map_name + '/' + method_name + "/auc/180step.csv"
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])
            metric = np.mean(metric, axis=1)
            print(metric.shape)

            metric_map.append(metric)

        # [map, episode] -- metric_map
        
        metric_map = np.array(metric_map)

        # [episode]
        metric_episode = np.mean(metric_map, axis=0)

        metric_agent.append(metric_episode)
    
    mean_metric = np.mean(metric_agent, axis=1)

    X = (np.arange(len(agent_names)) + 1) * 3
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.8, width=bar_width, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%d' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(10)
y_minor_Locator = MultipleLocator(5)
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
plt.ylim([75, 150])
plt.xticks((np.arange(len(agent_names)) + 1) * 3 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('ACS', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best',bbox_to_anchor=(0.4, 0.5), numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "bar_AUC.png", bbox_inches="tight")


map_names = ['16','20','21','22','36','43','48','49','61']
agent_names = ['2 Agents']
method_names = ['global_stack','rrt','utility','nearest','apf']
label_names = ['MAANS','RRT','Utility','Nearest','APF']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.45

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    print(method_name)
    metric_agent = []
    for agent_name in agent_names:
        print(agent_name)
        metric_map = []
        for map_name in map_names:
            print(map_name)
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/200step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/auc/200step.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat_3agents/' + map_name + '/' + method_name + "/auc/200step.csv"
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])
            metric = np.mean(metric, axis=1)
            print(metric.shape)

            metric_map.append(metric)

        # [map, episode] -- metric_map
        
        metric_map = np.array(metric_map)

        # [episode]
        metric_episode = np.mean(metric_map, axis=0)

        metric_agent.append(metric_episode)
    
    mean_metric = np.mean(metric_agent, axis=1)

    X = (np.arange(len(agent_names)) + 1) * 3
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.8, width=bar_width, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(10)
y_minor_Locator = MultipleLocator(5)
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
plt.ylim([80, 135])
plt.xticks((np.arange(len(agent_names)) + 1) * 3 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('ACS', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='upper right', numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "2agents_bar_AUC.png", bbox_inches="tight")

##########################unseen######################
map_names = ['40','26','27']
agent_names = ['2 Agents']
method_names = ['distill','rrt','utility','nearest','apf']
label_names = ['MAANS-TD','RRT','Utility','Nearest','APF']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.45

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    print(method_name)
    metric_agent = []
    for agent_name in agent_names:
        print(agent_name)
        metric_map = []
        for map_name in map_names:
            print(map_name)
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/200step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/auc/200step.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat_3agents/' + map_name + '/' + method_name + "/auc/200step.csv"
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])
            metric = np.mean(metric, axis=1)
            print(metric.shape)

            metric_map.append(metric)

        # [map, episode] -- metric_map
        
        metric_map = np.array(metric_map)

        # [episode]
        metric_episode = np.mean(metric_map, axis=0)

        metric_agent.append(metric_episode)
    
    mean_metric = np.mean(metric_agent, axis=1)

    X = (np.arange(len(agent_names)) + 1) * 3
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.8, width=bar_width, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(10)
y_minor_Locator = MultipleLocator(5)
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
plt.ylim([100, 145])
plt.xticks((np.arange(len(agent_names)) + 1) * 3 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('ACS', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='upper right', numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "2agents_unseen_bar_AUC.png", bbox_inches="tight")








map_names = ['16','20','21','22','36','43','48','49','61']
agent_names = ['3 Agents']
method_names = ['global_stack','rrt','utility','nearest','apf']
label_names = ['MAANS','RRT','Utility','Nearest','APF']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']


save_dir = './habitat_AUC/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.45

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    print(method_name)
    metric_agent = []
    for agent_name in agent_names:
        print(agent_name)
        metric_map = []
        for map_name in map_names:
            print(map_name)
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/200step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/auc/200step.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat_3agents/' + map_name + '/' + method_name + "/auc/200step.csv"
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])
            metric = np.mean(metric, axis=1)
            print(metric.shape)

            metric_map.append(metric)

        # [map, episode] -- metric_map
        
        metric_map = np.array(metric_map)

        # [episode]
        metric_episode = np.mean(metric_map, axis=0)

        metric_agent.append(metric_episode)
    
    mean_metric = np.mean(metric_agent, axis=1)

    X = (np.arange(len(agent_names)) + 1) * 3
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.8, width=bar_width, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(10)
y_minor_Locator = MultipleLocator(5)
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
plt.ylim([90, 150])
plt.xticks((np.arange(len(agent_names)) + 1) * 3 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('ACS', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='upper right', numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "3agents_bar_AUC.png", bbox_inches="tight")


map_names = ['16','20','21','22','36','43','48','49','61']
agent_names = ['2 Agents','3 Agents']

save_dir = './habitat_overlap/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure()

bar_width = 0.45

i = 0
for method_name, label_name, color_name in zip(method_names, label_names, color_names):
    print(method_name)
    metric_agent = []
    for agent_name in agent_names:
        print(agent_name)
        metric_map = []
        for map_name in map_names:
            print(map_name)
            # data_dir =  save_dir + map_name + '/' + method_name + "/auc.csv'
            if agent_name == "1 Agent":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/single_agent/auc/200step.csv"
            if agent_name == "2 Agents":
                data_dir =  './habitat/' + map_name + '/' + method_name + "/overlap/90ratio.csv"
            if agent_name == "3 Agents":
                data_dir =  './habitat_3agents/' + map_name + '/' + method_name + "/overlap/90ratio.csv"
            df = pandas.read_csv(data_dir)
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step']

            x_step = np.array(df[key_step]).squeeze(-1)
            metric = np.array(df[key_metric])
            metric = np.mean(metric, axis=1)
            print(metric.shape)

            metric_map.append(metric)

        # [map, episode] -- metric_map
        
        metric_map = np.array(metric_map)

        # [episode]
        metric_episode = np.mean(metric_map, axis=0)

        metric_agent.append(metric_episode)
    
    mean_metric = np.mean(metric_agent, axis=1)

    X = (np.arange(len(agent_names)) + 1) * 3
    X_final = X + i * bar_width
    plt.bar(X_final, mean_metric, alpha=0.8, width=bar_width, label=label_name, lw=1)
    for x,y in zip(X_final, mean_metric):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom',fontsize=15)

    i += 1
plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(10)
y_minor_Locator = MultipleLocator(5)
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
plt.ylim([0, 1.0])
plt.xticks((np.arange(len(agent_names)) + 1) * 3 + int(len(method_names)/2) * bar_width , agent_names)#rotation控制倾斜角度
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('Overlap Ratio', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best',bbox_to_anchor=(0.4, 0.5), numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "bar_overlap.png", bbox_inches="tight")