import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

plt.style.use('ggplot')

map_names = ['16','21','22','36','43','48','61','20','49']
method_names = ['global_stack','rrt','utility','nearest','apf']
label_names = ['MAANS','RRT','Utility','Nearest','APF']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']
metric_names = ['overlap']
ratio_names = ['30ratio','50ratio','70ratio','90ratio']
x_names = ['0.0', '0.3', '0.5', '0.7', '0.9']

save_dir = './habitat/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for metric_name in metric_names:
    if metric_name == "overlap":
        plt.figure()

        for method_name, label_name, color_name in zip(method_names, label_names, color_names):
            metric_ratio = []
            for ratio_name in ratio_names:
                metric_map = []
                for map_name in map_names:

                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    # [episode, seed]
                    metric = np.array(df[key_metric])

                    # [seed]
                    metric_seed = np.mean(metric, axis=0)

                    metric_map.append(metric_seed)

                # [map, seed] -> [seed]
                metric_seed_again = np.mean(metric_map, axis = 0)

                metric_ratio.append(metric_seed_again)
            
            # [ratio, seed]
            y_seed = np.array(metric_ratio)

            mean_seed = np.mean(y_seed, axis=1)
            std_seed = np.std(y_seed, axis=1)
            mean_seed = np.concatenate([np.zeros(1,), mean_seed])
            std_seed = np.concatenate([np.zeros(1,), std_seed])

            x_step = np.arange(len(x_names)) + 1

            plt.plot(x_step, mean_seed, label = label_name, linewidth=4)#, color=color_name)
            plt.fill_between(x_step,
                mean_seed - std_seed,
                mean_seed + std_seed,
                #color=color_name,
                alpha=0.1)


plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(0.15)
y_minor_Locator = MultipleLocator(0.075)
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
tx.set_fontsize(18) 
#ax.xaxis.grid(True, which='minor')
plt.xlim(1, 1.9)
plt.ylim([0.0, 0.7])
plt.xticks(np.arange(len(x_names)) + 1 , x_names)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Coverage Ratio', fontsize=20)
plt.ylabel('Overlap Ratio', fontsize=20)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=18, handlelength=0.9)

plt.savefig(save_dir + "overlap.png", bbox_inches="tight")


map_names = ['16','21','22','36','43','48','61','20','49']
method_names = ['global_stack','rrt','utility','nearest','apf']
label_names = ['MAANS','RRT','Utility','Nearest','APF']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']
metric_names = ['overlap']
ratio_names = ['30ratio','50ratio','70ratio','90ratio']
x_names = ['0.0', '0.3', '0.5', '0.7', '0.9']

save_dir = './habitat_3agents/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for metric_name in metric_names:
    if metric_name == "overlap":
        plt.figure()

        for method_name, label_name, color_name in zip(method_names, label_names, color_names):
            print(method_name)
            metric_ratio = []
            for ratio_name in ratio_names:
                print(ratio_name)
                metric_map = []
                for map_name in map_names:
                    print(map_name)
                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    # [episode, seed]
                    metric = np.array(df[key_metric])

                    # [seed]
                    metric_seed = np.mean(metric, axis=0)
                    print(metric_seed.shape)

                    metric_map.append(metric_seed)

                # [map, seed] -> [seed]
                print()
                metric_seed_again = np.mean(metric_map, axis = 0)

                metric_ratio.append(metric_seed_again)
            
            # [ratio, seed]
            y_seed = np.array(metric_ratio)

            mean_seed = np.mean(y_seed, axis=1)
            std_seed = np.std(y_seed, axis=1)
            mean_seed = np.concatenate([np.zeros(1,), mean_seed])
            std_seed = np.concatenate([np.zeros(1,), std_seed])

            x_step = np.arange(len(x_names)) + 1

            plt.plot(x_step, mean_seed, label = label_name, linewidth=4)#, color=color_name)
            plt.fill_between(x_step,
                mean_seed - std_seed,
                mean_seed + std_seed,
                #color=color_name,
                alpha=0.1)


plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(0.15)
y_minor_Locator = MultipleLocator(0.075)
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# ax.xaxis.set_minor_locator(x_minor_Locator)
ax.yaxis.set_minor_locator(y_minor_Locator)
# ax.xaxis.get_major_formatter().set_powerlimits((0,1))
tx = ax.xaxis.get_offset_text() 
tx.set_fontsize(18) 
#ax.xaxis.grid(True, which='minor')
plt.xlim(1, 1.9)
plt.ylim([0.0, 1.0])
plt.xticks(np.arange(len(x_names)) + 1 , x_names)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Coverage Ratio', fontsize=20)
plt.ylabel('Overlap Ratio', fontsize=20)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=18, handlelength=0.9)

plt.savefig(save_dir + "overlap_3agents.png", bbox_inches="tight")

                
    