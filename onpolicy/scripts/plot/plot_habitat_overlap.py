import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator

plt.style.use('ggplot')

map_names = ['16','21','22','36','43','48','61']
large_map_names = ['20','49']
method_names = ['nearest','apf','utility', 'rrt', 'global_stack']
label_names = ['Nearest','APF','Utility', 'RRT']#'MAANS']
color_names = ['limegreen', 'saddlebrown','purple','blue','red','gray']
metric_names = ['overlap']
ratio_names = ['30ratio','50ratio','70ratio','90ratio']
x_names = ['30%', '50%', '70%', '90%']

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
            print(y_seed.shape)

            mean_seed = np.mean(y_seed, axis=1)
            std_seed = np.std(y_seed, axis=1)

            x_step = np.arange(len(ratio_names)) + 1

            plt.plot(x_step, mean_seed, label = label_name, color=color_name)
            plt.fill_between(x_step,
                mean_seed - std_seed,
                mean_seed + std_seed,
                color=color_name,
                alpha=0.1)


plt.tick_params(axis='both',which='major') 
# final_max_step = 300
# x_major_locator = MultipleLocator(50)
# x_minor_Locator = MultipleLocator(10) 
y_major_locator = MultipleLocator(0.1)
y_minor_Locator = MultipleLocator(0.05)
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
plt.ylim([0.3, 0.7])
plt.xticks(np.arange(len(x_names)) + 1 , x_names)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel('Agents', fontsize=20)
plt.ylabel('Overlap Ratio', fontsize=15)
# plt.title('Comparison of AUC on Middle Maps', fontsize=20)
plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=15, handlelength=0.8)

plt.savefig(save_dir + "overlap_middle.png", bbox_inches="tight")

                
    