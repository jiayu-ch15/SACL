
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

map_names = ['16','48']#,'21','48']#,'22','36','43','48','49','61']
id_names = ['Colebrook','Quantico']
title_names = ['Map: ' + m for m in id_names]
method_names = ['global_stack','global_stack_noattn','global_stack_xy']
label_names = ['SCP','SCP w.o. RE','SCP w.o. AE']
color_names = ['red','blue','limegreen', 'saddlebrown','purple','gray']
metric_names = ['auc','overlap','step']
step_names = ['200step']
ratio_names = ['90ratio']

save_dir = './habitat_plot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for map_name, title_name in zip(map_names, title_names):
    
    for metric_name in metric_names:
        if metric_name == "auc":
            for step_name in step_names:
                plt.figure()
                max_steps = []
                for method_name, label_name, color_name in zip(method_names, label_names, color_names):
                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + step_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    df_final = df[key_cols].dropna()
                    step = df_final[key_step]
                    max_step = step.max()['Step']

                    max_steps.append(max_step)

                    df_final = df_final.loc[df_final['Step'] <= max_step] 

                    x_step = np.array(df_final[key_step]).squeeze(-1)
                    y_seed = np.array(df_final[key_metric]).transpose()
                    y_seed = [moving_average(y, avg) for y in y_seed]
                    y_seed = np.array(y_seed)[:, avg:-avg]
                    x_step = x_step[avg:-avg]

                    mean_seed = np.mean(y_seed, axis=0)
                    std_seed = np.std(y_seed, axis=0)
                    plt.plot(x_step, mean_seed, label = label_name, color=color_name)
                    plt.fill_between(x_step,
                        mean_seed - std_seed,
                        mean_seed + std_seed,
                        color=color_name,
                        alpha=0.1)

                plt.tick_params(axis='both',which='major') 
                final_max_step = np.min(max_steps)
                if final_max_step < 1e6:
                    final_max_step = 1e6
                elif final_max_step < 1.5e6:
                    final_max_step = 1.5e6
                elif final_max_step < 2e6:
                    final_max_step = 2e6
                elif final_max_step < 2.5e6:
                    final_max_step = 2.5e6
                else:
                    final_max_step = 3e6
                # print("final max step is {}".format(final_max_step))
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
                plt.xlim(50000, final_max_step)
                # plt.ylim([0, 1.1])
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Timesteps', fontsize=20)
                plt.ylabel("AUC", fontsize=20)
                plt.title(title_name, fontsize=20)
                plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

                plt.savefig(save_dir + map_name + "_" + step_name + '_' + metric_name + "_ARE-module.png", bbox_inches="tight")
            
        if metric_name == "overlap":
            for ratio_name in ratio_names:
                plt.figure()
                max_steps = []
                for method_name, label_name, color_name in zip(method_names, label_names, color_names):
                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    df_final = df[key_cols].dropna()
                    step = df_final[key_step]
                    max_step = step.max()['Step']

                    max_steps.append(max_step)

                    df_final = df_final.loc[df_final['Step'] <= max_step] 

                    x_step = np.array(df_final[key_step]).squeeze(-1)
                    y_seed = np.array(df_final[key_metric]).transpose()

                    y_seed = [moving_average(y, avg) for y in y_seed]
                    y_seed = np.array(y_seed)[:, avg:-avg]
                    x_step = x_step[avg:-avg]

                    mean_seed = np.mean(y_seed, axis=0)
                    std_seed = np.std(y_seed, axis=0)
                    plt.plot(x_step, mean_seed, label = label_name, color = color_name)
                    plt.fill_between(x_step,
                        mean_seed - std_seed,
                        mean_seed + std_seed,
                        color=color_name,
                        alpha=0.1)

                plt.tick_params(axis='both',which='major') 
                final_max_step = np.min(max_steps)
                if final_max_step < 1e6:
                    final_max_step = 1e6
                elif final_max_step < 1.5e6:
                    final_max_step = 1.5e6
                elif final_max_step < 2e6:
                    final_max_step = 2e6
                elif final_max_step < 2.5e6:
                    final_max_step = 2.5e6
                else:
                    final_max_step = 3e6
                # print("final max step is {}".format(final_max_step))
                x_major_locator = MultipleLocator(int(final_max_step/5))
                x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
                y_major_locator = MultipleLocator(0.1)
                y_minor_Locator = MultipleLocator(0.05)
                ax=plt.gca()
                ax.xaxis.set_major_locator(x_major_locator)
                ax.yaxis.set_major_locator(y_major_locator)
                ax.xaxis.set_minor_locator(x_minor_Locator)
                ax.yaxis.set_minor_locator(y_minor_Locator)
                ax.xaxis.get_major_formatter().set_powerlimits((0,1))
                tx = ax.xaxis.get_offset_text() 
                tx.set_fontsize(18) 
                #ax.xaxis.grid(True, which='minor')
                plt.xlim(50000, final_max_step)
                # plt.ylim([0, 1.1])
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Timesteps', fontsize=20)
                plt.ylabel("Overlap Ratio", fontsize=20)
                plt.title(title_name, fontsize=20)
                plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

                plt.savefig(save_dir + map_name + "_" + ratio_name + '_' + metric_name + "_ARE-module.png", bbox_inches="tight")

        if metric_name == "step":
            plt.figure()
            max_steps = []
            for method_name, label_name, color_name in zip(method_names, label_names, color_names):
                data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step']

                df_final = df[key_cols].dropna()
                step = df_final[key_step]
                max_step = step.max()['Step']

                max_steps.append(max_step)

                df_final = df_final.loc[df_final['Step'] <= max_step] 

                x_step = np.array(df_final[key_step]).squeeze(-1)
                y_seed = np.array(df_final[key_metric]).transpose()

                y_seed = [moving_average(y, avg) for y in y_seed]
                y_seed = np.array(y_seed)[:, avg:-avg]
                x_step = x_step[avg:-avg]

                mean_seed = np.mean(y_seed, axis=0)
                std_seed = np.std(y_seed, axis=0)
                plt.plot(x_step, mean_seed, label = label_name, color = color_name)
                plt.fill_between(x_step,
                    mean_seed - std_seed,
                    mean_seed + std_seed,
                    color=color_name,
                    alpha=0.1)

            plt.tick_params(axis='both',which='major')
            final_max_step = np.min(max_steps)
            if final_max_step < 1e6:
                final_max_step = 1e6
            elif final_max_step < 1.5e6:
                final_max_step = 1.5e6
            elif final_max_step < 2e6:
                final_max_step = 2e6
            elif final_max_step < 2.5e6:
                final_max_step = 2.5e6
            else:
                final_max_step = 3e6
            # print("final max step is {}".format(final_max_step))
            x_major_locator = MultipleLocator(int(final_max_step/5))
            x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
            y_major_locator = MultipleLocator(30)
            y_minor_Locator = MultipleLocator(15)
            ax=plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.xaxis.set_minor_locator(x_minor_Locator)
            ax.yaxis.set_minor_locator(y_minor_Locator)
            ax.xaxis.get_major_formatter().set_powerlimits((0,1))
            tx = ax.xaxis.get_offset_text() 
            tx.set_fontsize(18) 
            #ax.xaxis.grid(True, which='minor')
            plt.xlim(50000, final_max_step)
            # plt.ylim([0, 1.1])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('Timesteps', fontsize=20)
            plt.ylabel("Steps", fontsize=20)
            plt.title(title_name, fontsize=20)
            plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)

            plt.savefig(save_dir + map_name + "_" + metric_name + "_ARE-module.png", bbox_inches="tight")