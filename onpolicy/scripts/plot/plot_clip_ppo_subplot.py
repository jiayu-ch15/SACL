
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
save_dir = './subplot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# map_names = ['MMM2','6h_vs_8z','corridor','5m_vs_6m','3s5z_vs_3s6z']
map_names = ['5m_vs_6m','3s5z_vs_3s6z']
title_names = [name.replace("_vs_"," vs. ") for name in map_names]


for title_name, map_name in zip(title_names, map_names):

    fig, axes = plt.subplots(1, 5, figsize=(36, 4))
    lines = []

    # clip_names = ['clip0.05','clip0.1','clip0.2','clip0.3','clip0.5']
    clip_names = ['clip0.05','clip0.1', 'clip0.15','clip0.2','clip0.3']
    fig_names = ['Clip 0.05', 'Clip 0.1', 'Clip 0.15', 'Clip 0.2','Clip 0.3']

    for clip_name, fig_name, ax in zip(clip_names, fig_names, axes):

        ###################################PPO###################################
        exp_names = ['ppo5', 'ppo10', 'ppo15'] 
        label_names = ["5 epochs", "10 epochs", "15 epochs"]
        color_names = ['red','blue','limegreen']
               
        max_steps = []
        for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
            data_dir =  './clip_ppo/' + map_name + '/' + clip_name + '/' + map_name + '_' + clip_name + '_' + exp_name +'.csv'

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

            if "ppo" in exp_name and max_step < 4.96e6:
                print("error: broken data! double check!")

            if max_step < 4e6:
                max_step = 2e6
            elif max_step < 9e6:
                max_step = 5e6
            else:
                max_step = 10e6

            print("final step is {}".format(max_step))
            max_steps.append(max_step)

            df_final = df_final.loc[df_final['Step'] <= max_step] 

            x_step = np.array(df_final[key_step]).squeeze(-1)
            y_seed = np.array(df_final[key_win_rate])

            median_seed = np.median(y_seed, axis=1)
            std_seed = np.std(y_seed, axis=1)
            lines.append(ax.plot(x_step, median_seed, label = label_name, color=color_name))
            ax.fill_between(x_step,
                median_seed - std_seed,
                median_seed + std_seed,
                color=color_name,
                alpha=0.1)

        ax.tick_params(axis='both',which='major') 
        final_max_step = np.min(max_steps)
        print("final max step is {}".format(final_max_step))
        x_major_locator = MultipleLocator(int(final_max_step/5))
        x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
        y_major_locator = MultipleLocator(0.2)
        y_minor_Locator = MultipleLocator(0.1)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_minor_locator(x_minor_Locator)
        ax.yaxis.set_minor_locator(y_minor_Locator)
        ax.xaxis.get_major_formatter().set_powerlimits((0,2))
        tx = ax.xaxis.get_offset_text() 
        tx.set_fontsize(18) 
        #ax.xaxis.grid(True, which='minor')
        ax.set_xlim(0, final_max_step)
        ax.set_ylim([0, 1.1])
        ax.set_xlabel('Timesteps', fontsize=20)
        ax.set_ylabel('Win Rate', fontsize=20)
        ax.set_title(fig_name, fontsize=20)
        ax.tick_params(labelsize=20)

    # fig.legend(lines,     # The line objects
    #            labels=label_names,   # The labels for each line
    #            loc="lower center",   # Position of legend
    #            borderaxespad=0.1,    # Small spacing around legend box
    #            #title="RR",  # Title for the legend
    #            bbox_to_anchor=(0.5, -0.23),
    # 		   #bbox_transform=axes[2].transAxes,
    #            ncol=len(label_names),
    #            fontsize=20
    #            )

    fig.legend(lines,     # The line objects
            labels=label_names,   # The labels for each line
            loc="right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            #title="RR",  # Title for the legend
            bbox_to_anchor=(0.918, 0.28),
            #bbox_transform=axes[2].transAxes,
            ncol=1,
            fontsize=20
            )

    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=0.85)
    plt.savefig(save_dir + map_name + "_clip_ppo_subplot_rightlegend.png", bbox_inches="tight")