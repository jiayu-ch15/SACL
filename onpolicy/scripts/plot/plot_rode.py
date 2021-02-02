from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
import pandas

plt.style.use('ggplot')


def read_tensorboard_data(tensorboard_path, val_name):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    val = ea.scalars.Items(val_name)
    return val
    
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            L.append(os.path.join(root, file))  
    return L  

if __name__ == "__main__":
    map_names = ['2s_vs_1sc','2s3z',\
'5m_vs_6m','3s5z','1c3s5z','27m_vs_30m','bane_vs_bane','3s_vs_5z','6h_vs_8z',\
'corridor','3s5z_vs_3s6z','10m_vs_11m','MMM2','2c_vs_64zg']

    for map_name in map_names:
        plt.figure()
        

        ############################################################
        max_steps = []
        exp_name = "final_mappo"
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
        plt.plot(x_step, median_seed, label = 'MAPPO', color='red')
        plt.fill_between(x_step,
            median_seed - std_seed,
            median_seed + std_seed,
            color='red',
            alpha=0.1)




        print("########################QMIX##########################")
        exp_name = 'final_qmix'
        data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)

        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']


        one_run_max_step = []
        qmix_x_step = []
        qmix_y_seed = []
        for k in key_win_rate:
            print("one run original shape is ")
            print(np.array(df[k]).shape)

            df_final = df[[k, 'Step']].dropna()
            step = df_final[key_step]
            win_rate = df_final[k]
            
            print("one run drop nan shape is")
            print(np.array(step).shape)
            print(np.array(win_rate).shape)

            max_step = step.max()['Step']
            print("one run max step is {}".format(max_step))

            if max_step < 2e6:
                print("error: broken data! double check!")
                continue

            if max_step < 4e6:
                max_step = 2e6
            elif max_step < 9e6:
                max_step = 5e6
            else:
                max_step = 10e6

            if max_step == 2e6 and map_name in ["6h_vs_8z","MMM"]:
                continue
            
            one_run_max_step.append(max_step)
            print("final step is {}".format(max_step))

            df_final = df_final.loc[df_final['Step'] <= max_step] 
            qmix_x_step.append(np.array(df_final[key_step]).squeeze(-1))
            qmix_y_seed.append(np.array(df_final[k]))
            print("data shape is {}".format(np.array(df_final[k]).shape))

        # pick max qmix step
        qmix_max_step = np.min(one_run_max_step)
        max_steps.append(qmix_max_step)

        # adapt sample frequency
        sample_qmix_s_step = []
        sample_qmix_y_seed = []
        final_max_length = []
        for x, y in zip(qmix_x_step, qmix_y_seed):
            eval_interval = x[1] - x[0]
            if eval_interval - 10000 < 5000: # eval_interval = 10000
                print("warning: better not to use mixed data, try to one eval_interval")
                if map_name == "2m_vs_1z":
                    print("warning: drop this run!")
                    continue
                final_max_length.append(len(x[::8]))
                sample_qmix_s_step.append(x[::8])
                sample_qmix_y_seed.append(y[::8])
            elif eval_interval - 20000 < 5000: # eval_interval = 20000
                final_max_length.append(len(x[::4]))
                sample_qmix_s_step.append(x[::4])
                sample_qmix_y_seed.append(y[::4])
            elif eval_interval - 80000 < 5000: # eval_interval = 80000
                print("warning: better not to use mixed data, try to one eval_interval")
                if map_name == "2c_vs_64zg":
                    print("warning: drop this run!")
                    continue
                final_max_length.append(len(x))
                sample_qmix_s_step.append(x)
                sample_qmix_y_seed.append(y)
            else:
                raise NotImplementedError

        # truncate numpy
        max_common_length = np.min(final_max_length)
        print("max common qmix length is {}".format(max_common_length))
        final_qmix_x_step = []
        final_qmix_y_seed = []
        for x, y in zip(sample_qmix_s_step, sample_qmix_y_seed):
            final_qmix_x_step.append(x[:max_common_length])
            final_qmix_y_seed.append(y[:max_common_length])

        x_step = np.mean(final_qmix_x_step, axis=0)
        y_seed = np.array(final_qmix_y_seed)

        median_seed = np.median(y_seed, axis=0)
        std_seed = np.std(y_seed, axis=0)
        plt.plot(x_step, median_seed, label='QMIX', color='limegreen')
        plt.fill_between(x_step,
            median_seed - std_seed,
            median_seed + std_seed,
            color='limegreen',
            alpha=0.1)


        ####################################RODE##############################
        val_name = 'test_battle_won_mean'
        tensorboard_path = './RODE/' + map_name + '/tb_logs/'
        save_dir = './RODE/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_list = file_name(tensorboard_path)
        
        win_rate = []
        step = []     
        length = []
        for dl in data_list:
            val = read_tensorboard_data(dl, val_name)
            x = []
            y = []
            for v in val:
                x.append(v.step)
                y.append(v.value)
            step.append(np.array(x))
            length.append(len(x))
            win_rate.append(np.array(y))

        common_length = np.min(length)
        x_step = []
        y_seed = []
        for x, y in zip(step, win_rate):
            x_step.append(x[:common_length][::8])
            y_seed.append(y[:common_length][::8])

        x_step = np.mean(x_step, axis=0)
        y_seed = np.array(y_seed)

        median_seed = np.median(y_seed, axis=0)
        std_seed = np.std(y_seed, axis=0)
        plt.plot(x_step, median_seed, label = 'RODE', color='blue')
        plt.fill_between(x_step,
            median_seed - std_seed,
            median_seed + std_seed,
            color='blue',
            alpha=0.1)


        plt.tick_params(axis='both',which='major') 
        final_max_step = np.min(max_steps)
        if map_name in ["3s5z"]:
            final_max_step = 5e6

        # if map_name in ["1c3s5z","2s3z","2s_vs_1sc"]:
        #     final_max_step = 2e6

        

        print("final max step is {}".format(final_max_step))
        x_major_locator = MultipleLocator(int(final_max_step/5))
        x_minor_Locator = MultipleLocator(int(final_max_step/10)) 
        y_major_locator = MultipleLocator(0.2)
        y_minor_Locator = MultipleLocator(0.1)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_minor_locator(x_minor_Locator)
        ax.yaxis.set_minor_locator(y_minor_Locator)
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        #ax.xaxis.grid(True, which='minor')
        plt.xlim(0, final_max_step)
        plt.ylim([0, 1.1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Timesteps', fontsize=20)
        plt.ylabel('Win Rate', fontsize=20)
        plt.title(map_name, fontsize=20)
        plt.legend(loc='best', numpoints=1, fancybox=True, fontsize=20)
        plt.savefig(save_dir + map_name + "_mappo_qmix_rode.png", bbox_inches="tight")
