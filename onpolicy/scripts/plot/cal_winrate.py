
import pandas
import json
import numpy as np
import sys
import os

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

map_names = ['2s_vs_1sc','2s3z','3m','3s_vs_3z','3s_vs_4z','2m_vs_1z','MMM','so_many_baneling',\
'5m_vs_6m','3s5z','1c3s5z','8m','27m_vs_30m','25m','bane_vs_bane','3s_vs_5z','6h_vs_8z',\
'corridor','3s5z_vs_3s6z','10m_vs_11m','8m_vs_9m','MMM2','2c_vs_64zg']

all_final_max_step = []
for map_name in map_names:
    print("########################MAP##########################")
    print(map_name)
    ###################################PPO###################################
    exp_names = ['final_mappo', 'final_ippo', 'final_mappo_original'] 
    label_names = ["MAPPO", "IPPO", "MAPPO_original"]
    color_names = ['red','blue','limegreen']

    save_dir = './win_rate/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_steps = []
    for exp_name, label_name, color_name in zip(exp_names, label_names, color_names):
        data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']

        all_step = np.array(df[key_step])
        all_win_rate = np.array(df[key_win_rate])

        # print("original shape is ")
        # print(all_step.shape)
        # print(all_win_rate.shape)

        df_final = df[key_cols].dropna()
        step = df_final[key_step]
        win_rate = df_final[key_win_rate]

        # print("drop nan shape is")
        # print(np.array(step).shape)
        # print(np.array(win_rate).shape)

        max_step = step.max()['Step']
        # print("max step is {}".format(max_step))

        if "ppo" in exp_name and max_step < 4.96e6:
            print("error: broken data! double check!")
            print("drop one run!")
            continue

        if max_step < 4e6:
            max_step = 2e6
        elif max_step < 9e6:
            max_step = 5e6
        else:
            max_step = 10e6

        # print("final step is {}".format(max_step))
        max_steps.append(max_step)

        df_final = df_final.loc[df_final['Step'] <= max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        median_seed = np.median(y_seed, axis=1)
        std_seed = np.std(y_seed, axis=1)

    
    exp_name = 'final_qmix'
    label_name = "QMIX"
    color_name = 'saddlebrown'
    data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'

    df = pandas.read_csv(data_dir)

    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
    key_step = [n for n in key_cols if n == 'Step']
    key_win_rate = [n for n in key_cols if n != 'Step']


    one_run_max_step = []
    qmix_x_step = []
    qmix_y_seed = []
    for k in key_win_rate:
        # print("one run original shape is ")
        # print(np.array(df[k]).shape)

        df_final = df[[k, 'Step']].dropna()
        step = df_final[key_step]
        win_rate = df_final[k]
        
        # print("one run drop nan shape is")
        # print(np.array(step).shape)
        # print(np.array(win_rate).shape)

        max_step = step.max()['Step']
        # print("one run max step is {}".format(max_step))

        if max_step < 2e6:
            print("error: broken data! double check!")
            print("drop qmix run!")
            continue

        if max_step < 4e6:
            max_step = 2e6
        elif max_step < 9e6:
            max_step = 5e6
        else:
            max_step = 10e6

        # if max_step == 2e6 and map_name in ["6h_vs_8z","MMM"]:
        #     continue
        
        one_run_max_step.append(max_step)
        # print("final step is {}".format(max_step))

        df_final = df_final.loc[df_final['Step'] <= max_step] 
        qmix_x_step.append(np.array(df_final[key_step]).squeeze(-1))
        qmix_y_seed.append(np.array(df_final[k]))
        # print("data shape is {}".format(np.array(df_final[k]).shape))

    # pick max qmix step
    qmix_max_step = np.min(one_run_max_step)
    max_steps.append(qmix_max_step)

    # adapt sample frequency
    sample_qmix_s_step = []
    sample_qmix_y_seed = []
    final_max_length = []
    for x, y in zip(qmix_x_step, qmix_y_seed):
        eval_interval = x[10] - x[9]
        if eval_interval - 10000 < 5000: # eval_interval = 10000
            print("warning: better not to use mixed data, try to one eval_interval")
            print(map_name)
            print(eval_interval)
            final_max_length.append(len(x[::8]))
            sample_qmix_s_step.append(x[::8])
            sample_qmix_y_seed.append(y[::8])
        elif eval_interval - 20000 < 5000: # eval_interval = 20000
            final_max_length.append(len(x[::4]))
            sample_qmix_s_step.append(x[::4])
            sample_qmix_y_seed.append(y[::4])
        elif eval_interval - 80000 < 5000: # eval_interval = 80000
            print("warning: better not to use mixed data, try to one eval_interval")
            if map_name not in ["25m","27m_vs_30m","bane_vs_bane"]:
                print(map_name)
                print(eval_interval)
            final_max_length.append(len(x))
            sample_qmix_s_step.append(x)
            sample_qmix_y_seed.append(y)
        else:
            raise NotImplementedError

    # truncate numpy
    max_common_length = np.min(final_max_length)
    # print("max common qmix length is {}".format(max_common_length))
    final_qmix_x_step = []
    final_qmix_y_seed = []
    for x, y in zip(sample_qmix_s_step, sample_qmix_y_seed):
        final_qmix_x_step.append(x[:max_common_length])
        final_qmix_y_seed.append(y[:max_common_length])

    x_step = np.mean(final_qmix_x_step, axis=0)
    y_seed = np.array(final_qmix_y_seed)

    median_seed = np.median(y_seed, axis=0)
    std_seed = np.std(y_seed, axis=0)

    final_max_step = np.min(max_steps)
    print("final max step is {}".format(final_max_step))
    all_final_max_step.append(final_max_step)

#################################CAL#####################################

median_value = []
std_value = []
for map_name, final_max_step in zip(map_names, all_final_max_step):
    print("########################MAP##########################")
    print(map_name)
    ###################################PPO###################################
    exp_names = ['final_mappo', 'final_ippo']

    max_steps = []
    for exp_name in exp_names:
        print(exp_name)
        data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'

        df = pandas.read_csv(data_dir)
        
        key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
        key_step = [n for n in key_cols if n == 'Step']
        key_win_rate = [n for n in key_cols if n != 'Step']

        all_step = np.array(df[key_step])
        all_win_rate = np.array(df[key_win_rate])

        df_final = df[key_cols].dropna()
        step = df_final[key_step]
        win_rate = df_final[key_win_rate]

        max_step = step.max()['Step']

        if "ppo" in exp_name and max_step < 4.96e6:
            print("error: broken data! double check!")
            print("drop one run!")
            continue

        df_final = df_final.loc[df_final['Step'] <= final_max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        y_seed_last = np.array(y_seed)[-10:]

        median_seed = np.median(np.median(y_seed_last, axis=0))
        std_seed = np.std(np.median(y_seed_last, axis=0))
        print(median_seed, std_seed)

        median_value.append(median_seed)
        std_value.append(std_seed)
   
    exp_name = 'final_qmix'
    data_dir =  './' + map_name + '/' + map_name + '_' + exp_name + '.csv'

    df = pandas.read_csv(data_dir)

    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
    key_step = [n for n in key_cols if n == 'Step']
    key_win_rate = [n for n in key_cols if n != 'Step']

    qmix_x_step = []
    qmix_y_seed = []
    for k in key_win_rate:

        df_final = df[[k, 'Step']].dropna()
        step = df_final[key_step]
        win_rate = df_final[k]

        max_step = step.max()['Step']

        if max_step < 2e6:
            print("error: broken data! double check!")
            print("drop qmix run!")
            continue

        df_final = df_final.loc[df_final['Step'] <= final_max_step] 
        qmix_x_step.append(np.array(df_final[key_step]).squeeze(-1))
        qmix_y_seed.append(np.array(df_final[k]))

    # adapt sample frequency
    sample_qmix_y_seed = []
    for x, y in zip(qmix_x_step, qmix_y_seed):
        eval_interval = x[10] - x[9]
        if eval_interval - 10000 < 5000: # eval_interval = 10000
            print("warning: better not to use mixed data, try to one eval_interval")
            print(map_name)
            print(eval_interval)
            sample_qmix_y_seed.append(y[::8][-10:])
        elif eval_interval - 20000 < 5000: # eval_interval = 20000
            sample_qmix_y_seed.append(y[::4][-10:])
        elif eval_interval - 80000 < 5000: # eval_interval = 80000
            print("warning: better not to use mixed data, try to one eval_interval")
            if map_name not in ["25m","27m_vs_30m","bane_vs_bane"]:
                print(map_name)
                print(eval_interval)
            sample_qmix_y_seed.append(y[-10:])
        else:
            raise NotImplementedError

    median_seed = np.median(np.median(np.array(sample_qmix_y_seed), axis=0))
    std_seed = np.std(np.median(np.array(sample_qmix_y_seed), axis=0))
    print(median_seed, std_seed)
    median_value.append(median_seed)
    std_value.append(std_seed)

    df = pd.DataFrame()
