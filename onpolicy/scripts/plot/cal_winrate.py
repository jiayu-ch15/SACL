
import pandas
import json
import numpy as np
import sys
import os

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

# map_names = ['2s_vs_1sc','2s3z','3m','3s_vs_3z','3s_vs_4z','2m_vs_1z','MMM','so_many_baneling',\
# '5m_vs_6m','3s5z','1c3s5z','8m','27m_vs_30m','25m','bane_vs_bane','3s_vs_5z','6h_vs_8z',\
# 'corridor','3s5z_vs_3s6z','10m_vs_11m','8m_vs_9m','MMM2','2c_vs_64zg']

easy_maps = ["2m_vs_1z", "3m", "2s_vs_1sc", "2s3z", "3s_vs_3z", "3s_vs_4z", "so_many_baneling", "8m", "MMM", "1c3s5z", "bane_vs_bane"]
hard_maps = ["3s_vs_5z", "2c_vs_64zg", "8m_vs_9m", "25m", "5m_vs_6m", "3s5z", "10m_vs_11m"]
super_hard_maps = ["MMM2", "3s5z_vs_3s6z", "27m_vs_30m", "6h_vs_8z", "corridor"]

map_names = easy_maps + hard_maps + super_hard_maps
difficulties = len(easy_maps) * ["Easy"] + len(hard_maps) * ["Hard"] + len(super_hard_maps) * ["Super Hard"]

rode_scores = ["/", "/", "100(0.0)", "100(0.0)", "/", "/", "/", "/", "/", "100(0.0)",\
"100(46.4)", "78.9(4.2)", "100(0.0)", "/", "/", "71.1(9.2)", "93.75(1.95)", "95.3(2.2)",\
"89.8(6.7)", "96.8(25.11)", "96.8(1.5)", "78.1(37.0)", "65.6(32.1)"]
qplex_scores = ["/", "/", "98.4(1.6)", "100(4.3)", "/", "/", "/", "/", "/", "96.8(1.6)", \
"100(2.9)", "98.4(1.4)", "90.6(7.3)", "/", "/", "70.3(3.2)", "96.8(2.2)", "96.1(8.7)",\
"82.8(20.8)", "10.2(11.0)", "43.7(18.7)", "1.5(31.0)", "0.0(0.0)"]
cwqmix_scores = ["/", "/", "100(0.0)", "93.7(2.2)", "/", "/", "/", "/", "/", "96.9(1.4)", \
"100(0.0)", "34.4(6.5)", "85.9(3.3)", "/", "/", "57.8(9.1)", "70.3(20.3)", "75.0(3.3)",\
"0.0(0.0)", "53.1(12.9)", "82.8(7.8)", "49.2(14.8)","0.0(0.0)"]
aiqmix_scores = ["/", "/", "100(0.0)", "96.9(0.7)", "/", "/", "/", "/", "/", "92.2(10.4)", \
"85.9(34.7)", "82.8(10.6)", "97.6(2.3)", "/", "/", "64.1(5.5)", "96.9(2.9)", "96.9(1.4)",\
"67.2(12.4)", "0.0(0.0)", "62.5(34.3)", "0.0(0.0)", "12.5(7.6)"]

all_final_max_step = []
for map_name in map_names:
    print("########################MAP##########################")
    print(map_name)
    ###################################PPO###################################
    exp_names = ['final_mappo', 'final_ippo', 'final_calgl_dead'] 
    label_names = ["MAPPO", "IPPO", "MAPPO_share_local"]
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
    if map_name in ["5m_vs_6m","3s5z_vs_3s6z"]:
        final_max_step = 25e6
    
    if map_name == "27m_vs_30m":
        final_max_step = 7e6

    all_final_max_step.append(final_max_step)

#################################CAL#####################################

results = []
for map_name, final_max_step in zip(map_names, all_final_max_step):
    print("########################MAP##########################")
    print(map_name)
    print(final_max_step)
    ###################################PPO###################################
    exp_names = ['final_mappo', 'final_calgl_dead', 'final_ippo']

    median_value = []
    std_value = []
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

        df_final = df_final.loc[df_final['Step'] <= final_max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        y_seed_last = np.array(y_seed)[-10:]

        median_seed = np.median(np.median(y_seed_last, axis=0))
        std_seed = np.std(np.median(y_seed_last, axis=0))
        median_value.append(str(format(median_seed*100, '.1f')) + "(" + str(format(std_seed*100, '.1f')) + ")")

   
    exp_names = ['final_qmix', 'final_qmix_all']
    
    for exp_name in exp_names:
        print(exp_name)
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
        median_value.append(str(format(median_seed*100, '.1f')) + "(" + str(format(std_seed*100, '.1f')) + ")")


    if map_name in ["27m_vs_30m","corridor","6h_vs_8z","3s5z_vs_3s6z"]:
        cut_max_step = 5e6
    else:
        cut_max_step = 2e6
    
    print("########################MAP##########################")
    print(map_name)
    ###################################PPO###################################
    exp_names = ['final_mappo', 'final_calgl_dead', 'final_ippo'] 

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

        df_final = df_final.loc[df_final['Step'] <= cut_max_step] 

        x_step = np.array(df_final[key_step]).squeeze(-1)
        y_seed = np.array(df_final[key_win_rate])

        y_seed_last = np.array(y_seed)[-5:]

        median_seed = np.median(np.median(y_seed_last, axis=0))
        std_seed = np.std(np.median(y_seed_last, axis=0))
        median_value.append(str(format(median_seed*100, '.1f')) + "(" + str(format(std_seed*100, '.1f')) + ")")

    exp_names = ['final_qmix', 'final_qmix_all']
    
    for exp_name in exp_names:
        print(exp_name)

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

            df_final = df_final.loc[df_final['Step'] <= cut_max_step] 
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
                sample_qmix_y_seed.append(y[::8][-5:])
            elif eval_interval - 20000 < 5000: # eval_interval = 20000
                sample_qmix_y_seed.append(y[::4][-5:])
            elif eval_interval - 80000 < 5000: # eval_interval = 80000
                print("warning: better not to use mixed data, try to one eval_interval")
                if map_name not in ["25m","27m_vs_30m","bane_vs_bane"]:
                    print(map_name)
                    print(eval_interval)
                sample_qmix_y_seed.append(y[-5:])
            else:
                raise NotImplementedError

        median_seed = np.median(np.median(np.array(sample_qmix_y_seed), axis=0))
        std_seed = np.std(np.median(np.array(sample_qmix_y_seed), axis=0))    
        median_value.append(str(format(median_seed*100, '.1f')) + "(" + str(format(std_seed*100, '.1f')) + ")")
    results.append(np.array(median_value))

results = np.array(results)
print(results.shape)
df = pandas.DataFrame({'Map': np.array(map_names), "Map Difficulty": np.array(difficulties), 'MAPPO(FP)': results[:,0], 'MAPPO(AS)': results[:,1], 'IPPO': results[:,2], 'QMix': results[:,3], 'QMix(AS)': results[:,4], 'RODE': np.array(rode_scores), 'QPLEX': np.array(qplex_scores), 'CWQMix': np.array(cwqmix_scores), 'AIQMix': np.array(aiqmix_scores)})
print(df.to_latex(index=False))

df = pandas.DataFrame({'Map': np.array(map_names), "Map Difficulty": np.array(difficulties), 'MAPPO(FP)-c': results[:,5], 'MAPPO(AS)-c': results[:,6], 'IPPO-c': results[:,7], 'QMix-c': results[:,8], 'QMix(AS)-c': results[:,9], 'RODE': np.array(rode_scores), 'QPLEX': np.array(qplex_scores), 'CWQMix': np.array(cwqmix_scores), 'AIQMix': np.array(aiqmix_scores)})
print(df.to_latex(index=False))
