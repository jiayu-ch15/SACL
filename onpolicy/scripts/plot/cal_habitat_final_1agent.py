# python cal_habitat.py > latex.txt
import pandas
import json
import numpy as np
import sys
import os
from icecream import ic
from collections import defaultdict

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


map_names = ['27']#['16','20','21','22','36', '43','48','49','61']##
train_method_names = ['apf','global_stack']#['xy','multi_dis','no_attn', 'grid_simple', 'ans#,'distill']
train_label_names = ['apf','global_stack']#['2_Heads','Multi_Dis','No_Attn',  'Grid_Simple','ANS','MAANS']#,'MAANS-TD']
metric_names = ['repeat','step']#,'ratio','overlap']#auc','auc',
ratio_names = ['repeat_ratio']
step_names = ['180step'] 
save_dir = './habitat_1agent/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

value_dict = defaultdict(list)
value_dict['Methods'] = train_label_names.copy()
for metric_name in metric_names:
    
    # auc # agent_auc
    if metric_name in ["auc", "agent_auc"]:
        ic(metric_name)
        for step_name in step_names:
            ic(step_name + metric_name)
            
            for method_name, dict_name in zip(train_method_names, train_label_names):
                ic(method_name)
                avg_mean = []
                avg_std = []
                for map_name in map_names:
                    ic(map_name)
                    
                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + step_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c and 'seed' in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step' and 'seed' in n]

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])

                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))

                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    # result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    # value_dict[method_name].append(result)
                
                result = str(format(np.mean(avg_mean), '.2f')) + "scriptsize{(" + str(format(np.mean(avg_std), '.2f')) + ")}"
                value_dict['ACS'].append(result)
    # overlap
    if metric_name == "overlap":
        ic(metric_name)
        for ratio_name in ratio_names:
            for method_name, dict_name in zip(train_method_names, train_label_names):
                if method_name == "single_agent": break
                ic(method_name)
                avg_mean = []
                avg_std = []
                for map_name in map_names:
                    ic(map_name)

                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c and 'seed' in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step' and 'seed' in n]

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])

                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))

                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    # result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    # value_dict[method_name].append(result)
                result = str(format(np.mean(avg_mean), '.2f')) + "scriptsize{(" + str(format(np.mean(avg_std), '.2f')) + ")}"
                value_dict['Overlap Ratio'].append(result)
    # repeat_ratio
    if metric_name == "repeat":
        ic(metric_name)
        for ratio_name in ratio_names:
            for method_name, dict_name in zip(train_method_names, train_label_names):
                #if method_name != "single_agent": break
                ic(method_name)
                avg_mean = []
                avg_std = []
                for map_name in map_names:
                    ic(map_name)

                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'
                    #ratio_data_dir =  save_dir + map_name + '/' + method_name + "/" + 'ratio' + '/' + 'ratio' + '.csv'
                    df = pandas.read_csv(data_dir)
                    #ratio_df = pandas.read_csv(ratio_data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c and 'seed' in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step' and 'seed' in n]
                    
                    #ratio_key_cols = [c for c in ratio_df.columns if 'MIN' not in c and 'MAX' not in c and 'seed' in c]
                    #ratio_key_metric = [n for n in ratio_key_cols if n != 'Step' and 'seed' in n]

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])
                    #ratio_metric = np.array(ratio_df[ratio_key_metric])
                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))
                    
                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    # result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    # value_dict[method_name].append(result)
                result = str(format(np.mean(avg_mean), '.2f')) + "scriptsize{(" + str(format(np.mean(avg_std), '.2f')) + ")}"
                value_dict['Repeat Ratio'].append(result)

    # ratio # balance #step
    if metric_name in ["ratio","balance","step",'repeat']:
        ic(metric_name)
        for method_name, dict_name in zip(train_method_names, train_label_names):
            if method_name == "single_agent" and metric_name == "balance": break
            ic(method_name)
            avg_mean = []
            avg_std = []
            for map_name in map_names:
                ic(map_name)
                data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c and 'seed' in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step' and 'seed' in n]

                step = np.array(df[key_step])
                metric = np.array(df[key_metric])

                metric_mean = np.nanmean(np.nanmean(metric, axis=0))
                metric_std = np.nanstd(np.nanmean(metric, axis=0))

                avg_mean.append(metric_mean)
                avg_std.append(metric_std)

            result = str(format(np.nanmean(avg_mean), '.2f')) + "scriptsize{(" + str(format(np.nanmean(avg_std), '.2f')) + ")}"
            
            if metric_name == "ratio":
                value_dict['Coverage Ratio'].append(result)

            if metric_name == "step":
                value_dict['Steps'].append(result)
            if metric_name == "repeat":
                value_dict['Rep'].append(result)

    # ratio # balance #step
    if metric_name in ["success rate"]:
        ic(metric_name)
        for method_name, dict_name in zip(train_method_names, train_label_names):
            ic(method_name)
            avg_mean = []
            avg_std = []
            for map_name in map_names:
                ic(map_name)
                data_dir =  save_dir + map_name + '/' + method_name + "/ratio/ratio.csv"

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step']

                step = np.array(df[key_step])
                metric = np.array(df[key_metric])
                metric = metric > 0.9

                metric_mean = np.nanmean(np.nanmean(metric, axis=0))
                metric_std = np.nanstd(np.nanmean(metric, axis=0))

                avg_mean.append(metric_mean)
                avg_std.append(metric_std)

            result = str(format(np.nanmean(avg_mean), '.2f')) + "scriptsize{(" + str(format(np.nanmean(avg_std), '.2f')) + ")}"
            value_dict['Success'].append(result)

df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption='Average performance on trained maps', label='tab:trained'))

