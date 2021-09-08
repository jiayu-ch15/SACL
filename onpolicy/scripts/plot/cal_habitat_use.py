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


map_names = ['16','21','22','36','43','48','61']
large_map_names = ['20','49']
size_names = ['middle'] * len(map_names) + ['-'] + ['large'] * len(large_map_names) + ['-']
map_id_names = map_names + ['Avg.'] + large_map_names + ['Avg.']
method_names = ['nearest','apf','utility', 'rrt', 'global_stack','single_agent']
metric_names = ['auc','overlap','ratio','step','balance','success rate']
step_names = ['150step','180step','200step']
ratio_names = ['70ratio','90ratio']

save_dir = './habitat/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for metric_name in metric_names:
    # auc # agent_auc
    if metric_name in ["auc", "agent_auc"]:
        ic(metric_name)
        for step_name in step_names:
            ic(step_name + metric_name)
            value_dict = defaultdict(list)
            value_dict['Map ID'] = map_id_names.copy()
            value_dict['size'] = size_names.copy()
            for method_name in method_names:
                ic(method_name)
                avg_mean = []
                avg_std = []
                for map_name in map_names:
                    ic(map_name)
                    
                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + step_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])

                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))

                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    value_dict[method_name].append(result)
                result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
                value_dict[method_name].append(result)

                avg_mean = []
                avg_std = []
                for map_name in large_map_names:
                    ic(map_name)
                    
                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + step_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])

                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))

                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    value_dict[method_name].append(result)
                result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
                value_dict[method_name].append(result)

            df = pandas.DataFrame(value_dict)
            print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=step_name + metric_name, label='tab:' + step_name + metric_name))

    # overlap
    
    if metric_name == "overlap":
        ic(metric_name)
        for ratio_name in ratio_names:
            value_dict = defaultdict(list)
            value_dict['Map ID'] = map_id_names.copy()
            value_dict['size'] = size_names.copy()
            for method_name in method_names:
                if method_name == "single_agent": break
                ic(method_name)
                avg_mean = []
                avg_std = []
                for map_name in map_names:
                    ic(map_name)

                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])

                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))

                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    value_dict[method_name].append(result)
                result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
                value_dict[method_name].append(result)

                avg_mean = []
                avg_std = []
                for map_name in large_map_names:
                    ic(map_name)

                    data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

                    df = pandas.read_csv(data_dir)
                    
                    key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
                    key_step = [n for n in key_cols if n == 'Step']
                    key_metric = [n for n in key_cols if n != 'Step']

                    step = np.array(df[key_step])
                    metric = np.array(df[key_metric])

                    metric_mean = np.mean(np.mean(metric, axis=0))
                    metric_std = np.std(np.mean(metric, axis=0))

                    avg_mean.append(metric_mean)
                    avg_std.append(metric_std)

                    result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                    value_dict[method_name].append(result)
                result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
                value_dict[method_name].append(result)

            df = pandas.DataFrame(value_dict)
            print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()) , multicolumn_format='c', caption= ratio_name + metric_name, label='tab:'+ ratio_name + metric_name))
    
    # ratio # balance #step
    value_dict = defaultdict(list)
    value_dict['Map ID'] = map_id_names.copy()
    value_dict['size'] = size_names.copy()
    if metric_name in ["ratio","balance","step"]:
        ic(metric_name)
        for method_name in method_names:
            if method_name == "single_agent" and metric_name == "balance": break
            ic(method_name)
            avg_mean = []
            avg_std = []
            for map_name in map_names:
                ic(map_name)
                data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step']

                step = np.array(df[key_step])
                metric = np.array(df[key_metric])

                metric_mean = np.nanmean(np.nanmean(metric, axis=0))
                metric_std = np.nanstd(np.nanmean(metric, axis=0))

                avg_mean.append(metric_mean)
                avg_std.append(metric_std)

                result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                value_dict[method_name].append(result)

            result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
            value_dict[method_name].append(result)

            avg_mean = []
            avg_std = []
            for map_name in large_map_names:
                ic(map_name)
                data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step']

                step = np.array(df[key_step])
                metric = np.array(df[key_metric])

                metric_mean = np.nanmean(np.nanmean(metric, axis=0))
                metric_std = np.nanstd(np.nanmean(metric, axis=0))

                avg_mean.append(metric_mean)
                avg_std.append(metric_std)

                result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                value_dict[method_name].append(result)

            result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
            value_dict[method_name].append(result)
        
        df = pandas.DataFrame(value_dict)
        print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=metric_name, label='tab:'+ metric_name))

        # ratio # balance #step
    
    value_dict = defaultdict(list)
    value_dict['Map ID'] = map_id_names.copy()
    value_dict['size'] = size_names.copy()
    if metric_name in ["success rate"]:
        ic(metric_name)
        for method_name in method_names:
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

                result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                value_dict[method_name].append(result)

            result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
            value_dict[method_name].append(result)

            avg_mean = []
            avg_std = []
            for map_name in large_map_names:
                ic(map_name)
                data_dir =  save_dir + map_name + '/' + method_name + "/ratio/ratio.csv"

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step']

                step = np.array(df[key_step])
                metric = np.array(df[key_metric])
                metric = metric > 0.90

                metric_mean = np.nanmean(np.nanmean(metric, axis=0))
                metric_std = np.nanstd(np.nanmean(metric, axis=0))

                avg_mean.append(metric_mean)
                avg_std.append(metric_std)

                result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                value_dict[method_name].append(result)

            result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
            value_dict[method_name].append(result)
        
        df = pandas.DataFrame(value_dict)
        print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=metric_name, label='tab:'+ metric_name))