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

map_names = ['16','20','21']#,'22','36','43','48','49','61']
method_names = ['apf', 'rrt', 'nearest','utility', 'global_stack']
metric_names = ['auc','overlap','ratio','step','balance'] #,'agent_auc']
step_names = ['100step','120step','150step','180step','200step','250step']
ratio_names = ['30ratio','50ratio','70ratio','90ratio']

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
            value_dict['Map ID'] = np.array(map_names)
            for method_name in method_names:
                ic(method_name)
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

                    result = str(format(metric_mean, '.1f')) + "(" + str(format(metric_std, '.1f')) + ")"
                    value_dict[method_name].append(result)
        
            df = pandas.DataFrame(value_dict)
            print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=step_name + metric_name, label='tab:' + step_name + metric_name))

    # overlap
    
    if metric_name == "overlap":
        ic(metric_name)
        for ratio_name in ratio_names:
            value_dict = defaultdict(list)
            value_dict['Map ID'] = np.array(map_names)
            for method_name in method_names:
                ic(method_name)
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

                    result = str(format(metric_mean, '.1f')) + "(" + str(format(metric_std, '.1f')) + ")"
                    value_dict[method_name].append(result)
        
            df = pandas.DataFrame(value_dict)
            print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()) , multicolumn_format='c', caption= ratio_name + metric_name, label='tab:'+ ratio_name + metric_name))
    
    # ratio # balance #step
    value_dict = defaultdict(list)
    value_dict['Map ID'] = np.array(map_names)
    if metric_name in ["ratio","balance","step"]:
        ic(metric_name)
        for method_name in method_names:
            ic(method_name)

            for map_name in map_names:
                ic(map_name)
                data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step']

                step = np.array(df[key_step])
                metric = np.array(df[key_metric])

                metric_mean = np.mean(np.mean(metric, axis=0))
                metric_std = np.std(np.mean(metric, axis=0))

                result = str(format(metric_mean, '.1f')) + "(" + str(format(metric_std, '.1f')) + ")"
                value_dict[method_name].append(result)

        df = pandas.DataFrame(value_dict)
        print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=metric_name, label='tab:'+ metric_name))