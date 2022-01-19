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

scenario_names=['academy_3_vs_1_with_keeper',\
                'academy_counterattack_easy',\
                'academy_counterattack_hard',\
                'academy_corner',\
                'academy_pass_and_shoot_with_keeper',\
                'academy_run_pass_and_shoot_with_keeper']
label_names=['3v1',\
            'CA(easy)',\
            'CA(hard)',\
            'Corner',\
            'PS',\
            'RPS']


rollout_threads = ['5','10','25','50','100']

method_names = ['final_mappo','final_qmix','final_cds_qmix_denserew','final_cds_qplex_denserew','final_cds_qmix','final_cds_qplex','final_tikick']

algo_names = ['MAPPO','QMix','CDS(QMix-d)','CDS(QPlex-d)','CDS(QMix)','CDS(QPlex)','TiKick']

metrics = {'final_mappo_rollout'+rt:['expected_goal'] for rt in rollout_threads}
metrics.update(
    {
    'final_qmix': ['expected_win_rate'],
    'final_qmix_sparse': ['expected_win_rate'],
    'final_cds_qmix': ['test_score_reward_mean'],
    'final_cds_qplex': ['test_score_reward_mean'],
    'final_cds_qmix_denserew': ['test_score_reward_mean'],
    'final_cds_qplex_denserew': ['test_score_reward_mean'],
    'final_tikick': ['eval_win_rate'],
    'final_mappo': ['expected_goal'],
    'final_mappo_sparse': ['expected_goal'],
    'final_mappo_denserew': ['expected_goal'],
    'final_mappo_separated_sparserew': ['expected_goal'],
    'final_mappo_separated_sharedenserew': ['expected_goal'],
    'final_mappo_separated_denserew': ['expected_goal'],
    }
)
project_dir = './football/'
max_step=50e6

value_dict = defaultdict(list)
for scenario_name,label_name in zip(scenario_names,label_names):
    value_dict['scenario_name'].append(label_name)
    for method_name, algo_name in zip(method_names,algo_names):

        if method_name == "final_tikick" and scenario_name not in ['academy_3_vs_1_with_keeper',\
                'academy_counterattack_hard',\
                'academy_corner',\
                'academy_run_pass_and_shoot_with_keeper']:
            value_dict[algo_name].append('/')
        else:
            if "mappo" in method_name and scenario_name == "academy_corner":
                metric_names = ['eval_expected_win_rate']
            else:
                metric_names = metrics[method_name]

            for metric_name in metric_names:
                data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                df = df.loc[df['Step'] <= max_step]
                if method_name == "final_qmix":
                    if scenario_name == 'academy_counterattack_hard':
                        df = df.loc[df['Step'] <= 25e6]
                    if scenario_name == 'academy_corner':
                        df = df.loc[df['Step'] <= 20e6]

                if method_name == "final_cds_qmix_denserew" or method_name == "final_cds_qplex_denserew":
                    df = df.loc[df['Step'] <= 15e6]
                    if scenario_name == 'academy_corner':
                        df = df.loc[df['Step'] <= 12e6]

                if method_name == "final_cds_qmix":
                    df = df.loc[df['Step'] <= 19e6]
                    if scenario_name == 'academy_corner':
                        df = df.loc[df['Step'] <= 13e6]
                
                if method_name == "final_cds_qplex":
                    df = df.loc[df['Step'] <= 19e6]
                    if scenario_name == 'academy_counterattack_hard':
                        df = df.loc[df['Step'] <= 15e6]
                    if scenario_name == 'academy_3_vs_1_with_keeper':
                        df = df.loc[df['Step'] <= 17e6]
                    if scenario_name == 'academy_corner':
                        df = df.loc[df['Step'] <= 12e6]

                if method_name == "final_mappo_sparse":
                    if scenario_name == 'academy_run_pass_and_shoot_with_keeper':
                        df = df.loc[df['Step'] <= 14e6]
                    if scenario_name == 'academy_counterattack_hard':
                        df = df.loc[df['Step'] <= 40e6]
                    if scenario_name == 'academy_corner':
                        df = df.loc[df['Step'] <= 33e6]
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

                step = np.array(df[key_step])
                all_metric = np.array(df[key_metric])

                metric = all_metric[-10:,:] * 100
                metric_mean = np.mean(np.mean(metric, axis=1))
                metric_std = np.std(np.mean(metric, axis=1))

                result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                value_dict[algo_name].append(result)
        
df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption="football", label='tab:football'))

method_names = ['final_mappo',
'final_mappo_denserew',
'final_mappo_sparse', 
'final_mappo_separated_sharedenserew',
'final_mappo_separated_denserew',
'final_mappo_separated_sparserew']

algo_names = ['MAPPO(share-dense)','MAPPO(dense)','MAPPO(share-sparse)','SEP-MAPPO(share-dense)','SEP-MAPPO(dense)','SEP-MAPPO(share-sparse)']

value_dict = defaultdict(list)
for scenario_name,label_name in zip(scenario_names,label_names):
    value_dict['scenario_name'].append(label_name)
    for method_name, algo_name in zip(method_names,algo_names):
        
        if "mappo_" in method_name and (scenario_name == "academy_corner" or scenario_name == 'academy_counterattack_hard') or \
        "mappo_sparse" in method_name and scenario_name == 'academy_run_pass_and_shoot_with_keeper' :
            metric_names = ['eval_expected_win_rate']
        else:
            metric_names = metrics[method_name]

        for metric_name in metric_names:
            data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)
            df = df.loc[df['Step'] <= max_step]
            if method_name == "final_mappo_sparse":
                if scenario_name == 'academy_run_pass_and_shoot_with_keeper':
                    df = df.loc[df['Step'] <= 14e6]
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 40e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 33e6]
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            metric = all_metric[-10:,:] * 100
            metric_mean = np.mean(np.mean(metric, axis=1))
            metric_std = np.std(np.mean(metric, axis=1))

            result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
            value_dict[algo_name].append(result)
    
df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption="mappo (share policy or not, share reward or not, dense reward or sparse reward)", label='tab:football-mappo'))

# sparse reward

method_names = ['final_mappo_sparse','final_qmix_sparse','final_cds_qmix', 'final_cds_qplex']

algo_names = ['MAPPO','QMix','CDS(QMix)','CDS(QPlex)']

value_dict = defaultdict(list)
for scenario_name,label_name in zip(scenario_names,label_names):
    value_dict['scenario_name'].append(label_name)
    for method_name, algo_name in zip(method_names,algo_names):

        if "mappo" in method_name and (scenario_name == "academy_corner" or scenario_name == 'academy_run_pass_and_shoot_with_keeper' or scenario_name == 'academy_counterattack_hard'):
            metric_names = ['eval_expected_win_rate']
        else:
            metric_names = metrics[method_name]

        for metric_name in metric_names:
            data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

            df = pandas.read_csv(data_dir)
            df = df.loc[df['Step'] <= max_step]
            if method_name == "final_cds_qmix" or method_name == "final_cds_qplex":
                df = df.loc[df['Step'] <= 19e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 14e6]


            if method_name == "final_mappo_sparse":
                if scenario_name == 'academy_run_pass_and_shoot_with_keeper':
                    df = df.loc[df['Step'] <= 14e6]
                if scenario_name == 'academy_counterattack_hard':
                    df = df.loc[df['Step'] <= 40e6]
                if scenario_name == 'academy_corner':
                    df = df.loc[df['Step'] <= 33e6]
            
            key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
            
            key_step = [n for n in key_cols if n == 'Step']
            key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

            step = np.array(df[key_step])
            all_metric = np.array(df[key_metric])

            metric = all_metric[-10:,:] * 100
            metric_mean = np.mean(np.mean(metric, axis=1))
            metric_std = np.std(np.mean(metric, axis=1))

            result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
            value_dict[algo_name].append(result)
        
df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption="football results with sparse reward", label='tab:football-sparse'))
