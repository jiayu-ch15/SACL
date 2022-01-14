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

method_names = ['final_mappo','final_qmix','final_cds_qmix', 'final_cds_qplex','final_cds_qmix_denserew','final_cds_qplex_denserew','final_tikick']
algo_names = ['MAPPO','QMix','CDS(QMix)','CDS(QPlex)','CDS(QMix-d)','CDS(QPlex-d)','TiKick']

metrics = {'final_mappo_rollout'+rt:['expected_goal'] for rt in rollout_threads}
metrics.update(
    {
    'final_mappo': ['expected_goal'],
    'final_qmix': ['expected_win_rate'],
    'final_cds_qmix': ['test_score_reward_mean'],
    'final_cds_qplex': ['test_score_reward_mean'],
    'final_cds_qmix_denserew': ['test_score_reward_mean'],
    'final_cds_qplex_denserew': ['test_score_reward_mean'],
    'final_tikick': ['eval_win_rate']
    }
)
project_dir = './football/'

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
            metric_names = metrics[method_name]

            for metric_name in metric_names:
                data_dir = project_dir + scenario_name + '/' + method_name + "/" + metric_name + '.csv'

                df = pandas.read_csv(data_dir)
                
                key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
                key_step = [n for n in key_cols if n == 'Step']
                key_metric = [n for n in key_cols if n != 'Step' and n != 'Unnamed: 0']

                step = np.array(df[key_step])
                all_metric = np.array(df[key_metric])

                metric = all_metric[-10:,:] * 100
                metric_mean = np.nanmean(np.nanmean(metric, axis=1))
                metric_std = np.std(np.nanmean(metric, axis=1))

                result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
                value_dict[algo_name].append(result)
        
df = pandas.DataFrame(value_dict)
print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption="football", label='tab:football'))







# for metric_name in metric_names:
#     # auc # agent_auc
#     if metric_name in ["auc", "agent_auc"]:
#         ic(metric_name)
#         for step_name in step_names:
#             ic(step_name + metric_name)
            

#                 avg_mean = []
#                 avg_std = []
#                 for map_name in large_map_names:
#                     ic(map_name)
                    
#                     data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + step_name + '.csv'

#                     df = pandas.read_csv(data_dir)
                    
#                     key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
#                     key_step = [n for n in key_cols if n == 'Step']
#                     key_metric = [n for n in key_cols if n != 'Step']

#                     step = np.array(df[key_step])
#                     metric = np.array(df[key_metric])

#                     metric_mean = np.mean(np.mean(metric, axis=0))
#                     metric_std = np.std(np.mean(metric, axis=0))

#                     avg_mean.append(metric_mean)
#                     avg_std.append(metric_std)

#                     result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                     value_dict[method_name].append(result)
#                 result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
#                 value_dict[method_name].append(result)

#             df = pandas.DataFrame(value_dict)
#             print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=step_name + metric_name, label='tab:' + step_name + metric_name))

#     # overlap
    
#     if metric_name == "overlap":
#         ic(metric_name)
#         for ratio_name in ratio_names:
#             value_dict = defaultdict(list)
#             value_dict['Map ID'] = map_id_names.copy()
#             value_dict['size'] = size_names.copy()
#             for method_name in method_names:
#                 if method_name == "single_agent": break
#                 ic(method_name)
#                 avg_mean = []
#                 avg_std = []
#                 for map_name in map_names:
#                     ic(map_name)

#                     data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

#                     df = pandas.read_csv(data_dir)
                    
#                     key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
#                     key_step = [n for n in key_cols if n == 'Step']
#                     key_metric = [n for n in key_cols if n != 'Step']

#                     step = np.array(df[key_step])
#                     metric = np.array(df[key_metric])

#                     metric_mean = np.mean(np.mean(metric, axis=0))
#                     metric_std = np.std(np.mean(metric, axis=0))

#                     avg_mean.append(metric_mean)
#                     avg_std.append(metric_std)

#                     result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                     value_dict[method_name].append(result)
#                 result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
#                 value_dict[method_name].append(result)

#                 avg_mean = []
#                 avg_std = []
#                 for map_name in large_map_names:
#                     ic(map_name)

#                     data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + ratio_name + '.csv'

#                     df = pandas.read_csv(data_dir)
                    
#                     key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                    
#                     key_step = [n for n in key_cols if n == 'Step']
#                     key_metric = [n for n in key_cols if n != 'Step']

#                     step = np.array(df[key_step])
#                     metric = np.array(df[key_metric])

#                     metric_mean = np.mean(np.mean(metric, axis=0))
#                     metric_std = np.std(np.mean(metric, axis=0))

#                     avg_mean.append(metric_mean)
#                     avg_std.append(metric_std)

#                     result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                     value_dict[method_name].append(result)
#                 result = str(format(np.mean(avg_mean), '.2f')) + "(" + str(format(np.mean(avg_std), '.2f')) + ")"
#                 value_dict[method_name].append(result)

#             df = pandas.DataFrame(value_dict)
#             print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()) , multicolumn_format='c', caption= ratio_name + metric_name, label='tab:'+ ratio_name + metric_name))
    
#     # ratio # balance #step
#     value_dict = defaultdict(list)
#     value_dict['Map ID'] = map_id_names.copy()
#     value_dict['size'] = size_names.copy()
#     if metric_name in ["ratio","balance","step"]:
#         ic(metric_name)
#         for method_name in method_names:
#             if method_name == "single_agent" and metric_name == "balance": break
#             ic(method_name)
#             avg_mean = []
#             avg_std = []
#             for map_name in map_names:
#                 ic(map_name)
#                 data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

#                 df = pandas.read_csv(data_dir)
                
#                 key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
#                 key_step = [n for n in key_cols if n == 'Step']
#                 key_metric = [n for n in key_cols if n != 'Step']

#                 step = np.array(df[key_step])
#                 metric = np.array(df[key_metric])

#                 metric_mean = np.nanmean(np.nanmean(metric, axis=0))
#                 metric_std = np.nanstd(np.nanmean(metric, axis=0))

#                 avg_mean.append(metric_mean)
#                 avg_std.append(metric_std)

#                 result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                 value_dict[method_name].append(result)

#             result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
#             value_dict[method_name].append(result)

#             avg_mean = []
#             avg_std = []
#             for map_name in large_map_names:
#                 ic(map_name)
#                 data_dir =  save_dir + map_name + '/' + method_name + "/" + metric_name + '/' + metric_name + '.csv'

#                 df = pandas.read_csv(data_dir)
                
#                 key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
#                 key_step = [n for n in key_cols if n == 'Step']
#                 key_metric = [n for n in key_cols if n != 'Step']

#                 step = np.array(df[key_step])
#                 metric = np.array(df[key_metric])

#                 metric_mean = np.nanmean(np.nanmean(metric, axis=0))
#                 metric_std = np.nanstd(np.nanmean(metric, axis=0))

#                 avg_mean.append(metric_mean)
#                 avg_std.append(metric_std)

#                 result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                 value_dict[method_name].append(result)

#             result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
#             value_dict[method_name].append(result)
        
#         df = pandas.DataFrame(value_dict)
#         print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=metric_name, label='tab:'+ metric_name))

#         # ratio # balance #step
    
#     value_dict = defaultdict(list)
#     value_dict['Map ID'] = map_id_names.copy()
#     value_dict['size'] = size_names.copy()
#     if metric_name in ["success rate"]:
#         ic(metric_name)
#         for method_name in method_names:
#             ic(method_name)
#             avg_mean = []
#             avg_std = []
#             for map_name in map_names:
#                 ic(map_name)
#                 data_dir =  save_dir + map_name + '/' + method_name + "/ratio/ratio.csv"

#                 df = pandas.read_csv(data_dir)
                
#                 key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
#                 key_step = [n for n in key_cols if n == 'Step']
#                 key_metric = [n for n in key_cols if n != 'Step']

#                 step = np.array(df[key_step])
#                 metric = np.array(df[key_metric])
#                 metric = metric > 0.9

#                 metric_mean = np.nanmean(np.nanmean(metric, axis=0))
#                 metric_std = np.nanstd(np.nanmean(metric, axis=0))

#                 avg_mean.append(metric_mean)
#                 avg_std.append(metric_std)

#                 result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                 value_dict[method_name].append(result)

#             result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
#             value_dict[method_name].append(result)

#             avg_mean = []
#             avg_std = []
#             for map_name in large_map_names:
#                 ic(map_name)
#                 data_dir =  save_dir + map_name + '/' + method_name + "/ratio/ratio.csv"

#                 df = pandas.read_csv(data_dir)
                
#                 key_cols = [c for c in df.columns if 'MIN' not in c and 'MAX' not in c]
                
#                 key_step = [n for n in key_cols if n == 'Step']
#                 key_metric = [n for n in key_cols if n != 'Step']

#                 step = np.array(df[key_step])
#                 metric = np.array(df[key_metric])
#                 metric = metric > 0.90

#                 metric_mean = np.nanmean(np.nanmean(metric, axis=0))
#                 metric_std = np.nanstd(np.nanmean(metric, axis=0))

#                 avg_mean.append(metric_mean)
#                 avg_std.append(metric_std)

#                 result = str(format(metric_mean, '.2f')) + "(" + str(format(metric_std, '.2f')) + ")"
#                 value_dict[method_name].append(result)

#             result = str(format(np.nanmean(avg_mean), '.2f')) + "(" + str(format(np.nanmean(avg_std), '.2f')) + ")"
#             value_dict[method_name].append(result)
        
#         df = pandas.DataFrame(value_dict)
#         print(df.to_latex(index=False, column_format = 'c'*len(value_dict.keys()), multicolumn_format='c', caption=metric_name, label='tab:'+ metric_name))