import wandb
import pprint
import re
import pandas as pd
from icecream import ic
import json
import numpy as np
import sys
import os
from collections import defaultdict

def match_config(pattern, raw):
    try:
        for key, value in pattern.items():
            if raw[key] != value:
                return False
    except KeyError:
        return False
    return True


##############################change config##############################
#########################################################################
project_dir = './habitat_1agent/'
wandb_name = "mapping"
project_name = "Habitat"

api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)

export_names = ['16','20','21','22','36','43','48','49','61','27']#'26','40',,
varying_names = [16,20,21,22,36,43,48,49,61,27]#26,40,27
method_name = 'global_stack'
# Hyperparameters to filter.
filter_configs = {
    'use_eval': True,
    'algorithm_name': 'mappo',
    #'change_down_agents': True ,
    #'change_down_agents': False,
    'num_agents': 1,
    #'second_stage_agents':2
}
# Keywords for filter name.
keywords = [
    'save'
]
# Metrics you want to export.
metric_dir = {'sum_repeat_ratio':'repeat'}
#{'sum_merge_explored_ratio':'ratio', 'merge_explored_ratio_step':'step','180step_merge_auc':'auc','merge_overlap_ratio':'overlap'}
metric_name = {'sum_repeat_ratio':'repeat_ratio'}
#{'sum_merge_explored_ratio':'ratio', 'merge_explored_ratio_step':'step','300step_merge_auc':'300step'}#'merge_overlap_ratio':'90ratio',
metrics = ['sum_repeat_ratio']
#'merge_overlap_ratio','sum_merge_explored_ratio','merge_explored_ratio_step','300step_merge_auc'

for varying_name, export_name in zip(varying_names, export_names):
    filter_configs.update({'scene_id': varying_name}) # here is an example

    ################################export data##############################
    #########################################################################
    
    # Regular expression is preferred.
    pattern = ''.join([f"(?=.*{keyword})" for keyword in keywords])  # pattern for match all of the keywords 
    reg_obj = re.compile(pattern)
    panels = {metric: [] for metric in metrics}

    runs_filterd = []
    for _, run in enumerate(runs):
        if match_config(filter_configs, run.config) and (reg_obj.search(run.name) is not None):
            runs_filterd.append(run)
            print(run.name)

    """
    1. It is recommended to parallelize using multi-process | multi-thread for runs.
    2. If the number of steps (samples) does not match between two metrics, there can be nan in responsed `history`
    """

    for run in runs_filterd:
        # `history` is a pd.Dataframe instance.
        # `samples`: should be large to export all data once.
        history = run.history(keys=metrics, samples=1000000)
        for metric, lst in panels.items():
            lst.append(history[metric])

    # Merge panels by metrics.
    merged_panels = {
        metric: pd.DataFrame({
            runs_filterd[i].name: data
            for i, data in enumerate(data_list)
        })
        for metric, data_list in panels.items()
    }

    # Add 'step' column to all panels
    step = history['_step']
    for df in merged_panels.values():
        df.insert(0, "Step", step)

    # save csv
    for metric, data_list in merged_panels.items():
        save_dir = project_dir + export_name + "/" + method_name + "/"+ metric_dir[metric]+ "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        merged_panels[metric].to_csv(save_dir + metric_name[metric] + ".csv")

