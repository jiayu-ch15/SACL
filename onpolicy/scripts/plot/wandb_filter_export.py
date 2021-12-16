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
project_dir = './football/'
wandb_name = "football"
project_name = "tune_hyperparameters"

api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)

export_names = ['test','test1']
varying_names = [100, 50]

# Hyperparameters to filter.
filter_configs = {
    'share_policy': True,
    'algorithm_name': 'rmappo',
    'num_mini_batch': 2,
    'ppo_epoch': 15
}
# Keywords for filter name.
keywords = [
    'shared',
]
# Metrics you want to export.
metrics = [
    'expected_goal',
    'goal'
]

for varying_name, export_name in zip(varying_names, export_names):
    filter_configs.update({'n_rollout_threads': varying_name}) # here is an example

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
        save_dir = project_dir + export_name + "/" 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        merged_panels[metric].to_csv(save_dir + metric + ".csv")

