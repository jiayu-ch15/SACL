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
project_dir = './smac/'
wandb_name = "zoeyuchao"
project_name = "StarCraft2"

map_names=['MMM2', '6h_vs_8z']

rollout_threads = ['1','2','4','8','16']

tags = ['final_mappo_rollout' + rt for rt in rollout_threads]

filter_configs = {
}

# Metrics you want to export.
metrics = {'final_mappo_rollout'+rt:['eval_win_rate'] for rt in rollout_threads}


#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for map_name in map_names:
    filter_configs.update({'map_name': map_name})
    
    runs_taged_dict = defaultdict(list)
    for _, run in enumerate(runs):
        if match_config(filter_configs, run.config):
            for tag in tags:
                if tag in run.Tags:
                    runs_taged_dict[tag].append(run)
                    print(run.name)

    for tag, runs_taged in runs_taged_dict.items():
        panels = {metric: [] for metric in metrics[tag]}
        for run in runs_taged:
            # `history` is a pd.Dataframe instance.
            # `samples`: should be large to export all data once.
            history = run.history(keys=metrics[tag], samples=1000000000)
            for metric, lst in panels.items():
                lst.append(history[metric])

        # Merge panels by metrics.
        merged_panels = {
            metric: pd.DataFrame({
                runs_taged[i].name: data
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
            save_dir = project_dir + map_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")

###########################################################################################
###########################################################################################
###########################################################################################

map_names=['MMM2', '6h_vs_8z']

rollout_threads = ['1','2','4','8','16']

tags = ['final_mappo_rollout16_length2000']

filter_configs = {
}

# Metrics you want to export.
metrics = [
    'eval_win_rate'
]

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for map_name in map_names:
    filter_configs.update({'map_name': map_name})
    
    runs_taged_dict = defaultdict(list)
    for _, run in enumerate(runs):
        if match_config(filter_configs, run.config):
            for tag in tags:
                if tag in run.Tags:
                    runs_taged_dict[tag].append(run)
                    print(run.name)

    for tag, runs_taged in runs_taged_dict.items():
        panels = {metric: [] for metric in metrics}
        for run in runs_taged:
            # `history` is a pd.Dataframe instance.
            # `samples`: should be large to export all data once.
            history = run.history(keys=metrics, samples=1000000000)
            for metric, lst in panels.items():
                lst.append(history[metric])

        # Merge panels by metrics.
        merged_panels = {
            metric: pd.DataFrame({
                runs_taged[i].name: data
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
            save_dir = project_dir + map_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")