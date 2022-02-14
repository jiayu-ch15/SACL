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
project_dir = './hanabi/'
wandb_name = "samji2000"
project_name = "Hanabi"

agent_nums = ['2','3','4','5']

tags = [f'final_{n}agent' for n in agent_nums] 

filter_configs = {
    # 'Job Type': 'trainer_worker'
}

# Keywords for filter name.
keywords = [
    'trainer'
]

# Metrics you want to export.
metrics = ['score']


#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for agent_num in agent_nums:
    # filter_configs.update({'agent_num': agent_num})
    pattern = ''.join([f"(?=.*{keyword})" for keyword in keywords])  # pattern for match all of the keywords 
    reg_obj = re.compile(pattern)
    runs_taged_dict = defaultdict(list)
    for _, run in enumerate(runs):
        if reg_obj.search(run.name) is not None:
            for tag in tags:
                if tag in run.Tags:
                    runs_taged_dict[tag].append(run)

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
            save_dir = project_dir + agent_num + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")