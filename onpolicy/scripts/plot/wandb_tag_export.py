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

##############################change config##############################
#########################################################################
project_dir = './football_tag/'
wandb_name = "football"
project_name = "tune_hyperparameters"

tags = [
    'test',
    'test1'
]

# Metrics you want to export.
metrics = [
    'expected_goal',
    'goal'
]

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)

runs_taged_dict = defaultdict(list)
for _, run in enumerate(runs):
    for tag in tags:
        if tag in run.Tags:
            runs_taged_dict[tag].append(run)
            print(run.name)

for tag, runs_taged in runs_taged_dict.items():
    panels = {metric: [] for metric in metrics}
    for run in runs_taged:
        # `history` is a pd.Dataframe instance.
        # `samples`: should be large to export all data once.
        history = run.history(keys=metrics, samples=1000000)
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
        save_dir = project_dir + tag + "/" 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        merged_panels[metric].to_csv(save_dir + metric + ".csv")