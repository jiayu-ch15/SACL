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

scenario_names=['academy_3_vs_1_with_keeper',\
                'academy_counterattack_easy',\
                'academy_counterattack_hard',\
                'academy_corner',\
                'academy_pass_and_shoot_with_keeper',\
                'academy_run_pass_and_shoot_with_keeper']

rollout_threads = ['5','10','25','50','100']

tags = ['final_mappo_rollout' + rt for rt in rollout_threads] + \
['final_mappo',
'final_mappo_sparse',
'final_mappo_denserew',
'final_mappo_separated_sparserew',
'final_mappo_separated_sharedenserew',
'final_mappo_separated_denserew',
'final_qmix',
'final_qmix_sparse',
'final_cds_qmix',
'final_cds_qplex',
'final_cds_qmix_denserew',
'final_cds_qplex_denserew'
]

filter_configs = {
}

# Metrics you want to export.
metrics = {'final_mappo_rollout'+rt:['expected_goal'] for rt in rollout_threads}
metrics.update(
    {
    'final_mappo': ['expected_goal'],
    'final_mappo_sparse': ['expected_goal'],
    'final_mappo_denserew': ['expected_goal'],
    'final_mappo_separated_sparserew': ['expected_goal'],
    'final_mappo_separated_sharedenserew': ['expected_goal'],
    'final_mappo_separated_denserew': ['expected_goal'],
    'final_qmix': ['expected_win_rate'],
    'final_qmix_sparse': ['expected_win_rate'],
    'final_cds_qmix': ['test_score_reward_mean'],
    'final_cds_qplex': ['test_score_reward_mean'],
    'final_cds_qmix_denserew': ['test_score_reward_mean'],
    'final_cds_qplex_denserew': ['test_score_reward_mean']
    }
)

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for scenario_name in scenario_names:
    filter_configs.update({'scenario_name': scenario_name})
    print(scenario_name)
    
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
            save_dir = project_dir + scenario_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")

###########################################################################################
###########################################################################################
###########################################################################################


scenario_names=['academy_3_vs_1_with_keeper',\
                'academy_counterattack_easy']

tags = ['final_mappo_rollout100_length1000']

filter_configs = {
    'share_policy': True,
    'share_reward': True,
    'algorithm_name': 'rmappo'
}

# Metrics you want to export.
metrics = [
    'expected_goal'
]

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for scenario_name in scenario_names:
    filter_configs.update({'scenario_name': scenario_name})
    
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
            save_dir = project_dir + scenario_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")

###########################################################################################
###########################################################################################
###########################################################################################

scenario_names=['academy_corner']

rollout_threads = ['50']

tags = ['final_mappo_rollout' + rt for rt in rollout_threads] + ['final_mappo','final_mappo_sparse',
'final_mappo_denserew',
'final_mappo_separated_sparserew',
'final_mappo_separated_sharedenserew',
'final_mappo_separated_denserew']

filter_configs = {
}

# Metrics you want to export.
metrics = [
    'eval_expected_win_rate'
]

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for scenario_name in scenario_names:
    filter_configs.update({'scenario_name': scenario_name})
    
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
            save_dir = project_dir + scenario_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")

scenario_names=['academy_run_pass_and_shoot_with_keeper','academy_counterattack_hard']

tags = ['final_mappo_sparse']

filter_configs = {
}

# Metrics you want to export.
metrics = [
    'eval_expected_win_rate'
]

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for scenario_name in scenario_names:
    filter_configs.update({'scenario_name': scenario_name})
    
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
            save_dir = project_dir + scenario_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")


scenario_names=['academy_counterattack_hard']

tags = ['final_mappo_sparse','final_mappo_denserew',
'final_mappo_separated_sparserew',
'final_mappo_separated_sharedenserew',
'final_mappo_separated_denserew']

filter_configs = {
}

# Metrics you want to export.
metrics = [
    'eval_expected_win_rate'
]

#########################################################################
#########################################################################

################################export data##############################
#########################################################################
api = wandb.Api()
runs = api.runs(wandb_name + "/"+ project_name)


for scenario_name in scenario_names:
    filter_configs.update({'scenario_name': scenario_name})
    
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
            save_dir = project_dir + scenario_name + "/" +  tag + "/" 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            merged_panels[metric].to_csv(save_dir + metric + ".csv")
