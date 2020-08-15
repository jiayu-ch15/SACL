#!/usr/bin/env python

import copy
import glob
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from envs import StarCraft2Env, get_map_params

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import shutil
import numpy as np

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "StarCraft2":
                env = StarCraft2Env(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            # np.random.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_test_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "StarCraft2":
                env = StarCraft2Env(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            # np.random.seed(args.seed + rank * 1000)
            return env
        return init_env
    return SubprocVecEnv([get_env_fn(0)])

def main():
    args = get_config()
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    run_dir = Path(args.model_dir)/ ("run" + str(args.seed)) / 'eval'
    if os.path.exists(run_dir): 
        shutil.rmtree(run_dir)
        os.mkdir(run_dir)
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir))
    
    #Policy network    	       
    actor_critic = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/models/agent_model.pt")['model'].to(device)

    # env
    eval_env = make_test_env(args)
    num_agents = get_map_params(args.map_name)["n_agents"]

    for eval_episode in range(args.eval_episodes):
        print(eval_episode)
        eval_obs, eval_available_actions = eval_env.reset()
        eval_share_obs = eval_obs.reshape(1, -1)
        eval_recurrent_hidden_states = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
        eval_recurrent_hidden_states_critic = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
        eval_masks = np.ones((1,num_agents,1)).astype(np.float32)
        
        while True:
            eval_actions = []
            for i in range(num_agents):
                _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(i,
                    torch.tensor(eval_share_obs), 
                    torch.tensor(eval_obs[:,i]), 
                    torch.tensor(eval_recurrent_hidden_states[:,i]), 
                    torch.tensor(eval_recurrent_hidden_states_critic[:,i]),
                    torch.tensor(eval_masks[:,i]),
                    torch.tensor(eval_available_actions[:,i,:]))

                eval_actions.append(action.detach().cpu().numpy())
                eval_recurrent_hidden_states[:,i] = recurrent_hidden_states.detach().cpu().numpy()
                eval_recurrent_hidden_states_critic[:,i] = recurrent_hidden_states_critic.detach().cpu().numpy()

            # rearrange action          
            eval_actions_env = []
            for k in range(num_agents):
                one_hot_action = np.zeros(eval_env.action_space[0].n)
                one_hot_action[eval_actions[k][0]] = 1.0
                eval_actions_env.append(one_hot_action)
                    
            # Obser reward and next obs
            print(eval_actions_env)
            eval_obs, eval_reward, eval_done, eval_infos, eval_available_actions = eval_env.step([eval_actions_env])
            eval_share_obs = eval_obs.reshape(1, -1)
            # If done then clean the history of observations.
            # insert data in buffer
            if eval_done[0]:
                break
                
    #logger.add_scalars('eval_win_rate',{'eval_win_rate': eval_battles_won/args.eval_episodes},total_num_steps)
    print(eval_infos[0][0]['battles_won'], eval_infos[0][0]['battles_game'])
    print(eval_infos[0][0]['battles_won']/eval_infos[0][0]['battles_game'])
                
    #logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    #logger.close()
    #eval_env.close()
if __name__ == "__main__":
    main()
