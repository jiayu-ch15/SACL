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

from envs import MPEEnv

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import shutil
import numpy as np
import imageio

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
    gifs_dir = run_dir / 'gifs'	
    os.makedirs(str(gifs_dir))
    
    num_agents = args.num_agents
    #Policy network 
    if args.share_policy:   	       
        actor_critic = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/models/agent_model.pt")['model'].to(device)
    else:
        actor_critic = []
        for agent_id in range(num_agents):
            ac = torch.load(str(args.model_dir) + 'run' + str(args.seed) + "/models/agent" + str(agent_id) + "_model.pt")['model'].to(device)
            actor_critic.append(ac)
   
    all_frames = []
    for eval_episode in range(args.eval_episodes):
        print(eval_episode)
        eval_env = MPEEnv(args)
        if args.save_gifs:
            image = eval_env.render('rgb_array', close=False)[0]
            all_frames.append(image)
        
        eval_obs, _ = eval_env.reset()
        eval_obs = np.array(eval_obs)        
        eval_share_obs = eval_obs.reshape(1, -1)
        eval_recurrent_hidden_states = np.zeros((num_agents,args.hidden_size)).astype(np.float32)
        eval_recurrent_hidden_states_critic = np.zeros((num_agents,args.hidden_size)).astype(np.float32)
        eval_masks = np.ones((num_agents,1)).astype(np.float32)
        
        for step in range(args.episode_length): 
            calc_start = time.time()              
            eval_actions = []            
            for agent_id in range(num_agents):
                if args.share_policy:
                    actor_critic.eval()
                    _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                        torch.FloatTensor(eval_share_obs), 
                        torch.FloatTensor(eval_obs[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states_critic[agent_id]),
                        torch.FloatTensor(eval_masks[agent_id]),
                        None,
                        deterministic=True)
                else:
                    actor_critic[agent_id].eval()
                    _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                        torch.FloatTensor(eval_share_obs), 
                        torch.FloatTensor(eval_obs[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states[agent_id]), 
                        torch.FloatTensor(eval_recurrent_hidden_states_critic[agent_id]),
                        torch.FloatTensor(eval_masks[agent_id]),
                        None,
                        deterministic=True)
    
                eval_actions.append(action.detach().cpu().numpy())
                eval_recurrent_hidden_states[agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                eval_recurrent_hidden_states_critic[agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()
    
            # rearrange action           
            eval_actions_env = []
            for agent_id in range(num_agents):
                one_hot_action = np.zeros(eval_env.action_space[0].n)
                one_hot_action[eval_actions[agent_id][0]] = 1
                eval_actions_env.append(one_hot_action)
                    
            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos, _ = eval_env.step(eval_actions_env)
            eval_obs = np.array(eval_obs)
            eval_share_obs = eval_obs.reshape(1, -1)
            
            if args.save_gifs:
                image = eval_env.render('rgb_array', close=False)[0]
                all_frames.append(image)
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < args.ifi:
                    time.sleep(ifi - elapsed)
                                                
    if args.save_gifs:
        imageio.mimsave(str(gifs_dir / args.scenario_name) + '.gif',
                    all_frames, duration=args.ifi)          
    eval_env.close()
if __name__ == "__main__":
    main()
