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
from algorithm.ppo import PPO
from algorithm.model import Policy

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
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

def main():
    args = get_config()

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(1)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    # path
    model_dir = Path('./results') / args.env_name / args.map_name / args.algorithm_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    os.makedirs(str(log_dir))
    os.makedirs(str(save_dir))
    logger = SummaryWriter(str(log_dir)) 

    # env
    envs = make_parallel_env(args)
    num_agents = get_map_params(args.map_name)["n_agents"]
    #Policy network

    if args.share_policy:
        actor_critic = Policy(envs.observation_space[0], 
                    envs.action_space[0],
                    num_agents = num_agents,
                    base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'attn': args.attn,                                 
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart
                                 },
                    device = device)
        actor_critic.to(device)
        # algorithm
        agents = PPO(actor_critic,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device)
                   
        #replay buffer
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)        
    else:
        actor_critic = []
        agents = []
        for agent_id in range(num_agents):
            ac = Policy(envs.observation_space[0], 
                      envs.action_space[0],
                      num_agents = num_agents,
                      base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'attn': args.attn,                                 
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart
                                 },
                      device = device)
            ac.to(device)
            # algorithm
            agent = PPO(ac,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   device=device)
                               
            actor_critic.append(ac)
            agents.append(agent) 
              
        #replay buffer
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)
    
    # reset env 
    obs, available_actions = envs.reset()
    
    # replay buffer  
    if len(envs.observation_space[0]) == 3:
        share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])        
    else:
        share_obs = obs.reshape(args.n_rollout_threads, -1)
        
    share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
    rollouts.share_obs[0] = share_obs.copy() 
    rollouts.obs[0] = obs.copy()                
    rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
    rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    timesteps = 0
    last_battles_game = np.zeros(args.n_rollout_threads)
    last_battles_won = np.zeros(args.n_rollout_threads)

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for i in range(num_agents):
                    update_linear_schedule(agents[i].optimizer, episode, episodes, args.lr)           

        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            
            with torch.no_grad():                
                for i in range(num_agents):
                    if args.share_policy:
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(i,
                        torch.tensor(rollouts.share_obs[step,:,i]), 
                        torch.tensor(rollouts.obs[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,i]),
                        torch.tensor(rollouts.masks[step,:,i]),
                        available_actions[:,i,:])
                    else:
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[i].act(i,
                        torch.tensor(rollouts.share_obs[step,:,i]), 
                        torch.tensor(rollouts.obs[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,i]),
                        torch.tensor(rollouts.masks[step,:,i]),
                        available_actions[:,i,:])
                        
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    action_log_probs.append(action_log_prob.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
            
            # rearrange action           
            actions_env = []
            for i in range(args.n_rollout_threads):
                one_hot_action_env = []
                for k in range(num_agents):
                    one_hot_action = np.zeros(envs.action_space[0].n)
                    one_hot_action[actions[k][i]] = 1
                    one_hot_action_env.append(one_hot_action)
                actions_env.append(one_hot_action_env)
                       
            # Obser reward and next obs
            obs, reward, done, infos, available_actions = envs.step(actions_env)

            # If done then clean the history of observations.
            # insert data in buffer
            masks = []
            for done_ in done: 
                mask = []               
                for i in range(num_agents): 
                    if done_:              
                        mask.append([0.0])
                    else:
                        mask.append([1.0])
                masks.append(mask)
                
            bad_masks = []
            for info in infos: 
                bad_mask = []               
                for i in range(num_agents): 
                    if info['bad_transition']:              
                        bad_mask.append([0.0])
                    else:
                        bad_mask.append([1.0])
                bad_masks.append(bad_mask)
                            
            if len(envs.observation_space[0]) == 3:
                share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])
                share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)
                
                rollouts.insert(share_obs, 
                                obs, 
                                recurrent_hidden_statess.transpose(1,0,2), 
                                recurrent_hidden_statess_critic.transpose(1,0,2), 
                                actions.transpose(1,0,2),
                                action_log_probs.transpose(1,0,2), 
                                values.transpose(1,0,2),
                                reward, 
                                masks, 
                                bad_masks)
            else:
                share_obs = obs.reshape(args.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)
        
                rollouts.insert(share_obs, 
                                obs, 
                                np.array(recurrent_hidden_statess).transpose(1,0,2), 
                                np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                                np.array(actions).transpose(1,0,2),
                                np.array(action_log_probs).transpose(1,0,2), 
                                np.array(values).transpose(1,0,2),
                                reward, 
                                masks, 
                                bad_masks)
                           
        with torch.no_grad(): 
            for i in range(num_agents):         
                if args.share_policy:                 
                    next_value = actor_critic.get_value(i,
                                                   torch.tensor(rollouts.share_obs[-1,:,i]), 
                                                   torch.tensor(rollouts.obs[-1,:,i]), 
                                                   torch.tensor(rollouts.recurrent_hidden_states[-1,:,i]),
                                                   torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,i]),
                                                   torch.tensor(rollouts.masks[-1,:,i])).detach().cpu().numpy()
                    rollouts.compute_returns(i,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents.value_normalizer)
                else:
                    next_value = actor_critic[i].get_value(i,
                                                   torch.tensor(rollouts.share_obs[-1,:,i]), 
                                                   torch.tensor(rollouts.obs[-1,:,i]), 
                                                   torch.tensor(rollouts.recurrent_hidden_states[-1,:,i]),
                                                   torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,i]),
                                                   torch.tensor(rollouts.masks[-1,:,i])).detach().cpu().numpy()
                    rollouts.compute_returns(i,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents[i].value_normalizer)

         
        # update the network
        if args.share_policy:
            value_loss, action_loss, dist_entropy = agents.update_share(num_agents, rollouts)
                           
            logger.add_scalars('reward',
                {'reward': np.mean(rollouts.rewards)},
                (episode + 1) * args.episode_length * args.n_rollout_threads)
        else:
            value_losses = []
            action_losses = []
            dist_entropies = [] 
            
            for i in range(num_agents):
                value_loss, action_loss, dist_entropy = agents[i].update(i, rollouts)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                
                rew = []
                for j in range(rollouts.rewards.shape[1]):
                    rew.append(rollouts.rewards[:,j,i].sum())
                    
                logger.add_scalars('agent%i/reward' % i,
                    {'reward': np.mean(np.array(rew)/(rollouts.rewards.shape[0]))},
                    (episode + 1) * args.episode_length * args.n_rollout_threads)
                                                                     
        # clean the buffer and reset
        obs, available_actions = envs.reset()
        
        if len(envs.observation_space[0]) == 3:
            share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])
        else:
            share_obs = obs.reshape(args.n_rollout_threads, -1)
            
        share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
        rollouts.share_obs[0] = share_obs.copy() 
        rollouts.obs[0] = obs.copy()  
        rollouts.recurrent_hidden_states[0] = np.zeros(rollouts.recurrent_hidden_states.shape[1:]).copy()
        rollouts.recurrent_hidden_states_critic[0] = np.zeros(rollouts.recurrent_hidden_states_critic.shape[1:]).copy()
        rollouts.masks[0] = np.ones(rollouts.masks.shape[1:]).copy()
        rollouts.bad_masks[0] = np.ones(rollouts.bad_masks.shape[1:]).copy()
        
        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(save_dir) + "/agent_model.pt")
            else:
                for i in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[i]
                                }, 
                                str(save_dir) + "/agent%i_model" % i + ".pt")

        # log information
        if episode % args.log_interval == 0:
            total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads
            end = time.time()
            print("\n Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - start))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
            else:
                for i in range(num_agents):
                    print("value loss of agent%i: " %i + str(value_losses[i]))

            if args.env_name == "StarCraft2":                
                battles_won = []
                battles_game = []
                battles_draw = []
                win_rate = []
                incre_win_rate = []
                for i,info in enumerate(infos):
                    if 'battles_won' in info.keys():
                        battles_won.append(info['battles_won'])                         
                    if 'battles_game' in info.keys():
                        battles_game.append(info['battles_game'])                        
                        if info['battles_game'] == 0:
                            win_rate.append(0)
                            incre_win_rate.append(0) 
                        else:
                            win_rate.append(info['battles_won']/info['battles_game'])
                            if info['battles_game']-last_battles_game[i] == 0:
                                incre_win_rate.append(0)
                            else:
                                incre_win_rate.append((info['battles_won']-last_battles_won[i])/(info['battles_game']-last_battles_game[i]))
                          
                    if 'battles_draw' in info.keys():
                        battles_draw.append(info['battles_draw'])
                logger.add_scalars('win_rate',
                                    {'win_rate': np.mean(win_rate)},
                                    total_num_steps)
                logger.add_scalars('incre_win_rate',
                                    {'incre_win_rate': np.mean(incre_win_rate)},
                                    total_num_steps)
                last_battles_game = battles_game
                last_battles_won = battles_won
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
if __name__ == "__main__":
    main()
