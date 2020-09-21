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
from algorithm.share_model import Policy

from config import get_config
from utils.env_wrappers import ShareSubprocVecEnv
from utils.util import update_linear_schedule
from utils.share_storage import RolloutStorage
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
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return ShareSubprocVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_eval_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "StarCraft2":
                env = StarCraft2Env(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    return ShareSubprocVecEnv([get_env_fn(0)])

def main():
    args = get_config()

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
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
    if args.eval:
        eval_env = make_eval_env(args)
    num_agents = get_map_params(args.map_name)["n_agents"]
    #Policy network

    if args.share_policy:
        actor_critic = Policy(envs.observation_space[0], 
                    envs.share_observation_space[0], 
                    envs.action_space[0],
                    gain = args.gain,
                    base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'recurrent_N': args.recurrent_N,
                                 'attn': args.attn,                            
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal,
                                 'layer_N':args.layer_N,
                                 'use_ReLU':args.use_ReLU
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
                   use_value_high_masks=args.use_value_high_masks,
                   device=device)
                   
        #replay buffer
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0], 
                    envs.share_observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)        
    else:
        actor_critic = []
        agents = []
        for agent_id in range(num_agents):
            ac = Policy(envs.observation_space[0],
                        envs.share_observation_space[0],  
                        envs.action_space[0],
                        gain = args.gain,
                        base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size,
                                 'recurrent_N': args.recurrent_N,
                                 'attn': args.attn,                               
                                 'attn_size': args.attn_size,
                                 'attn_N': args.attn_N,
                                 'attn_heads': args.attn_heads,
                                 'dropout': args.dropout,
                                 'use_average_pool': args.use_average_pool,
                                 'use_common_layer':args.use_common_layer,
                                 'use_feature_normlization':args.use_feature_normlization,
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal,
                                 'layer_N':args.layer_N,
                                 'use_ReLU':args.use_ReLU
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
                   use_value_high_masks=args.use_value_high_masks,
                   device=device)
                               
            actor_critic.append(ac)
            agents.append(agent) 
              
        #replay buffer
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0],
                    envs.share_observation_space[0],  
                    envs.action_space[0],
                    args.hidden_size)
    
    # reset env 
    obs, share_obs, available_actions = envs.reset()
    
    # replay buffer       
    share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
    rollouts.share_obs[0] = share_obs.copy() 
    rollouts.obs[0] = obs.copy()  
    rollouts.available_actions[0] = available_actions.copy()                
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
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           

        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            
            with torch.no_grad():                
                for agent_id in range(num_agents):
                    if args.share_policy:
                        actor_critic.eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                                        torch.tensor(rollouts.obs[step,:,agent_id]), 
                                        torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                                        torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                                        torch.tensor(rollouts.masks[step,:,agent_id]),
                                        torch.tensor(rollouts.available_actions[step,:,agent_id]))
                    else:
                        actor_critic[agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(torch.tensor(rollouts.share_obs[step,:,agent_id]), 
                                        torch.tensor(rollouts.obs[step,:,agent_id]), 
                                        torch.tensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                                        torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                                        torch.tensor(rollouts.masks[step,:,agent_id]),
                                        torch.tensor(rollouts.available_actions[step,:,agent_id]))
                        
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    action_log_probs.append(action_log_prob.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
            
            # rearrange action           
            actions_env = []
            for i in range(args.n_rollout_threads):
                one_hot_action_env = []
                for agent_id in range(num_agents):
                    one_hot_action = np.zeros(envs.action_space[agent_id].n)
                    one_hot_action[actions[agent_id][i]] = 1
                    one_hot_action_env.append(one_hot_action)
                actions_env.append(one_hot_action_env)
                       
            # Obser reward and next obs
            obs, share_obs, reward, dones, infos, available_actions = envs.step(actions_env)

            # If done then clean the history of observations.
            # insert data in buffer
            masks = []
            for i, done in enumerate(dones): 
                mask = []               
                for agent_id in range(num_agents): 
                    if done:    
                        recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                        recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
                        mask.append([0.0])
                    else:
                        mask.append([1.0])
                masks.append(mask)
                
            bad_masks = []
            high_masks = []
            for info in infos: 
                bad_mask = []  
                high_mask = []             
                for agent_id in range(num_agents): 
                    if info[agent_id]['bad_transition']:              
                        bad_mask.append([0.0])
                    else:
                        bad_mask.append([1.0])
                        
                    if info[agent_id]['high_masks']:              
                        high_mask.append([1.0])
                    else:
                        high_mask.append([0.0])
                bad_masks.append(bad_mask)
                high_masks.append(high_mask)
                            
            if len(envs.observation_space[0]) == 3:
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
                                bad_masks,
                                high_masks,
                                available_actions)
            else:
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
                                bad_masks,
                                high_masks,
                                available_actions)
                           
        with torch.no_grad(): 
            for agent_id in range(num_agents):         
                if args.share_policy: 
                    actor_critic.eval()                
                    next_value,_,_ = actor_critic.get_value(torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
                                                   torch.tensor(rollouts.obs[-1,:,agent_id]), 
                                                   torch.tensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                   torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                   torch.tensor(rollouts.masks[-1,:,agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.compute_returns(agent_id,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents.value_normalizer)
                else:
                    actor_critic[agent_id].eval()
                    next_value,_,_ = actor_critic[agent_id].get_value(torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
                                                   torch.tensor(rollouts.obs[-1,:,agent_id]), 
                                                   torch.tensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                   torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                   torch.tensor(rollouts.masks[-1,:,agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.compute_returns(agent_id,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents[agent_id].value_normalizer)

         
        # update the network
        if args.share_policy:
            actor_critic.train()
            value_loss, action_loss, dist_entropy = agents.update_share(num_agents, rollouts)
                           
            logger.add_scalars('reward',
                {'reward': np.mean(rollouts.rewards)},
                (episode + 1) * args.episode_length * args.n_rollout_threads)
        else:
            value_losses = []
            action_losses = []
            dist_entropies = [] 
            
            for agent_id in range(num_agents):
                actor_critic[agent_id].train()
                value_loss, action_loss, dist_entropy = agents[agent_id].update(agent_id, rollouts)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                                    
                logger.add_scalars('agent%i/reward' % agent_id,
                    {'reward': np.mean(rollouts.rewards[:,:,agent_id])},
                    (episode + 1) * args.episode_length * args.n_rollout_threads)
                                                                     
        # clean the buffer and reset
        rollouts.after_update()

        total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads

        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(save_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
                                }, 
                                str(save_dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % args.log_interval == 0:
            end = time.time()
            print("\n Map {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(args.map_name,
                        args.algorithm_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - start))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id]))

            if args.env_name == "StarCraft2":                
                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i,info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])                         
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])                                                
                        incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                if np.sum(incre_battles_game)>0:
                    logger.add_scalars('incre_win_rate',
                                    {'incre_win_rate': np.sum(incre_battles_won)/np.sum(incre_battles_game)},
                                    total_num_steps)
                else:
                    logger.add_scalars('incre_win_rate',
                                    {'incre_win_rate': 0},
                                    total_num_steps)
                last_battles_game = battles_game
                last_battles_won = battles_won

        if episode % args.eval_interval == 0 and args.eval:
            eval_battles_won = 0
            eval_episode = 0
            eval_obs, eval_share_obs, eval_available_actions = eval_env.reset()
            eval_recurrent_hidden_states = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
            eval_masks = np.ones((1,num_agents,1)).astype(np.float32)
            
            while True:
                eval_actions = []               
                for agent_id in range(num_agents):
                    if args.share_policy:
                        actor_critic.eval()
                        _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(torch.tensor(eval_share_obs), 
                                        torch.tensor(eval_obs[:,agent_id]), 
                                        torch.tensor(eval_recurrent_hidden_states[:,agent_id]), 
                                        torch.tensor(eval_recurrent_hidden_states_critic[:,agent_id]),
                                        torch.tensor(eval_masks[:,agent_id]),
                                        torch.tensor(eval_available_actions[:,agent_id,:]),
                                        deterministic=True)
                    else:
                        actor_critic[agent_id].eval()
                        _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(torch.tensor(eval_share_obs), 
                                        torch.tensor(eval_obs[:,agent_id]), 
                                        torch.tensor(eval_recurrent_hidden_states[:,agent_id]), 
                                        torch.tensor(eval_recurrent_hidden_states_critic[:,agent_id]),
                                        torch.tensor(eval_masks[:,agent_id]),
                                        torch.tensor(eval_available_actions[:,agent_id,:]),
                                        deterministic=True)

                    eval_actions.append(action.detach().cpu().numpy())
                    eval_recurrent_hidden_states[:,agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                    eval_recurrent_hidden_states_critic[:,agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()

                # rearrange action           
                eval_actions_env = []
                for agent_id in range(num_agents):
                    one_hot_action = np.zeros(eval_env.action_space[agent_id].n)
                    one_hot_action[eval_actions[agent_id][0]] = 1
                    eval_actions_env.append(one_hot_action)
                        
                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_env.step([eval_actions_env])
                                                    
                if eval_dones[0]: 
                    eval_episode += 1
                    if eval_infos[0][0]['won']:
                        eval_battles_won += 1
                    for agent_id in range(num_agents):    
                        eval_recurrent_hidden_states[0][agent_id] = np.zeros(args.hidden_size).astype(np.float32)
                        eval_recurrent_hidden_states_critic[0][agent_id] = np.zeros(args.hidden_size).astype(np.float32)    
                        eval_masks[0][agent_id]=0.0
                else:
                    for agent_id in range(num_agents):
                        eval_masks[0][agent_id]=1.0
                
                if eval_episode>=args.eval_episodes:
                    logger.add_scalars('eval_win_rate',
                                    {'eval_win_rate': eval_battles_won/eval_episode},
                                    total_num_steps)
                    break
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
    if args.eval:
        eval_env.close()
if __name__ == "__main__":
    main()
