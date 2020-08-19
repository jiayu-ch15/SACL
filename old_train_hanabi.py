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

from envs import HanabiEnv
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
            if args.env_name == "Hanabi":
                assert args.num_agents>1 and args.num_agents<6, ("num_agents can be only between 2-5.")
                env = HanabiEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_test_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "Hanabi":
                assert args.num_agents>1 and args.num_agents<6, ("num_agents can be only between 2-5.")
                env = HanabiEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            return env
        return init_env
    return SubprocVecEnv([get_env_fn(0)])

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
    model_dir = Path('./results') / args.env_name / args.algorithm_name
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
    eval_env = make_test_env(args)
    num_agents = args.num_agents
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
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal
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
                                 'use_feature_popart':args.use_feature_popart,
                                 'use_orthogonal':args.use_orthogonal
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
    rollouts.available_actions[0] = available_actions.copy()                
    rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
    rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    timesteps = 0
    
    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for i in range(num_agents):
                    update_linear_schedule(agents[i].optimizer, episode, episodes, args.lr)          
        scores = []          
        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
                        
            with torch.no_grad():   
                for i in range(num_agents): 
                    if step==0:
                        replace_actions = torch.tensor(rollouts.actions[step,:,i])
                        replace_action_log_probs = torch.tensor(rollouts.action_log_probs[step,:,i])
                    else:
                        replace_actions = torch.tensor(rollouts.actions[step-1,:,i])  
                        replace_action_log_probs = torch.tensor(rollouts.action_log_probs[step-1,:,i])                        
                    if args.share_policy:
                        actor_critic.eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act_hanabi(i,
                        torch.tensor(rollouts.share_obs[step,:,i]), 
                        torch.tensor(rollouts.obs[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,i]),
                        torch.tensor(rollouts.masks[step,:,i]),
                        torch.tensor(rollouts.available_actions[step,:,i]),
                        replace_actions,
                        replace_action_log_probs)
                    else:
                        actor_critic[i].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[i].act_hanabi(i,
                        torch.tensor(rollouts.share_obs[step,:,i]), 
                        torch.tensor(rollouts.obs[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states[step,:,i]), 
                        torch.tensor(rollouts.recurrent_hidden_states_critic[step,:,i]),
                        torch.tensor(rollouts.masks[step,:,i]),
                        torch.tensor(rollouts.available_actions[step,:,i]),
                        replace_actions,
                        replace_action_log_probs)
                       
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    action_log_probs.append(action_log_prob.detach().cpu().numpy())                   
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())

                    
            # rearrange action 
            actions_env = np.array(actions).transpose(1,0,2)                        
            # Obser reward and next obs
            obs, reward, done, infos, available_actions = envs.step(actions_env)

            # If done then clean the history of observations.
            # insert data in buffer
            masks = []
            
            for k,done_ in enumerate(done):                
                if done_:
                    if 'score' in infos[k][0].keys():
                        score = infos[k][0]['score']
                        scores.append(score)                        
                mask = []               
                for i in range(num_agents): 
                    if done_: 
                        recurrent_hidden_statess[i][k] = np.zeros(args.hidden_size).astype(np.float32)
                        recurrent_hidden_statess_critic[i][k] = np.zeros(args.hidden_size).astype(np.float32)              
                        mask.append([0.0])
                    else:
                        mask.append([1.0])
                masks.append(mask)
                                             
            bad_masks = []
            high_masks = []
            for info in infos: 
                bad_mask = []  
                high_mask = []             
                for i in range(num_agents): 
                    bad_mask.append([1.0])
                        
                    if info[i]['high_masks']:              
                        high_mask.append([1.0])
                    else:
                        high_mask.append([0.0])
                bad_masks.append(bad_mask)
                high_masks.append(high_mask)
                             
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
                                bad_masks,
                                high_masks,
                                available_actions)
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
                                bad_masks,
                                high_masks,
                                available_actions)
                        
        with torch.no_grad(): 
            for i in range(num_agents):         
                if args.share_policy:
                    actor_critic.eval()                 
                    next_value,_,_ = actor_critic.get_value(i,
                                                   torch.tensor(rollouts.share_obs[-1,:,i]), 
                                                   torch.tensor(rollouts.obs[-1,:,i]), 
                                                   torch.tensor(rollouts.recurrent_hidden_states[-1,:,i]),
                                                   torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,i]),
                                                   torch.tensor(rollouts.masks[-1,:,i]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.compute_returns(i,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents.value_normalizer)
                else:
                    actor_critic[i].eval()
                    next_value,_,_ = actor_critic[i].get_value(i,
                                                   torch.tensor(rollouts.share_obs[-1,:,i]), 
                                                   torch.tensor(rollouts.obs[-1,:,i]), 
                                                   torch.tensor(rollouts.recurrent_hidden_states[-1,:,i]),
                                                   torch.tensor(rollouts.recurrent_hidden_states_critic[-1,:,i]),
                                                   torch.tensor(rollouts.masks[-1,:,i]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.compute_returns(i,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents[i].value_normalizer)
        
        # remove useless data in buffer
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
            
            for i in range(num_agents):
                actor_critic[i].train()
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
        rollouts.after_update()
        
        total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads

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
            if args.env_name == "Hanabi":        
                logger.add_scalars('score',{'score': np.mean(scores)},total_num_steps)
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
if __name__ == "__main__":
    main()
