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
from utils.env_wrappers import ChooseSubprocVecEnv, DummyVecEnv
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
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])

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
    model_dir = Path('./results') / args.env_name / args.hanabi_name /args.algorithm_name
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
    reset_choose = np.ones(args.n_rollout_threads)==1.0 
    obs, available_actions = envs.reset(reset_choose)
        
    # replay buffer  
    if len(envs.observation_space[0]) == 3:
        share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])        
    else:
        share_obs = obs.reshape(args.n_rollout_threads, -1)

    share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)
       
    use_obs = obs
    use_share_obs = share_obs
    use_available_actions = available_actions
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    timesteps = 0

    turn_obs = np.zeros((args.n_rollout_threads, num_agents, *rollouts.obs.shape[3:])).astype(np.float32)
    turn_share_obs = np.zeros((args.n_rollout_threads, num_agents, *rollouts.share_obs.shape[3:])).astype(np.float32)
    turn_available_actions = np.zeros((args.n_rollout_threads, num_agents, *rollouts.available_actions.shape[3:])).astype(np.float32)
    turn_values =  np.zeros((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_actions = np.zeros((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    env_actions = np.ones((args.n_rollout_threads, num_agents, 1)).astype(np.float32)*(-1.0)
    turn_action_log_probs = np.zeros((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_recurrent_hidden_states = np.zeros((args.n_rollout_threads, num_agents, *rollouts.recurrent_hidden_states.shape[3:])).astype(np.float32)
    turn_recurrent_hidden_states_critic = np.zeros((args.n_rollout_threads, num_agents, *rollouts.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)
    turn_masks = np.ones((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_high_masks = np.ones((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_bad_masks = np.ones((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_rewards_since_last_action = np.zeros((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_rewards = np.zeros((args.n_rollout_threads, num_agents, 1)).astype(np.float32)

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
            reset_choose = np.zeros(args.n_rollout_threads)==1.0          
            with torch.no_grad():                  
                for current_agent_id in range(num_agents):
                    env_actions[:,current_agent_id] = np.ones((args.n_rollout_threads, 1)).astype(np.float32)*(-1.0)                                                               
                    choose = np.any(use_available_actions[:,current_agent_id]==1,axis=1) 
                    if ~np.any(choose):
                        reset_choose = np.ones(args.n_rollout_threads)==1.0
                        break                 
                    if args.share_policy:
                        actor_critic.eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(current_agent_id,
                            torch.FloatTensor(use_share_obs[choose,current_agent_id]), 
                            torch.FloatTensor(use_obs[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states_critic[choose,current_agent_id]),
                            torch.FloatTensor(turn_masks[choose,current_agent_id]),
                            torch.FloatTensor(use_available_actions[choose,current_agent_id]))
                    else:
                        actor_critic[current_agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[current_agent_id].act(current_agent_id,
                            torch.FloatTensor(use_share_obs[choose,current_agent_id]), 
                            torch.FloatTensor(use_obs[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states_critic[choose,current_agent_id]),
                            torch.FloatTensor(turn_masks[choose,current_agent_id]),
                            torch.FloatTensor(use_available_actions[choose,current_agent_id]))
                    
                    turn_obs[choose,current_agent_id] = use_obs[choose,current_agent_id]
                    turn_share_obs[choose,current_agent_id] = use_share_obs[choose,current_agent_id]
                    turn_available_actions[choose,current_agent_id] = use_available_actions[choose,current_agent_id]
                    turn_values[choose,current_agent_id] = value.detach().cpu().numpy()
                    turn_actions[choose,current_agent_id] = action.detach().cpu().numpy()
                    env_actions[choose,current_agent_id] = action.detach().cpu().numpy()
                    turn_action_log_probs[choose,current_agent_id] = action_log_prob.detach().cpu().numpy()
                    turn_recurrent_hidden_states[choose,current_agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                    turn_recurrent_hidden_states_critic[choose,current_agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()
                    
                    #obs, reward, done, infos, available_actions = envs.step_choose(turn_actions, choose)
                    obs, reward, done, infos, available_actions = envs.step(env_actions) #TODO
                    
                    if len(envs.observation_space[0]) == 3:
                        share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])        
                    else:
                        share_obs = obs.reshape(args.n_rollout_threads, -1)
                       
                    share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)
                    
                    use_obs = obs
                    use_share_obs = share_obs
                    use_available_actions = available_actions
                    
                    turn_rewards_since_last_action[choose] += reward[choose]                    
                    turn_rewards[choose, current_agent_id] = turn_rewards_since_last_action[choose, current_agent_id]
                    turn_rewards_since_last_action[choose, current_agent_id] = 0.0
                    
                    for n_rollout_thread in range(args.n_rollout_threads):
                        if done[n_rollout_thread]:
                            use_available_actions[n_rollout_thread] = np.zeros((num_agents, *rollouts.available_actions.shape[3:]))
                            reset_choose[n_rollout_thread]=True
                            turn_high_masks[n_rollout_thread, current_agent_id] = 1.0                              
                            for left_agent_id in range(current_agent_id + 1, num_agents):
                                turn_high_masks[n_rollout_thread, left_agent_id] = 0.0
                                turn_rewards[n_rollout_thread, left_agent_id] = turn_rewards_since_last_action[n_rollout_thread, left_agent_id]
                                turn_rewards_since_last_action[n_rollout_thread, left_agent_id] = 0.0
                                # other variables use what at last time, action will be useless.
                                if args.share_policy:                            
                                    actor_critic.eval()
                                    value, _, _ = actor_critic.get_value(left_agent_id,
                                                        torch.FloatTensor(use_share_obs[n_rollout_thread,left_agent_id]).unsqueeze(0), 
                                                        torch.FloatTensor(use_obs[n_rollout_thread,left_agent_id]).unsqueeze(0), 
                                                        torch.FloatTensor(turn_recurrent_hidden_states[n_rollout_thread,left_agent_id]).unsqueeze(0), 
                                                        torch.FloatTensor(turn_recurrent_hidden_states_critic[n_rollout_thread,left_agent_id]).unsqueeze(0),
                                                        torch.FloatTensor(turn_masks[n_rollout_thread,left_agent_id]).unsqueeze(0)
                                                        )
                                else:
                                    actor_critic[left_agent_id].eval()
                                    value, _, _ = actor_critic[left_agent_id].get_value(left_agent_id,
                                                        torch.FloatTensor(use_share_obs[n_rollout_thread,left_agent_id]).unsqueeze(0), 
                                                        torch.FloatTensor(use_obs[n_rollout_thread,left_agent_id]).unsqueeze(0), 
                                                        torch.FloatTensor(turn_recurrent_hidden_states[n_rollout_thread,left_agent_id]).unsqueeze(0), 
                                                        torch.FloatTensor(turn_recurrent_hidden_states_critic[n_rollout_thread,left_agent_id]).unsqueeze(0),
                                                        torch.FloatTensor(turn_masks[n_rollout_thread,left_agent_id]).unsqueeze(0)
                                                        )
                                turn_obs[n_rollout_thread,left_agent_id] = use_obs[n_rollout_thread,left_agent_id]                                
                                turn_values[n_rollout_thread,left_agent_id] = value.detach().cpu().numpy()                                
                                turn_share_obs[n_rollout_thread,left_agent_id] = use_share_obs[n_rollout_thread,left_agent_id]                            
                            turn_masks[n_rollout_thread] = np.zeros((num_agents, 1)).astype(np.float32)
                            turn_recurrent_hidden_states[n_rollout_thread] = np.zeros((num_agents, *rollouts.recurrent_hidden_states.shape[3:])).astype(np.float32)
                            turn_recurrent_hidden_states_critic[n_rollout_thread] = np.zeros((num_agents, *rollouts.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)                            
                            
                            if 'score' in infos[n_rollout_thread].keys():
                                print(infos[n_rollout_thread]['score'])
                                scores.append(infos[n_rollout_thread]['score'])
                        elif done[n_rollout_thread] == None:
                            pass
                        else:
                            turn_masks[n_rollout_thread,current_agent_id]=1.0
                            turn_high_masks[n_rollout_thread,current_agent_id] = 1.0
            
            # insert turn data into buffer
            rollouts.chooseinsert(turn_share_obs, 
                                  turn_obs, 
                                  turn_recurrent_hidden_states, 
                                  turn_recurrent_hidden_states_critic, 
                                  turn_actions,
                                  turn_action_log_probs, 
                                  turn_values,
                                  turn_rewards, 
                                  turn_masks,
                                  turn_bad_masks,
                                  turn_high_masks,
                                  turn_available_actions)
                            
            # env reset
            obs, available_actions = envs.reset(reset_choose)          

            if len(envs.observation_space[0]) == 3:
                share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])        
            else:
                share_obs = obs.reshape(args.n_rollout_threads, -1)
                
            share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)
            
            use_obs[reset_choose] = obs[reset_choose]
            use_share_obs[reset_choose] = share_obs[reset_choose]
            use_available_actions[reset_choose] = available_actions[reset_choose]
                      
        rollouts.share_obs[-1] = use_share_obs.copy()
        rollouts.obs[-1] = use_obs.copy()
        rollouts.available_actions[-1] = use_available_actions.copy()
        
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
            # reward mask
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
        rollouts.chooseafter_update()
        
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
if __name__ == "__main__":
    main()
