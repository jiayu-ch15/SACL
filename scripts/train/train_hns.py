#!/usr/bin/env python

import copy
import glob
import os
import time
import shutil
import numpy as np
import wandb
import socket
from functools import reduce
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from algorithm.ppo import PPO
from utils.util import update_linear_schedule, MultiDiscrete

from envs import HNSEnv
from utils.env_wrappers import ShareSubprocVecEnv, ChooseSubprocVecEnv, ShareDummyVecEnv, ChooseDummyVecEnv
from utils.shared_storage import SharedRolloutStorage
from algorithm.model import Policy

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "HideAndSeek" or args.env_name == "BlueprintConstruction" or args.env_name == "BoxLocking":
                env = HNSEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            # TODO: check seed
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_eval_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "HideAndSeek" or args.env_name == "BlueprintConstruction" or args.env_name == "BoxLocking":
                env = HNSEnv(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(args.n_eval_rollout_threads)])

def main():
    args = get_config()
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    model_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / args.env_name / args.scenario_name / args.algorithm_name
    if not model_dir.exists():
        os.makedirs(str(model_dir))

    run = wandb.init(config=args, 
            project=args.env_name, 
            entity="yuchao",
            notes=socket.gethostname(),
            name=str(args.algorithm_name) + "_seed" + str(args.seed),
            group=args.scenario_name,
            dir=str(model_dir),
            job_type="training",
            reinit=True)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # env
    envs = make_parallel_env(args)
    if args.eval:
        eval_envs = make_eval_env(args)

    if args.env_name == "HideAndSeek":
        num_seekers = args.num_seekers
        num_hiders = args.num_hiders
        num_agents = num_seekers + num_hiders
    else:
        num_agents = args.num_agents

    if args.share_policy:
        if args.model_dir==None or args.model_dir=="":
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
                                'use_ReLU':args.use_ReLU,
                                'use_cat_self':args.use_cat_self
                                },
                    device = device)
        else:       
            actor_critic = torch.load(str(args.model_dir) + "/agent_model.pt")['model']
        
        actor_critic.to(device)
        # algorithm
        agents = PPO(actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.data_chunk_length,
                args.value_loss_coef,
                args.entropy_coef,
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
                use_value_active_masks=args.use_value_active_masks,
                device=device)
                
        #replay buffer
        rollouts = SharedRolloutStorage(num_agents,
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
            if args.model_dir==None or args.model_dir=="":
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
                                    'use_ReLU':args.use_ReLU,
                                    'use_cat_self':args.use_cat_self
                                    },
                        device = device)
            else:
                ac = torch.load(str(args.model_dir) + "/agent"+ str(agent_id) + "_model.pt")['model']
            
            ac.to(device)
            # algorithm
            agent = PPO(ac,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.data_chunk_length,
                args.value_loss_coef,
                args.entropy_coef,
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
                use_value_active_masks=args.use_value_active_masks,
                device=device)
                            
            actor_critic.append(ac)
            agents.append(agent) 
            
        #replay buffer
        rollouts = SharedRolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0],
                    envs.share_observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)
    
    # reset env
    obs, share_obs, _ = envs.reset() 

    # replay buffer 
    rollouts.share_obs[0] = share_obs.copy() 
    rollouts.obs[0] = obs.copy()                
    rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
    rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads

    for episode in range(episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           
        # info list
        # hide and seek
        max_box_move_prep = []
        max_box_move = []
        num_box_lock_prep = []
        num_box_lock = []
        max_ramp_move_prep = []
        max_ramp_move = []
        num_ramp_lock_prep = []
        num_ramp_lock = []
        food_eaten = []
        food_eaten_prep = []

        # transfer task
        discard_episode = 0
        success = 0
        trials = 0
        lock_rate = []
        activated_sites = []

        for step in range(args.episode_length):
            # Sample actions
            with torch.no_grad():
                if args.share_policy:
                    actor_critic.eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                        = actor_critic.act(torch.FloatTensor(np.concatenate(rollouts.share_obs[step])), 
                                            torch.FloatTensor(np.concatenate(rollouts.obs[step])), 
                                            torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states[step])), 
                                            torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states_critic[step])),
                                            torch.FloatTensor(np.concatenate(rollouts.masks[step])))
                    # [envs, agents, dim]
                    values = np.array(np.split(value.detach().cpu().numpy(),args.n_rollout_threads))
                    actions = np.array(np.split(action.detach().cpu().numpy(),args.n_rollout_threads))
                    action_log_probs = np.array(np.split(action_log_prob.detach().cpu().numpy(),args.n_rollout_threads))
                    recurrent_hidden_statess = np.array(np.split(recurrent_hidden_states.detach().cpu().numpy(),args.n_rollout_threads))
                    recurrent_hidden_statess_critic = np.array(np.split(recurrent_hidden_states_critic.detach().cpu().numpy(),args.n_rollout_threads))            
                else:
                    values = []
                    actions= []
                    action_log_probs = []
                    recurrent_hidden_statess = []
                    recurrent_hidden_statess_critic = []

                    for agent_id in range(num_agents):
                        actor_critic[agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                            = actor_critic[agent_id].act(torch.FloatTensor(rollouts.share_obs[step,:,agent_id]), 
                                                        torch.FloatTensor(rollouts.obs[step,:,agent_id]), 
                                                        torch.FloatTensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                                                        torch.FloatTensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                                                        torch.FloatTensor(rollouts.masks[step,:,agent_id]))
                        
                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
                    
                    values = np.array(values).transpose(1,0,2)
                    actions = np.array(actions).transpose(1,0,2)
                    action_log_probs = np.array(action_log_probs).transpose(1,0,2)
                    recurrent_hidden_statess = np.array(recurrent_hidden_statess).transpose(1,0,2)
                    recurrent_hidden_statess_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
  
            # Obser reward and next obs
            obs, share_obs, rewards, dones, infos, _ = envs.step(actions)
            if len(rewards.shape) < 3:
                rewards=rewards[:,:,np.newaxis]            

            # insert data in buffer
            recurrent_hidden_statess[dones==True] = np.zeros(((dones==True).sum(), num_agents, args.hidden_size)).astype(np.float32)
            recurrent_hidden_statess_critic[dones==True] = np.zeros(((dones==True).sum(), num_agents, args.hidden_size)).astype(np.float32)
            masks = np.ones((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
            masks[dones==True] = np.zeros(((dones==True).sum(), num_agents, 1)).astype(np.float32)
             
            for i in range(args.n_rollout_threads):              
                for agent_id in range(num_agents): 
                    if dones[i]: 
                        # get info to tensorboard
                        if args.env_name == "HideAndSeek":
                            if args.num_boxes > 0:
                                if 'max_box_move_prep' in infos[i].keys():
                                    max_box_move_prep.append(infos[i]['max_box_move_prep'])
                                if 'max_box_move' in infos[i].keys():
                                    max_box_move.append(infos[i]['max_box_move'])
                                if 'num_box_lock_prep' in infos[i].keys():
                                    num_box_lock_prep.append(infos[i]['num_box_lock_prep'])
                                if 'num_box_lock' in infos[i].keys():
                                    num_box_lock.append(infos[i]['num_box_lock'])
                            if args.num_ramps > 0:
                                if 'max_ramp_move_prep' in infos[i].keys():
                                    max_ramp_move_prep.append(infos[i]['max_ramp_move_prep'])
                                if 'max_ramp_move' in infos[i].keys():
                                    max_ramp_move.append(infos[i]['max_ramp_move'])
                                if 'num_ramp_lock_prep' in infos[i].keys():
                                    max_ramp_move.append(infos[i]['num_ramp_lock_prep'])
                                if 'num_ramp_lock' in infos[i].keys():
                                    max_ramp_move.append(infos[i]['num_ramp_lock'])
                            if args.num_food > 0:
                                if 'food_eaten' in infos[i].keys():
                                    food_eaten.append(infos[i]['food_eaten'])
                                if 'food_eaten_prep' in infos[i].keys():
                                    food_eaten_prep.append(infos[i]['food_eaten_prep'])                            
                        if args.env_name == "BlueprintConstruction" or args.env_name == "BoxLocking":
                            if "discard_episode" in infos[i].keys():
                                if infos[i]['discard_episode']:
                                    discard_episode += 1
                                else:
                                    trials += 1
                            else:
                                trials += 1

                            if "success" in infos[i].keys():
                                if infos[i]['success']:
                                    success += 1
                            if "lock_rate" in infos[i].keys():
                                lock_rate.append(infos[i]['lock_rate'])
                            if "activated_sites" in infos[i].keys():
                                activated_sites.append(infos[i]['activated_sites'])

            rollouts.insert(share_obs, 
                            obs, 
                            recurrent_hidden_statess, 
                            recurrent_hidden_statess_critic, 
                            actions,
                            action_log_probs, 
                            values,
                            rewards, 
                            masks)
                        
        if args.share_policy: 
            with torch.no_grad():
                actor_critic.eval()                
                next_value, _, _ = actor_critic.get_value(torch.FloatTensor(np.concatenate(rollouts.share_obs[-1])), 
                                                        torch.FloatTensor(np.concatenate(rollouts.obs[-1])), 
                                                        torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states[-1])),
                                                        torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states_critic[-1])),
                                                        torch.FloatTensor(np.concatenate(rollouts.masks[-1])))
                next_values = np.array(np.split(next_value.detach().cpu().numpy(), args.n_rollout_threads))
                rollouts.shared_compute_returns(next_values, 
                                                args.use_gae, 
                                                args.gamma,
                                                args.gae_lambda, 
                                                args.use_proper_time_limits,
                                                args.use_popart,
                                                agents.value_normalizer)
            # update network
            actor_critic.train()
            value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = agents.shared_update(rollouts)
            
        else:
            value_losses = []
            action_losses = []
            dist_entropies = [] 
            grad_norms = []
            KL_divlosses = []
            ratios = []
                            
            for agent_id in range(num_agents): 
                with torch.no_grad(): 
                    actor_critic[agent_id].eval()
                    next_value, _, _ = actor_critic[agent_id].get_value(torch.FloatTensor(rollouts.share_obs[-1,:,agent_id]), 
                                                torch.FloatTensor(rollouts.obs[-1,:,agent_id]), 
                                                torch.FloatTensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                torch.FloatTensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                torch.FloatTensor(rollouts.masks[-1,:,agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.single_compute_returns(agent_id,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents[agent_id].value_normalizer)
                # update network
                actor_critic[agent_id].train()
                value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = agents[agent_id].single_update(agent_id, rollouts)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                grad_norms.append(grad_norm)
                KL_divlosses.append(KL_divloss)
                ratios.append(ratio)                                                                      
        
        # clean the buffer and reset
        rollouts.after_update()

        # post process
        total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads
        
        # save model
        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(wandb.run.dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
                                }, 
                                str(wandb.run.dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % args.log_interval == 0:
            end = time.time()
            print("\n Scenario {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(args.scenario_name,
                        args.algorithm_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        args.num_env_steps,
                        int(total_num_steps / (end - start))))
            if args.share_policy:
                print("value loss of agent: " + str(value_loss))
                wandb.log({"value_loss": value_loss}, step=total_num_steps)
                wandb.log({"action_loss": action_loss}, step=total_num_steps)
                wandb.log({"dist_entropy": dist_entropy}, step=total_num_steps)
                wandb.log({"grad_norm": grad_norm}, step=total_num_steps)
                wandb.log({"KL_divloss": KL_divloss}, step=total_num_steps)
                wandb.log({"ratio": ratio}, step=total_num_steps)
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id])) 
                    wandb.log({"agent%i/value_loss" % agent_id: value_losses[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/action_loss" % agent_id: action_losses[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/grad_norm" % agent_id: grad_norms[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/KL_divloss" % agent_id: KL_divlosses[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/ratio"% agent_id: ratios[agent_id]}, step=total_num_steps)

            if args.env_name == "HideAndSeek":
                for hider_id in range(num_hiders):
                    wandb.log({'hider%i/average_step_rewards' % hider_id: np.mean(rollouts.rewards[:,:,hider_id])}, step=total_num_steps)
                for seeker_id in range(num_seekers):
                    wandb.log({'seeker%i/average_step_rewards' % seeker_id: np.mean(rollouts.rewards[:,:,num_hiders+seeker_id])}, step=total_num_steps)           

                if args.num_boxes > 0:
                    if len(max_box_move_prep) > 0:
                        wandb.log({'max_box_move_prep': np.mean(max_box_move_prep)}, step=total_num_steps)
                    if len(max_box_move) > 0:
                        wandb.log({'max_box_move': np.mean(max_box_move)}, step=total_num_steps)
                    if len(num_box_lock_prep) > 0:
                        wandb.log({'num_box_lock_prep': np.mean(num_box_lock_prep)}, step=total_num_steps)
                    if len(num_box_lock) > 0:
                        wandb.log({'num_box_lock': np.mean(num_box_lock)}, step=total_num_steps)
                if args.num_ramps > 0:
                    if len(max_ramp_move_prep) > 0:
                        wandb.log({'max_ramp_move_prep': np.mean(max_ramp_move_prep)}, step=total_num_steps)
                    if len(max_ramp_move) > 0:
                        wandb.log({'max_ramp_move': np.mean(max_ramp_move)}, step=total_num_steps)
                    if len(num_ramp_lock_prep) > 0:
                        wandb.log({'num_ramp_lock_prep': np.mean(num_ramp_lock_prep)}, step=total_num_steps)
                    if len(num_ramp_lock) > 0:
                        wandb.log({'num_ramp_lock': np.mean(num_ramp_lock)}, step=total_num_steps)
                if args.num_food > 0:
                    if len(food_eaten) > 0:
                        wandb.log({'food_eaten': np.mean(food_eaten)}, step=total_num_steps)
                    if len(food_eaten_prep) > 0:
                        wandb.log({'food_eaten_prep': np.mean(food_eaten_prep)}, step=total_num_steps)
            
            if args.env_name == "BoxLocking" or args.env_name == "BlueprintConstruction":  
                if args.share_policy:       
                    wandb.log({"average_step_rewards": np.mean(rollouts.rewards)}, step=total_num_steps)
                else:
                    for agent_id in range(num_agents):
                        wandb.log({"agent%i/average_step_rewards" % agent_id: np.mean(rollouts.rewards[:,:,agent_id])}, step=total_num_steps) 
                
                wandb.log({'discard_episode': discard_episode}, step=total_num_steps)
                
                if trials > 0:
                    wandb.log({'success_rate': success/trials}, step=total_num_steps) 
                    print("success rate is {}.".format(success/trials))
                else:
                    wandb.log({'success_rate': 0.0}, step=total_num_steps) 
                        
                if len(lock_rate) > 0: 
                    wandb.log({'lock_rate': np.mean(lock_rate)}, step=total_num_steps)  
 
                if len(activated_sites) > 0: 
                    wandb.log({'activated_sites': np.mean(activated_sites)}, step=total_num_steps)  
                    
        # eval 
        if episode % args.eval_interval == 0 and args.eval:
            action_shape = eval_envs.action_space[0].shape
            # hide and seek
            eval_num_box_lock_prep = []
            eval_num_box_lock = []

            # transfer task
            eval_success = 0
            eval_trials = 0
            eval_lock_rate = []
            eval_activated_sites = []
            eval_episode_rewards = []

            eval_reset_choose = np.ones(args.n_eval_rollout_threads)==1.0
            eval_obs, eval_share_obs, _ = eval_envs.reset(eval_reset_choose)

            eval_recurrent_hidden_states = np.zeros((args.n_eval_rollout_threads, num_agents, args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros((args.n_eval_rollout_threads, num_agents, args.hidden_size)).astype(np.float32)
            eval_masks = np.ones((args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
            eval_dones = np.zeros(args.n_eval_rollout_threads, dtype=bool)

            while True:  
                eval_choose = eval_dones==False
                if ~np.any(eval_choose):
                    break  
                if args.share_policy:
                    eval_actions = np.ones((args.n_eval_rollout_threads, num_agents, action_shape)).astype(np.int)
                    actor_critic.eval()
                    _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = actor_critic.act(torch.FloatTensor(np.concatenate(eval_share_obs[eval_choose])), 
                                    torch.FloatTensor(np.concatenate(eval_obs[eval_choose])), 
                                    torch.FloatTensor(np.concatenate(eval_recurrent_hidden_states[eval_choose])), 
                                    torch.FloatTensor(np.concatenate(eval_recurrent_hidden_states_critic[eval_choose])),
                                    torch.FloatTensor(np.concatenate(eval_masks[eval_choose])),
                                    deterministic=True)
                    eval_actions[eval_choose] = np.array(np.split(eval_action.detach().cpu().numpy(), args.n_eval_rollout_threads))         
                else:
                    eval_actions = [] 
                    for agent_id in range(num_agents):
                        agent_eval_actions = np.ones((args.n_eval_rollout_threads, action_shape)).astype(np.int)
                        actor_critic[agent_id].eval()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = actor_critic[agent_id].act(torch.FloatTensor(eval_share_obs[eval_choose,agent_id]), 
                                        torch.FloatTensor(eval_obs[eval_choose,agent_id]), 
                                        torch.FloatTensor(eval_recurrent_hidden_states[eval_choose,agent_id]), 
                                        torch.FloatTensor(eval_recurrent_hidden_states_critic[eval_choose,agent_id]),
                                        torch.FloatTensor(eval_masks[eval_choose,agent_id]),
                                        deterministic=True)
                        agent_eval_actions[eval_choose] = eval_action.detach().cpu().numpy()
                        eval_actions.append(agent_eval_actions)
                        eval_recurrent_hidden_states[eval_choose,agent_id] = eval_recurrent_hidden_state.detach().cpu().numpy()
                        eval_recurrent_hidden_states_critic[eval_choose,agent_id] = eval_recurrent_hidden_state_critic.detach().cpu().numpy()

                    eval_actions = np.array(eval_actions).transpose(1,0,2)
                    eval_recurrent_hidden_states = np.array(eval_recurrent_hidden_states).transpose(1,0,2)
                    eval_recurrent_hidden_states_critic = np.array(eval_recurrent_hidden_states_critic).transpose(1,0,2)

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = eval_envs.step(eval_actions)
                eval_episode_rewards.append(eval_rewards)

                eval_recurrent_hidden_states[eval_dones==True] = np.zeros(((eval_dones==True).sum(), num_agents, args.hidden_size)).astype(np.float32)
                eval_recurrent_hidden_states_critic[eval_dones==True] = np.zeros(((eval_dones==True).sum(), num_agents, args.hidden_size)).astype(np.float32)
                eval_masks = np.ones((args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
                eval_masks[eval_dones==True] = np.zeros(((eval_dones==True).sum(), num_agents, 1)).astype(np.float32)                                  
                
                for i in range(args.n_eval_rollout_threads):              
                    for agent_id in range(num_agents): 
                        if eval_dones[i]: 
                            # get info to tensorboard
                            if args.env_name == "HideAndSeek":
                                if args.num_boxes > 0:
                                    if 'num_box_lock_prep' in eval_infos[i].keys():
                                        eval_num_box_lock_prep.append(eval_infos[i]['num_box_lock_prep'])
                                    if 'num_box_lock' in eval_infos[i].keys():
                                        eval_num_box_lock.append(eval_infos[i]['num_box_lock'])
                            if args.env_name == "BlueprintConstruction" or args.env_name == "BoxLocking":
                                if "discard_episode" in eval_infos[i].keys():
                                    if not eval_infos[i]['discard_episode']:
                                        eval_trials += 1
                                else:
                                    eval_trials += 1

                                if "success" in eval_infos[i].keys():
                                    if eval_infos[i]['success']:
                                        eval_success += 1
                                if "lock_rate" in eval_infos[i].keys():
                                    eval_lock_rate.append(eval_infos[i]['lock_rate'])
                                if "activated_sites" in eval_infos[i].keys():
                                    eval_activated_sites.append(eval_infos[i]['activated_sites'])

            eval_episode_rewards = np.array(eval_episode_rewards)

            if args.env_name == "HideAndSeek":
                for hider_id in range(num_hiders):
                    wandb.log({'hider%i/eval_average_step_rewards' % hider_id: np.mean(eval_episode_rewards[:,:,hider_id])}, step=total_num_steps)
                for seeker_id in range(num_seekers):
                    wandb.log({'seeker%i/eval_average_step_rewards' % seeker_id: np.mean(eval_episode_rewards[:,:,num_hiders+seeker_id])}, step=total_num_steps)           
            
            if args.env_name == "BoxLocking" or args.env_name == "BlueprintConstruction":  
                if args.share_policy:       
                    wandb.log({"eval_average_step_rewards": np.mean(eval_episode_rewards)}, step=total_num_steps)
                else:
                    for agent_id in range(num_agents):
                        wandb.log({"agent%i/eval_average_step_rewards" % agent_id: np.mean(eval_episode_rewards[:,:,agent_id])}, step=total_num_steps) 
                            
                if eval_trials > 0:
                    wandb.log({'eval_success_rate': eval_success/eval_trials}, step=total_num_steps) 
                    print("eval success rate is {}.".format(eval_success/eval_trials))
                else:
                    wandb.log({'eval_success_rate': 0.0}, step=total_num_steps)                       
                if len(eval_lock_rate) > 0: 
                    wandb.log({'eval_lock_rate': np.mean(eval_lock_rate)}, step=total_num_steps)  
                if len(eval_activated_sites) > 0: 
                    wandb.log({'eval_activated_sites': np.mean(eval_activated_sites)}, step=total_num_steps)  
    
    envs.close()
    if args.eval:
        eval_envs.close()       
    run.finish()

if __name__ == "__main__":
    main()
