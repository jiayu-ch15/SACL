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

from envs import HanabiEnv
from algorithm.ppo import PPO
from algorithm.share_model import Policy

from config import get_config
from utils.env_wrappers import ChooseSubprocVecEnv
from utils.util import update_linear_schedule
from utils.share_storage import RolloutStorage
import shutil
import numpy as np

import wandb

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
        return ChooseSubprocVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
        
def make_eval_env(args):
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
    return ChooseSubprocVecEnv([get_env_fn(0)])

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
        eval_env = make_eval_env(args)
    num_agents = args.num_agents

    #Policy network
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
                                'use_ReLU':args.use_ReLU                                 
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
                                'use_ReLU':args.use_ReLUm
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
    reset_choose = np.ones(args.n_rollout_threads)==1.0 
    obs, share_obs, available_actions = envs.reset(reset_choose)
    # replay buffer  
    use_obs = obs.copy()
    use_share_obs = share_obs.copy()
    use_available_actions = available_actions.copy()
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads

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
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)          
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
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(torch.FloatTensor(use_share_obs[choose,current_agent_id]), 
                            torch.FloatTensor(use_obs[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states_critic[choose,current_agent_id]),
                            torch.FloatTensor(turn_masks[choose,current_agent_id]),
                            torch.FloatTensor(use_available_actions[choose,current_agent_id]))
                    else:
                        actor_critic[current_agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[current_agent_id].act(torch.FloatTensor(use_share_obs[choose,current_agent_id]), 
                            torch.FloatTensor(use_obs[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states[choose,current_agent_id]), 
                            torch.FloatTensor(turn_recurrent_hidden_states_critic[choose,current_agent_id]),
                            torch.FloatTensor(turn_masks[choose,current_agent_id]),
                            torch.FloatTensor(use_available_actions[choose,current_agent_id]))
                    
                    turn_obs[choose,current_agent_id] = use_obs[choose,current_agent_id].copy()
                    turn_share_obs[choose,current_agent_id] = use_share_obs[choose,current_agent_id].copy()
                    turn_available_actions[choose,current_agent_id] = use_available_actions[choose,current_agent_id].copy()
                    turn_values[choose,current_agent_id] = value.detach().cpu().numpy()
                    turn_actions[choose,current_agent_id] = action.detach().cpu().numpy()
                    env_actions[choose,current_agent_id] = action.detach().cpu().numpy()
                    turn_action_log_probs[choose,current_agent_id] = action_log_prob.detach().cpu().numpy()
                    turn_recurrent_hidden_states[choose,current_agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                    turn_recurrent_hidden_states_critic[choose,current_agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()
                    
                    obs, share_obs, reward, done, infos, available_actions = envs.step(env_actions) 
                    
                    use_obs = obs.copy()
                    use_share_obs = share_obs.copy()
                    use_available_actions = available_actions.copy()
                    
                    turn_rewards_since_last_action[choose] += reward[choose]                    
                    turn_rewards[choose, current_agent_id] = turn_rewards_since_last_action[choose, current_agent_id].copy()
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
                                
                                turn_values[n_rollout_thread,left_agent_id] = 0.0
                                turn_obs[n_rollout_thread,left_agent_id] = use_obs[n_rollout_thread,left_agent_id]                                      
                                turn_share_obs[n_rollout_thread,left_agent_id] = use_share_obs[n_rollout_thread,left_agent_id]                             
                            turn_masks[n_rollout_thread] = np.zeros((num_agents, 1)).astype(np.float32)
                            turn_recurrent_hidden_states[n_rollout_thread] = np.zeros((num_agents, *rollouts.recurrent_hidden_states.shape[3:])).astype(np.float32)
                            turn_recurrent_hidden_states_critic[n_rollout_thread] = np.zeros((num_agents, *rollouts.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)                            
                            
                            if 'score' in infos[n_rollout_thread].keys():
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
            obs, share_obs, available_actions = envs.reset(reset_choose)

            use_obs[reset_choose] = obs[reset_choose]
            use_share_obs[reset_choose] = share_obs[reset_choose]
            use_available_actions[reset_choose] = available_actions[reset_choose]
                    
        rollouts.share_obs[-1] = use_share_obs.copy()
        rollouts.obs[-1] = use_obs.copy()
        rollouts.available_actions[-1] = use_available_actions.copy()
        
        with torch.no_grad(): 
            for agent_id in range(num_agents):         
                if args.share_policy:
                    actor_critic.eval()                 
                    next_value, _, _ = actor_critic.get_value(torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
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
                    next_value, _, _ = actor_critic[agent_id].get_value(torch.tensor(rollouts.share_obs[-1,:,agent_id]), 
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
        
        # remove useless data in buffer
        # update the network
        if args.share_policy:
            actor_critic.train()
            value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = agents.update_share(num_agents, rollouts)
            
        else:
            value_losses = []
            action_losses = []
            dist_entropies = [] 
            grad_norms = []
            KL_divlosses = []
            ratios = []        
            for agent_id in range(num_agents):
                actor_critic[agent_id].train()
                value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = agents[agent_id].update(agent_id, rollouts)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                grad_norms.append(grad_norm)
                KL_divlosses.append(KL_divloss)
                ratios.append(ratio))
                                                                    
        # clean the buffer and reset
        rollouts.chooseafter_update()
        
        total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads
        
        # save model
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
            print("\n Algos {} Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(args.algorithm_name,
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
                wandb.log({"average_step_rewards": np.mean(rollouts.rewards)}, step=total_num_steps)
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id]))
                    wandb.log({"agent%i/value_loss" % agent_id: value_losses[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/action_loss" % agent_id: action_losses[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/grad_norm" % agent_id: grad_norms[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/KL_divloss" % agent_id: KL_divlosses[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/ratio"% agent_id: ratios[agent_id]}, step=total_num_steps)
                    wandb.log({"agent%i/average_step_rewards" % agent_id: np.mean(rollouts.rewards[:,:,agent_id])}, step=total_num_steps)

            if args.env_name == "Hanabi":  
                if len(scores)>0: 
                    wandb.log({'score': np.mean(scores)}, step=total_num_steps)
                    print("Mean score is {}.".format(np.mean(scores)))
                else:
                    wandb.log({'score': 0}, step=total_num_steps)
                    print("Can not access mean score.")
                
                
        if episode % args.eval_interval == 0 and args.eval:
            eval_scores = []
            eval_episode = 0
            eval_obs, eval_share_obs, eval_available_actions = eval_env.reset([True])
            eval_actions = np.ones((1, num_agents, 1)).astype(np.float32)*(-1.0)
            eval_recurrent_hidden_states = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
            eval_masks = np.ones((1,num_agents,1)).astype(np.float32)
            
            while True:               
                for agent_id in range(num_agents):
                    if args.share_policy:
                        actor_critic.eval()
                        _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(torch.FloatTensor(eval_share_obs[:,agent_id]), 
                            torch.FloatTensor(eval_obs[:,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states[:,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states_critic[:,agent_id]),
                            torch.FloatTensor(eval_masks[:,agent_id]),
                            torch.FloatTensor(eval_available_actions[:,agent_id]),
                            deterministic=True)
                    else:
                        actor_critic[agent_id].eval()
                        _, action, _, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(torch.FloatTensor(eval_share_obs[:,agent_id]), 
                            torch.FloatTensor(eval_obs[:,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states[:,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states_critic[:,agent_id]),
                            torch.FloatTensor(eval_masks[:,agent_id]),
                            torch.FloatTensor(eval_available_actions[:,agent_id]),
                            deterministic=True)

                    eval_actions[:,agent_id] = action.detach().cpu().numpy()
                    eval_recurrent_hidden_states[:,agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                    eval_recurrent_hidden_states_critic[:,agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()
                        
                    # Obser reward and next obs
                    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_env.step(eval_actions)
                                                    
                    if eval_dones[0]:                         
                        eval_episode += 1
                        if 'score' in eval_infos[0].keys():
                            eval_scores.append(eval_infos[0]['score'])
                        eval_obs, eval_share_obs, eval_available_actions = eval_env.reset([True])
                        
                        eval_recurrent_hidden_states = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
                        eval_recurrent_hidden_states_critic = np.zeros((1,num_agents,args.hidden_size)).astype(np.float32)
                        eval_actions = np.ones((1, num_agents, 1)).astype(np.float32)*(-1.0)
                        break
                
                if eval_episode>=args.eval_episodes:
                    wandb.log({'eval_score': np.mean(eval_scores)}, step=total_num_steps)
                    break
    envs.close()
    if args.eval:
        eval_env.close()
    run.finish()

if __name__ == "__main__":
    main()
