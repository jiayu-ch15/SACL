#!/usr/bin/env python
import sys
import copy
import glob
import os
import time
import shutil
import wandb
import socket
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.ppo import PPO
from config import get_config
from utils.util import update_linear_schedule

from envs import HanabiEnv
from utils.env_wrappers import ChooseSubprocVecEnv, ChooseDummyVecEnv
from algorithm.model import Policy
from utils.shared_storage import SharedRolloutStorage

def make_parallel_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert all_args.num_agents > 1 and all_args.num_agents < 6, ("num_agents can be only between 2-5.")
                env = HanabiEnv(all_args, (all_args.seed + rank * 1000))
            else:
                print("Can not support the " + all_args.env_name + "environment." )
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env            
        return init_env
    if all_args.n_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
        
def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert all_args.num_agents > 1 and all_args.num_agents < 6, ("num_agents can be only between 2-5.")
                env = HanabiEnv(all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                print("Can not support the " + all_args.env_name + "environment." )
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--hanabi_name', type=str, default='Hanabi-Very-Small', help="Which env to run on")
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config() 
    all_args = parse_args(args, parser)

    if all_args.algorithm_name=="rmappo":
        assert (all_args.recurrent_policy or all_args.naive_recurrent_policy), ("check recurrent policy!")  
    elif all_args.algorithm_name=="mappo":
        assert (all_args.recurrent_policy and all_args.naive_recurrent_policy) == False, ("check recurrent policy!")  
    else:
        raise NotImplementedError
    
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.hanabi_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    run = wandb.init(config=all_args, 
            project=all_args.env_name, 
            entity="yuchao",
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.hanabi_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_parallel_env(all_args)
    if all_args.eval:
        eval_envs = make_eval_env(all_args)
    num_agents = all_args.num_agents

    #Policy network
    if all_args.share_policy:
        if all_args.model_dir==None or all_args.model_dir=="":
            actor_critic = Policy(envs.observation_space[0], 
                    envs.share_observation_space[0],
                    envs.action_space[0],
                    gain = all_args.gain,
                    base_kwargs={'naive_recurrent': all_args.naive_recurrent_policy,
                                'recurrent': all_args.recurrent_policy,
                                'hidden_size': all_args.hidden_size,
                                'recurrent_N': all_args.recurrent_N,
                                'attn': all_args.attn,                                 
                                'attn_size': all_args.attn_size,
                                'attn_N': all_args.attn_N,
                                'attn_heads': all_args.attn_heads,
                                'dropout': all_args.dropout,
                                'use_average_pool': all_args.use_average_pool,
                                'use_common_layer':all_args.use_common_layer,
                                'use_feature_normlization':all_args.use_feature_normlization,
                                'use_feature_popart':all_args.use_feature_popart,
                                'use_orthogonal':all_args.use_orthogonal,
                                'layer_N':all_args.layer_N,
                                'use_ReLU':all_args.use_ReLU,
                                'use_cat_self': all_args.use_cat_self                                
                                },
                    device = device)
        else:       
            actor_critic = torch.load(str(all_args.model_dir) + "/agent_model.pt")['model']
        
        actor_critic.to(device)
        # algorithm
        agents = PPO(actor_critic,
                all_args.clip_param,
                all_args.ppo_epoch,
                all_args.num_mini_batch,
                all_args.data_chunk_length,
                all_args.value_loss_coef,
                all_args.entropy_coef,
                lr=all_args.lr,
                eps=all_args.eps,
                weight_decay=all_args.weight_decay,
                max_grad_norm=all_args.max_grad_norm,
                use_max_grad_norm=all_args.use_max_grad_norm,
                use_clipped_value_loss= all_args.use_clipped_value_loss,
                use_common_layer=all_args.use_common_layer,
                use_huber_loss=all_args.use_huber_loss,
                huber_delta=all_args.huber_delta,
                use_popart=all_args.use_popart,
                use_value_active_masks=all_args.use_value_active_masks,
                device=device)
                
        #replay buffer
        rollouts = SharedRolloutStorage(num_agents,
                    all_args.episode_length, 
                    all_args.n_rollout_threads,
                    envs.observation_space[0],
                    envs.share_observation_space[0], 
                    envs.action_space[0],
                    all_args.hidden_size)        
    else:
        actor_critic = []
        agents = []
        for agent_id in range(num_agents):
            if all_args.model_dir==None or all_args.model_dir=="":
                ac = Policy(envs.observation_space[0],
                    envs.share_observation_space[0], 
                    envs.action_space[0],
                    gain = all_args.gain,
                    base_kwargs={'naive_recurrent': all_args.naive_recurrent_policy,
                                'recurrent': all_args.recurrent_policy,
                                'hidden_size': all_args.hidden_size,
                                'recurrent_N': all_args.recurrent_N,
                                'attn': all_args.attn,                                
                                'attn_size': all_args.attn_size,
                                'attn_N': all_args.attn_N,
                                'attn_heads': all_args.attn_heads,
                                'dropout': all_args.dropout,
                                'use_average_pool': all_args.use_average_pool,
                                'use_common_layer':all_args.use_common_layer,
                                'use_feature_normlization':all_args.use_feature_normlization,
                                'use_feature_popart':all_args.use_feature_popart,
                                'use_orthogonal':all_args.use_orthogonal,
                                'layer_N':all_args.layer_N,
                                'use_ReLU':all_args.use_ReLU,
                                'use_cat_self': all_args.use_cat_self
                                },
                    device = device)
            else:       
                ac = torch.load(str(all_args.model_dir) + "/agent"+ str(agent_id) + "_model.pt")['model']
            ac.to(device)
            # algorithm
            agent = PPO(ac,
                all_args.clip_param,
                all_args.ppo_epoch,
                all_args.num_mini_batch,
                all_args.data_chunk_length,
                all_args.value_loss_coef,
                all_args.entropy_coef,
                lr=all_args.lr,
                eps=all_args.eps,
                weight_decay=all_args.weight_decay,
                max_grad_norm=all_args.max_grad_norm,
                use_max_grad_norm=all_args.use_max_grad_norm,
                use_clipped_value_loss= all_args.use_clipped_value_loss,
                use_common_layer=all_args.use_common_layer,
                use_huber_loss=all_args.use_huber_loss,
                huber_delta=all_args.huber_delta,
                use_popart=all_args.use_popart,
                use_value_active_masks=all_args.use_value_active_masks,
                device=device)
                            
            actor_critic.append(ac)
            agents.append(agent) 
            
        #replay buffer
        rollouts = SharedRolloutStorage(num_agents,
                    all_args.episode_length, 
                    all_args.n_rollout_threads,
                    envs.observation_space[0],
                    envs.share_observation_space[0], 
                    envs.action_space[0],
                    all_args.hidden_size)
    
    # reset env 
    reset_choose = np.ones(all_args.n_rollout_threads)==1.0 
    obs, share_obs, available_actions = envs.reset(reset_choose)

    # replay buffer  
    use_obs = obs.copy()
    use_share_obs = share_obs.copy()
    use_available_actions = available_actions.copy()

    # run
    start = time.time()
    episodes = int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    turn_obs = np.zeros((all_args.n_rollout_threads, num_agents, *rollouts.obs.shape[3:])).astype(np.float32)
    turn_share_obs = np.zeros((all_args.n_rollout_threads, num_agents, *rollouts.share_obs.shape[3:])).astype(np.float32)
    turn_available_actions = np.zeros((all_args.n_rollout_threads, num_agents, *rollouts.available_actions.shape[3:])).astype(np.float32)
    turn_values =  np.zeros((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_actions = np.zeros((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    env_actions = np.ones((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)*(-1.0)
    turn_action_log_probs = np.zeros((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_recurrent_hidden_states = np.zeros((all_args.n_rollout_threads, num_agents, *rollouts.recurrent_hidden_states.shape[3:])).astype(np.float32)
    turn_recurrent_hidden_states_critic = np.zeros((all_args.n_rollout_threads, num_agents, *rollouts.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)
    turn_masks = np.ones((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_active_masks = np.ones((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_bad_masks = np.ones((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_rewards_since_last_action = np.zeros((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_rewards = np.zeros((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)

    for episode in range(episodes):
        if all_args.use_linear_lr_decay:# decrease learning rate linearly
            if all_args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, all_args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, all_args.lr)          
        scores = []          
        for step in range(all_args.episode_length):
            # Sample actions
            reset_choose = np.zeros(all_args.n_rollout_threads)==1.0          
            with torch.no_grad():                  
                for current_agent_id in range(num_agents):
                    env_actions[:,current_agent_id] = np.ones((all_args.n_rollout_threads, 1)).astype(np.float32)*(-1.0)                                                               
                    choose = np.any(use_available_actions[:,current_agent_id]==1,axis=1) 
                    if ~np.any(choose):
                        reset_choose = np.ones(all_args.n_rollout_threads)==1.0
                        break                 
                    if all_args.share_policy:
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
                    
                    # truly used value
                    use_obs = obs.copy()
                    use_share_obs = share_obs.copy()
                    use_available_actions = available_actions.copy()
                    
                    # rearrange reward
                    turn_rewards_since_last_action[choose] += reward[choose]                    
                    turn_rewards[choose, current_agent_id] = turn_rewards_since_last_action[choose, current_agent_id].copy()
                    turn_rewards_since_last_action[choose, current_agent_id] = 0.0
                    
                    # done==True env

                    # deal with reset_choose
                    reset_choose[done==True] = np.ones((done==True).sum(), dtype=bool)

                    # deal with all agents 
                    use_available_actions[done==True] = np.zeros(((done==True).sum(), num_agents, *rollouts.available_actions.shape[3:])).astype(np.float32)
                    turn_masks[done==True] = np.zeros(((done==True).sum(), num_agents, 1)).astype(np.float32)
                    turn_recurrent_hidden_states[done==True] = np.zeros(((done==True).sum(), num_agents, *rollouts.recurrent_hidden_states.shape[3:])).astype(np.float32)
                    turn_recurrent_hidden_states_critic[done==True] = np.zeros(((done==True).sum(), num_agents, *rollouts.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)
        
                    # deal with current agent
                    turn_active_masks[done==True, current_agent_id] = np.ones(((done==True).sum(), 1)).astype(np.float32)
                    
                    # deal with left agents
                    left_agent_id = current_agent_id + 1
                    left_agents_num = num_agents - left_agent_id
                    turn_active_masks[done==True, left_agent_id:] = np.zeros(((done==True).sum(), left_agents_num, 1)).astype(np.float32)
                    turn_rewards[done==True, left_agent_id:] = turn_rewards_since_last_action[done==True, left_agent_id:]
                    turn_rewards_since_last_action[done==True, left_agent_id:] = np.zeros(((done==True).sum(), left_agents_num, 1)).astype(np.float32)
                    # other variables use what at last time, action will be useless.
                    turn_values[done==True, left_agent_id:] = np.zeros(((done==True).sum(), left_agents_num, 1)).astype(np.float32)
                    turn_obs[done==True, left_agent_id:] = use_obs[done==True,left_agent_id:]                                      
                    turn_share_obs[done==True, left_agent_id:] = use_share_obs[done==True,left_agent_id:]                             
                   
                    # done==False env
                    # deal with current agent
                    turn_masks[done==False, current_agent_id] = np.ones(((done==False).sum(), 1)).astype(np.float32)
                    turn_active_masks[done==False, current_agent_id] = np.ones(((done==False).sum(), 1)).astype(np.float32)

                    # done==None
                    # pass

                    for i in range(all_args.n_rollout_threads):
                        if done[i]:
                            if 'score' in infos[i].keys():
                                scores.append(infos[i]['score'])
                     
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
                                turn_active_masks,
                                turn_available_actions)
                            
            # env reset
            obs, share_obs, available_actions = envs.reset(reset_choose)

            use_obs[reset_choose] = obs[reset_choose]
            use_share_obs[reset_choose] = share_obs[reset_choose]
            use_available_actions[reset_choose] = available_actions[reset_choose]
                    
        rollouts.share_obs[-1] = use_share_obs.copy()
        rollouts.obs[-1] = use_obs.copy()
        rollouts.available_actions[-1] = use_available_actions.copy()        
                     
        if all_args.share_policy:
            # compute returns
            with torch.no_grad(): 
                actor_critic.eval()                
                next_value, _, _ = actor_critic.get_value(torch.FloatTensor(np.concatenate(rollouts.share_obs[-1])), 
                                                        torch.FloatTensor(np.concatenate(rollouts.obs[-1])), 
                                                        torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states[-1])),
                                                        torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states_critic[-1])),
                                                        torch.FloatTensor(np.concatenate(rollouts.masks[-1])))
                next_values = np.array(np.split(next_value.detach().cpu().numpy(), all_args.n_rollout_threads))
                rollouts.shared_compute_returns(next_values, 
                                                all_args.use_gae, 
                                                all_args.gamma,
                                                all_args.gae_lambda, 
                                                all_args.use_proper_time_limits,
                                                all_args.use_popart,
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
                                    all_args.use_gae, 
                                    all_args.gamma,
                                    all_args.gae_lambda, 
                                    all_args.use_proper_time_limits,
                                    all_args.use_popart,
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
        rollouts.chooseafter_update()
        
        # post process
        total_num_steps = (episode + 1) * all_args.episode_length * all_args.n_rollout_threads
        
        # save model
        if (episode % all_args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if all_args.share_policy:
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
        if episode % all_args.log_interval == 0:
            end = time.time()
            print("\n Algos {} Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(all_args.algorithm_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        all_args.num_env_steps,
                        int(total_num_steps / (end - start))))
            if all_args.share_policy:
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

            if all_args.env_name == "Hanabi":  
                if len(scores)>0: 
                    wandb.log({'score': np.mean(scores)}, step=total_num_steps)
                    print("mean score is {}.".format(np.mean(scores)))
                else:
                    wandb.log({'score': 0}, step=total_num_steps)
                    print("Can not access mean score.")
                               
        if episode % all_args.eval_interval == 0 and all_args.eval:
            eval_finish = False
            eval_reset_choose = np.ones(all_args.n_eval_rollout_threads)==1.0 
            eval_scores = []
            eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)
            eval_actions = np.ones((all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)*(-1.0)
            eval_recurrent_hidden_states = np.zeros((all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros((all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
            
            while True: 
                if eval_finish:
                    break              
                for agent_id in range(num_agents):
                    eval_actions[:, agent_id] = np.ones((all_args.n_eval_rollout_threads, 1)).astype(np.float32)*(-1.0) 
                    eval_choose = np.any(eval_available_actions[:, agent_id]==1,axis=1) 
                    
                    if ~np.any(eval_choose):
                        eval_finish = True
                        break
                    
                    if all_args.share_policy:
                        actor_critic.eval()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = actor_critic.act(torch.FloatTensor(eval_share_obs[eval_choose,agent_id]), 
                            torch.FloatTensor(eval_obs[eval_choose,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states[eval_choose,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states_critic[eval_choose,agent_id]),
                            torch.FloatTensor(eval_masks[eval_choose,agent_id]),
                            torch.FloatTensor(eval_available_actions[eval_choose,agent_id]),
                            deterministic=True)
                    else:
                        actor_critic[agent_id].eval()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = actor_critic[agent_id].act(torch.FloatTensor(eval_share_obs[eval_choose,agent_id]), 
                            torch.FloatTensor(eval_obs[eval_choose,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states[eval_choose,agent_id]), 
                            torch.FloatTensor(eval_recurrent_hidden_states_critic[eval_choose,agent_id]),
                            torch.FloatTensor(eval_masks[eval_choose,agent_id]),
                            torch.FloatTensor(eval_available_actions[eval_choose,agent_id]),
                            deterministic=True)

                    eval_actions[eval_choose,agent_id] = eval_action.detach().cpu().numpy()
                    eval_recurrent_hidden_states[eval_choose,agent_id] = eval_recurrent_hidden_state.detach().cpu().numpy()
                    eval_recurrent_hidden_states_critic[eval_choose,agent_id] = eval_recurrent_hidden_state_critic.detach().cpu().numpy()
                        
                    # Obser reward and next obs
                    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)

                    eval_available_actions[eval_dones==True] = np.zeros(((eval_dones==True).sum(), num_agents, *rollouts.available_actions.shape[3:])).astype(np.float32)   

                    for i in range(all_args.n_eval_rollout_threads):
                        if eval_dones[i]:
                            if 'score' in eval_infos[i].keys():
                                eval_scores.append(eval_infos[i]['score'])
            
            wandb.log({'eval_score': np.mean(eval_scores)}, step=total_num_steps)
            print("eval mean score is {}.".format(np.mean(eval_scores)))
    
    envs.close()
    if all_args.eval:
        eval_envs.close()
    run.finish()

if __name__ == "__main__":
    main(sys.argv[1:])
