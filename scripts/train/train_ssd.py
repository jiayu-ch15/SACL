#!/usr/bin/env python
import sys
import os
import time
import copy
import glob
import shutil
import numpy as np
import itertools
import wandb
from tensorboardX import SummaryWriter
import socket
import setproctitle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from algorithm.ppo import PPO
from algorithm.model import Policy
from utils.util import update_linear_schedule
from utils.shared_buffer import SharedReplayBuffer
from utils.separated_buffer import SeparatedReplayBuffer

from envs import HarvestEnv, CleanupEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv

def make_parallel_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "Harvest":
                env = HarvestEnv(all_args)
            elif args.env_name == "Cleanup":
                env = CleanupEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment." )
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "Harvest":
                env = HarvestEnv(all_args)
            elif args.env_name == "Cleanup":
                env = CleanupEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment." )
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int, default=5, help="number of agents")
    parser.add_argument("--share_reward", action='store_true', default=False)
    parser.add_argument("--shape_reward", action='store_true', default=False)
    parser.add_argument("--shape_beta", type=float, default=0.8, help='use how much global reward')

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main():
    parser = get_config() 
    all_args = parse_args(args, parser)

    if all_args.algorithm_name=="rmappo":
        assert (all_args.recurrent_policy or all_args.naive_recurrent_policy), ("check recurrent policy!")  
    elif all_args.algorithm_name=="mappo":
        assert (all_args.recurrent_policy and all_args.naive_recurrent_policy) == False, ("check recurrent policy!")  
    else:
        raise NotImplementedError 
     
    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, ("The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

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

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args, 
                project=all_args.env_name,
                entity=all_args.user_name,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
                group=all_args.share_reward,
                dir=str(run_dir),
                job_type="training",
                reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        log_dir = str(run_dir / 'logs')       
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writter = SummaryWriter(log_dir)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))                  
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_parallel_env(all_args)
    if all_args.eval:
        eval_envs = make_eval_env(all_args)
    num_agents = all_args.num_agents

    # Policy network
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
                                            'use_feature_normalization':all_args.use_feature_normalization,
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
                
        # replay buffer
        buffer = SharedReplayBuffer(num_agents,
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
                ac = Policy(envs.observation_space[agent_id],
                            envs.share_observation_space[agent_id],  
                            envs.action_space[agent_id],
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
                                    'use_feature_normalization':all_args.use_feature_normalization,
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
            
        # replay buffer
        buffer = SharedReplayBuffer(num_agents,
                                    all_args.episode_length, 
                                    all_args.n_rollout_threads,
                                    envs.observation_space[agent_id],
                                    envs.share_observation_space[agent_id],  
                                    envs.action_space[agent_id],
                                    all_args.hidden_size)
    
    # reset env 
    obs = envs.reset()

    # replay buffer
    share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])        
    share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
    buffer.share_obs[0] = share_obs.copy() 
    buffer.obs[0] = obs.copy()               
    buffer.recurrent_hidden_states = np.zeros(buffer.recurrent_hidden_states.shape).astype(np.float32)
    buffer.recurrent_hidden_states_critic = np.zeros(buffer.recurrent_hidden_states_critic.shape).astype(np.float32)

    # run
    start = time.time()
    episodes = int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    for episode in range(episodes):
        if all_args.use_linear_lr_decay:# decrease learning rate linearly
            if all_args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, all_args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, all_args.lr)           

        for step in range(args.episode_length):
            # Sample actions
            with torch.no_grad():
                if all_args.share_policy:
                    actor_critic.eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                        = actor_critic.act(torch.FloatTensor(np.concatenate(buffer.share_obs[step])), 
                                            torch.FloatTensor(np.concatenate(buffer.obs[step])), 
                                            torch.FloatTensor(np.concatenate(buffer.recurrent_hidden_states[step])), 
                                            torch.FloatTensor(np.concatenate(buffer.recurrent_hidden_states_critic[step])),
                                            torch.FloatTensor(np.concatenate(buffer.masks[step])))
                    # [envs, agents, dim]
                    values = np.array(np.split(value.detach().cpu().numpy(),all_args.n_rollout_threads))
                    actions = np.array(np.split(action.detach().cpu().numpy(),all_args.n_rollout_threads))
                    action_log_probs = np.array(np.split(action_log_prob.detach().cpu().numpy(),all_args.n_rollout_threads))
                    recurrent_hidden_statess = np.array(np.split(recurrent_hidden_states.detach().cpu().numpy(),all_args.n_rollout_threads))
                    recurrent_hidden_statess_critic = np.array(np.split(recurrent_hidden_states_critic.detach().cpu().numpy(),all_args.n_rollout_threads))            
                else:
                    values = []
                    actions= []
                    action_log_probs = []
                    recurrent_hidden_statess = []
                    recurrent_hidden_statess_critic = []

                    for agent_id in range(num_agents):
                        actor_critic[agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                            = actor_critic[agent_id].act(torch.FloatTensor(buffer.share_obs[step,:,agent_id]), 
                                                        torch.FloatTensor(buffer.obs[step,:,agent_id]), 
                                                        torch.FloatTensor(buffer.recurrent_hidden_states[step,:,agent_id]), 
                                                        torch.FloatTensor(buffer.recurrent_hidden_states_critic[step,:,agent_id]),
                                                        torch.FloatTensor(buffer.masks[step,:,agent_id]))
                        
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
            obs, rewards, dones, infos = envs.step(actions)

            dones_env = np.all(dones, axis=1) 

            # insert data in buffer
            recurrent_hidden_statess[dones_env==True] = np.zeros(((dones_env==True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
            recurrent_hidden_statess_critic[dones_env==True] = np.zeros(((dones_env==True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
            masks = np.ones((all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
            masks[dones_env==True] = np.zeros(((dones_env==True).sum(), num_agents, 1)).astype(np.float32)
                                       
            # insert data to buffer
            share_obs = obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])
            share_obs = np.expand_dims(share_obs, 1).repeat(num_agents,axis=1)
            buffer.insert(share_obs, 
                            obs, 
                            recurrent_hidden_statess, 
                            recurrent_hidden_statess_critic, 
                            actions,
                            action_log_probs, 
                            values,
                            rewards, 
                            masks)
                           
        if all_args.share_policy: 
            with torch.no_grad():
                actor_critic.eval()                
                next_value, _, _ = actor_critic.get_value(torch.FloatTensor(np.concatenate(buffer.share_obs[-1])), 
                                                        torch.FloatTensor(np.concatenate(buffer.obs[-1])), 
                                                        torch.FloatTensor(np.concatenate(buffer.recurrent_hidden_states[-1])),
                                                        torch.FloatTensor(np.concatenate(buffer.recurrent_hidden_states_critic[-1])),
                                                        torch.FloatTensor(np.concatenate(buffer.masks[-1])))
                next_values = np.array(np.split(next_value.detach().cpu().numpy(), all_args.n_rollout_threads))
                buffer.shared_compute_returns(next_values, 
                                                all_args.use_gae, 
                                                all_args.gamma,
                                                all_args.gae_lambda, 
                                                all_args.use_proper_time_limits,
                                                all_args.use_popart,
                                                agents.value_normalizer)
            # update network
            actor_critic.train()
            value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = agents.shared_update(buffer)
            
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
                    next_value, _, _ = actor_critic[agent_id].get_value(torch.FloatTensor(buffer.share_obs[-1,:,agent_id]), 
                                                torch.FloatTensor(buffer.obs[-1,:,agent_id]), 
                                                torch.FloatTensor(buffer.recurrent_hidden_states[-1,:,agent_id]),
                                                torch.FloatTensor(buffer.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                torch.FloatTensor(buffer.masks[-1,:,agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    buffer.single_compute_returns(agent_id,
                                    next_value, 
                                    all_args.use_gae, 
                                    all_args.gamma,
                                    all_args.gae_lambda, 
                                    all_args.use_proper_time_limits,
                                    all_args.use_popart,
                                    agents[agent_id].value_normalizer)
                # update network
                actor_critic[agent_id].train()
                value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = agents[agent_id].single_update(agent_id, buffer)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                grad_norms.append(grad_norm)
                KL_divlosses.append(KL_divloss)
                ratios.append(ratio)
            
        # clean the buffer and reset
        buffer.after_update()

        # post process
        total_num_steps = (episode + 1) * all_args.episode_length * all_args.n_rollout_threads

        # save model
        if (episode % all_args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if all_args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(run_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
                                }, 
                                str(run_dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % all_args.log_interval == 0:
            end = time.time()
            print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                .format(all_args.map_name,
                        all_args.algorithm_name,
                        all_args.experiment_name,
                        episode, 
                        episodes,
                        total_num_steps,
                        all_args.num_env_steps,
                        int(total_num_steps / (end - start))))
            if all_args.share_policy:
                print("value loss of agent: " + str(value_loss))
                if all_args.use_wandb:
                    wandb.log({"value_loss": value_loss}, step=total_num_steps)
                    wandb.log({"action_loss": action_loss}, step=total_num_steps)
                    wandb.log({"dist_entropy": dist_entropy}, step=total_num_steps)
                    wandb.log({"grad_norm": grad_norm}, step=total_num_steps)
                    wandb.log({"KL_divloss": KL_divloss}, step=total_num_steps)
                    wandb.log({"ratio": ratio}, step=total_num_steps)
                    wandb.log({"average_step_rewards": np.mean(buffer.rewards)}, step=total_num_steps)
                else:
                    writter.add_scalars("value_loss", {"value_loss": value_loss}, total_num_steps)
                    writter.add_scalars("action_loss", {"action_loss": action_loss}, total_num_steps)
                    writter.add_scalars("dist_entropy", {"dist_entropy": dist_entropy}, total_num_steps)
                    writter.add_scalars("grad_norm", {"grad_norm": grad_norm}, total_num_steps)
                    writter.add_scalars("KL_divloss", {"KL_divloss": KL_divloss}, total_num_steps)
                    writter.add_scalars("ratio", {"ratio": ratio}, total_num_steps)
                    writter.add_scalars("average_step_rewards", {"average_step_rewards": np.mean(buffer.rewards)}, total_num_steps)
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " % agent_id + str(value_losses[agent_id]))
                    if all_args.use_wandb:
                        wandb.log({"agent%i/value_loss" % agent_id: value_losses[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/action_loss" % agent_id: action_losses[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/grad_norm" % agent_id: grad_norms[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/KL_divloss" % agent_id: KL_divlosses[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/ratio"% agent_id: ratios[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/average_step_rewards" % agent_id: np.mean(buffer.rewards[:,:,agent_id])}, step=total_num_steps)
                    else:
                        writter.add_scalars("agent%i/value_loss" % agent_id,{"agent%i/value_loss" % agent_id: value_losses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/action_loss" % agent_id,{"agent%i/action_loss" % agent_id: action_losses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/dist_entropy" % agent_id,{"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/grad_norm" % agent_id,{"agent%i/grad_norm" % agent_id: grad_norms[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/KL_divloss" % agent_id,{"agent%i/KL_divloss" % agent_id: KL_divlosses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/ratio"% agent_id,{"agent%i/ratio"% agent_id: ratios[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/average_step_rewards" % agent_id,{"agent%i/average_step_rewards" % agent_id: np.mean(buffer.rewards[:,:,agent_id])}, total_num_steps)

            if args.env_name == "Cleanup":                
                collective_return = []
                sustainability = []
                waste_cleared = []
                fire = []
                for i, info in enumerate(infos):
                    if 'collective_return' in info.keys():
                        collective_return.append(info['collective_return'])                         
                    if 'sustainability' in info.keys():
                        sustainability.append(info['sustainability'])                        
                    if 'waste_cleared' in info.keys():
                        waste_cleared.append(info['waste_cleared'])
                    if 'fire' in info.keys():
                        fire.append(info['fire'])
           
                if all_args.use_wandb:
                    wandb.log({'collective_return': np.mean(collective_return)}, step=total_num_steps)
                    wandb.log({'sustainability': np.mean(sustainability)}, step=total_num_steps)
                    wandb.log({'waste_cleared': np.mean(waste_cleared)}, step=total_num_steps)
                    wandb.log({'fire': np.mean(fire)}, step=total_num_steps)
                else:
                    writter.add_scalars('collective_return',{'collective_return': np.mean(collective_return)},total_num_steps)
                    writter.add_scalars('sustainability',{'sustainability': np.mean(sustainability)},total_num_steps)
                    writter.add_scalars('waste_cleared',{'waste_cleared': np.mean(waste_cleared)},total_num_steps)
                    writter.add_scalars('fire',{'fire': np.mean(fire)},total_num_steps)
                                    
            elif args.env_name == "Harvest":                
                collective_return = []
                sustainability = []
                apple_consumption = []
                fire = []
                for i,info in enumerate(infos):
                    if 'collective_return' in info.keys():
                        collective_return.append(info['collective_return'])                         
                    if 'sustainability' in info.keys():
                        sustainability.append(info['sustainability'])                        
                    if 'apple_consumption' in info.keys():
                        apple_consumption.append(info['apple_consumption'])
                    if 'fire' in info.keys():
                        fire.append(info['fire'])

                if all_args.use_wandb:
                    wandb.log({'collective_return': np.mean(collective_return)}, step=total_num_steps)
                    wandb.log({'sustainability': np.mean(sustainability)}, step=total_num_steps)
                    wandb.log({'apple_consumption': np.mean(apple_consumption)}, step=total_num_steps)
                    wandb.log({'fire': np.mean(fire)}, step=total_num_steps)
                else:
                    writter.add_scalars('collective_return',{'collective_return': np.mean(collective_return)},total_num_steps)
                    writter.add_scalars('sustainability',{'sustainability': np.mean(sustainability)},total_num_steps)
                    writter.add_scalars('apple_consumption',{'apple_consumption': np.mean(apple_consumption)},total_num_steps)
                    writter.add_scalars('fire',{'fire': np.mean(fire)},total_num_steps)
                
    envs.close()
    run.finish()

    if all_args.use_wandb:
        run.finish()
    else:
        writter.export_scalars_to_json(str(log_dir + '/summary.json'))
        writter.close()
if __name__ == "__main__":
    main()
