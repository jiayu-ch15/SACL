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

from envs import MGEnv, CleanupEnv, HarvestEnv
from algorithm.ppo import PPO
from algorithm.model import Policy

from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv, SubprocVecEnv2, DummyVecEnv2
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "PrisonerDilemma" or args.env_name == "StagHunt" or args.env_name == "Chicken":
                assert args.num_agents == 2, ("only 2 agents is supported, check the config.py.")
                env = MGEnv(args)
            elif args.env_name == "Cleanup":
                env = CleanupEnv(args)
            elif args.env_name == "Harvest":
                env = HarvestEnv(args)
            elif args.env_name == "single_navigation":
                from utils.make_env import make_env
                assert args.num_agents == 1, ("only 1 agent is supported in single_navigation env, check the config.py.")
                env = make_env(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            # np.random.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_rollout_threads == 1:
        if args.critic_full_obs:
            return DummyVecEnv2([get_env_fn(0)])
        else:
            return DummyVecEnv([get_env_fn(0)])
    else:
        if args.critic_full_obs:
            return SubprocVecEnv2([get_env_fn(i) for i in range(args.n_rollout_threads)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])

def main():
    args = get_config()
    assert args.env_name == "single_navigation" or args.env_name == "single_spread", ("only MPE is supported in this file, check the config.py.")
                
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
     
    #Policy network
    actor_critic = []
    if args.share_policy:
        ac = Policy(envs.observation_space[0].shape, 
                    envs.action_space[0],
                    num_agents = args.num_agents,
                    base_kwargs={'lstm':args.lstm,
                                 'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size})
        ac.to(device)
        for agent_id in range(args.num_agents):
            actor_critic.append(ac)         
    else:
        for agent_id in range(args.num_agents):
            ac = Policy(envs.observation_space[0].shape, 
                      envs.action_space[0],
                      num_agents = args.num_agents,
                      base_kwargs={'lstm':args.lstm,
                                   'naive_recurrent': args.naive_recurrent_policy,
                                   'recurrent': args.recurrent_policy,
                                   'hidden_size': args.hidden_size})
            ac.to(device)
            actor_critic.append(ac) 
          
    agents = []
    rollouts = [] 
    for agent_id in range(args.num_agents):
        # algorithm
        agent = PPO(actor_critic[agent_id],
                   agent_id,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   max_grad_norm=args.max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss)

        #replay buffer
        ro = RolloutStorage(args.num_agents,
                            agent_id,
                            args.episode_length, 
                            args.n_rollout_threads,
                            envs.observation_space[agent_id].shape, 
                            envs.action_space[agent_id],
                            actor_critic[agent_id].recurrent_hidden_state_size)
                       
        
        agents.append(agent)
        rollouts.append(ro)
    
    # reset env 
    if args.critic_full_obs:
        obs, obs_critic = envs.reset()
    else:
        obs = envs.reset()
    # rollout
    for i in range(args.num_agents):
        if len(envs.observation_space[0].shape) == 1:
            if args.critic_full_obs:
                rollouts[i].share_obs[0].copy_(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1)))
            else:
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1)))
            rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
            rollouts[i].recurrent_hidden_states.zero_()
            rollouts[i].recurrent_hidden_states_critic.zero_()
            rollouts[i].recurrent_c_states.zero_()
            rollouts[i].recurrent_c_states_critic.zero_()
        elif len(envs.observation_space[0].shape) == 3:
            if args.critic_full_obs:
                rollouts[i].share_obs[0].copy_(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2] )))
            else:
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2] )))
            rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:,:,:]))
            rollouts[i].recurrent_hidden_states.zero_()
            rollouts[i].recurrent_hidden_states_critic.zero_()
            rollouts[i].recurrent_c_states.zero_()
            rollouts[i].recurrent_c_states_critic.zero_()
        else:
            raise NotImplementedError
        rollouts[i].to(device) 
                   
    
    # run
    coop_num = []
    defect_num = []
    coopdefect_num = []
    collective_return = []
    apple_consumption = []
    waste_cleared = []
    sustainability = []
    fire = []

    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    all_episode = 0

    for episode in range(episodes):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(args.num_agents):
                update_linear_schedule(agents[i].optimizer, 
                                       episode, 
                                       episodes, 
                                       args.lr)

        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            recurrent_c_statess = []
            recurrent_c_statess_critic = []

            with torch.no_grad():
                for i in range(args.num_agents):
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic, recurrent_c_states, recurrent_c_states_critic = actor_critic[i].act(rollouts[i].share_obs[step], 
                                                        rollouts[i].obs[step], 
                                                        rollouts[i].recurrent_hidden_states[step], 
                                                        rollouts[i].recurrent_hidden_states_critic[step],
                                                        rollouts[i].recurrent_c_states[step], 
                                                        rollouts[i].recurrent_c_states_critic[step], 
                                                        rollouts[i].masks[step])
                    values.append(value)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)
                    recurrent_hidden_statess.append(recurrent_hidden_states)
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic)
                    recurrent_c_statess.append(recurrent_c_states)
                    recurrent_c_statess_critic.append(recurrent_c_states_critic)
            
            # rearrange action           
            actions_env = []
            for i in range(args.n_rollout_threads):
                one_hot_action_env = []
                for k in range(args.num_agents):
                    one_hot_action = np.zeros(envs.action_space[0].n)
                    one_hot_action[actions[k][i]] = 1
                    one_hot_action_env.append(one_hot_action)
                actions_env.append(one_hot_action_env)
            
            
            # Obser reward and next obs
            if args.critic_full_obs:
                obs, obs_critic, reward, done, infos = envs.step(actions_env)
            else:
                obs, reward, done, infos = envs.step(actions_env)

            # If done then clean the history of observations.
            '''
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            '''
            # insert data in buffer
            masks = []
            bad_masks = []
            masks_critic = []
            bad_masks_critic = []
            for i in range(args.num_agents):
                mask = []
                bad_mask = []
                for done_ in done:  
                    if done_[i]:              
                        mask.append([0.0])
                    else:
                        mask.append([1.0])
                        bad_mask.append([1.0])
                masks.append(torch.FloatTensor(mask))
                bad_masks.append(torch.FloatTensor(bad_mask))
                
            #for done_ in done:
                #for i in range(args.num_agents):
                   #if done_[i]:
                       #mask.append([0.0])
                
            
            # masks = torch.ones(args.n_rollout_threads, 1)
            # bad_masks = torch.ones(args.n_rollout_threads, 1)
            
            for i in range(args.num_agents):
                if len(envs.observation_space[0].shape) == 1:
                    if args.critic_full_obs:
                        rollouts[i].insert(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1)), 
                                        torch.tensor(obs[:,i,:]), 
                                        recurrent_hidden_statess[i], 
                                        recurrent_hidden_statess_critic[i],
                                        recurrent_c_statess[i], 
                                        recurrent_c_statess_critic[i], 
                                        actions[i],
                                        action_log_probs[i], 
                                        values[i], 
                                        torch.tensor(reward[:, i].reshape(-1,1)), 
                                        masks[i], 
                                        bad_masks[i])
                    else:
                        rollouts[i].insert(torch.tensor(obs.reshape(args.n_rollout_threads, -1)), 
                                        torch.tensor(obs[:,i,:]), 
                                        recurrent_hidden_statess[i], 
                                        recurrent_hidden_statess_critic[i],
                                        recurrent_c_statess[i], 
                                        recurrent_c_statess_critic[i], 
                                        actions[i],
                                        action_log_probs[i], 
                                        values[i], 
                                        torch.tensor(reward[:, i].reshape(-1,1)), 
                                        masks[i], 
                                        bad_masks[i])
                elif len(envs.observation_space[0].shape) == 3:
                    if args.critic_full_obs:
                        rollouts[i].insert(torch.tensor(obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])), 
                                        torch.tensor(obs[:,i,:,:,:]), 
                                        recurrent_hidden_statess[i], 
                                        recurrent_hidden_statess_critic[i],
                                        recurrent_c_statess[i], 
                                        recurrent_c_statess_critic[i], 
                                        actions[i],
                                        action_log_probs[i], 
                                        values[i], 
                                        torch.tensor(reward[:, i].reshape(-1,1)), 
                                        masks[i], 
                                        bad_masks[i])
                    else:
                        rollouts[i].insert(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2])), 
                                        torch.tensor(obs[:,i,:,:,:]), 
                                        recurrent_hidden_statess[i], 
                                        recurrent_hidden_statess_critic[i], 
                                        recurrent_c_statess[i], 
                                        recurrent_c_statess_critic[i], 
                                        actions[i],
                                        action_log_probs[i], 
                                        values[i], 
                                        torch.tensor(reward[:, i].reshape(-1,1)), 
                                        masks[i], 
                                        bad_masks[i])
                else:
                    raise NotImplementedError

        with torch.no_grad():
            next_values = []
            for i in range(args.num_agents):
                next_value = actor_critic[i].get_value(rollouts[i].share_obs[-1], 
                                                       rollouts[i].obs[-1], 
                                                       rollouts[i].recurrent_hidden_states[-1],
                                                       rollouts[i].recurrent_hidden_states_critic[-1],
                                                       rollouts[i].recurrent_c_states[-1],
                                                       rollouts[i].recurrent_c_states_critic[-1],
                                                       rollouts[i].masks[-1]).detach()
                next_values.append(next_value)

        for i in range(args.num_agents):
            rollouts[i].compute_returns(next_values[i], 
                                        args.use_gae, 
                                        args.gamma,
                                        args.gae_lambda, 
                                        args.use_proper_time_limits)

        # update the network
        value_losses = []
        action_losses = []
        dist_entropies = []
        for i in range(args.num_agents):
            value_loss, action_loss, dist_entropy = agents[i].update(rollouts[i])
            value_losses.append(value_loss)
            action_losses.append(action_loss)
            dist_entropies.append(dist_entropy)
            
        if args.env_name == "PrisonerDilemma" or args.env_name == "Chicken" or args.env_name == "StagHunt":
            for info in infos:
                if 'coop&coop_num' in info.keys():
                    coop_num.append(info['coop&coop_num'])
                if 'defect&defect_num' in info.keys():
                    defect_num.append(info['defect&defect_num'])
                if 'coop&defect_num' in info.keys():
                    coopdefect_num.append(info['coop&defect_num'])
            
            for i in range(args.n_rollout_threads):            
                logger.add_scalars('coop&coop_num_per_episode',
                        {'coop&coop_num_per_episode': coop_num[all_episode]},
                        all_episode)
                logger.add_scalars('defect&defect_num_per_episode',
                        {'defect&defect_num_per_episode': defect_num[all_episode]},
                        all_episode)
                logger.add_scalars('coop&defect_num_per_episode',
                        {'coop&defect_num_per_episode': coopdefect_num[all_episode]},
                        all_episode)
                all_episode += 1
        elif args.env_name == "Cleanup":
            for info in infos:
                if 'collective_return' in info.keys():
                    collective_return.append(info['collective_return'])
                if 'sustainability' in info.keys():
                    sustainability.append(info['sustainability'])
                if 'waste_cleared' in info.keys():
                    waste_cleared.append(info['waste_cleared'])
                if 'fire' in info.keys():
                    fire.append(info['fire'])

            for i in range(args.n_rollout_threads):            
                logger.add_scalars('collective_return',
                    {'collective_return': collective_return[all_episode]},
                    all_episode)
                logger.add_scalars('sustainability',
                    {'sustainability': sustainability[all_episode]/args.num_agents},
                    all_episode)
                logger.add_scalars('waste_cleared',
                    {'waste_cleared': waste_cleared[all_episode]},
                    all_episode)
                logger.add_scalars('fire',
                    {'fire': fire[all_episode]},
                    all_episode)
                all_episode += 1
        elif args.env_name == "Harvest":
            for info in infos:
                if 'collective_return' in info.keys(): 
                    collective_return.append(info['collective_return'])
                if 'sustainability' in info.keys():
                    sustainability.append(info['sustainability'])
                if 'apple_consumption' in info.keys():
                    apple_consumption.append(info['apple_consumption'])
                if 'fire' in info.keys():
                    fire.append(info['fire'])  

            for i in range(args.n_rollout_threads):  
                logger.add_scalars('collective_return',
                    {'collective_return': collective_return[all_episode]},
                    all_episode)
                logger.add_scalars('sustainability',
                    {'sustainability': sustainability[all_episode]/args.num_agents},
                    all_episode)
                logger.add_scalars('apple_consumption',
                    {'apple_consumption': apple_consumption[all_episode]},
                    all_episode)
                logger.add_scalars('fire',
                    {'fire': fire[all_episode]},
                    all_episode)    
                all_episode += 1       
        
        # clean the buffer and reset
        if args.critic_full_obs:
            obs, obs_critic = envs.reset()
        else:
            obs = envs.reset()
        for i in range(args.num_agents):
            if len(envs.observation_space[0].shape) == 1:
                if args.critic_full_obs:
                    rollouts[i].share_obs[0].copy_(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1)))
                else:
                    rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1)))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                rollouts[i].recurrent_hidden_states.zero_()
                rollouts[i].recurrent_hidden_states_critic.zero_()
                rollouts[i].recurrent_c_states.zero_()
                rollouts[i].recurrent_c_states_critic.zero_()
            elif len(envs.observation_space[0].shape) == 3:
                if args.critic_full_obs:
                    rollouts[i].share_obs[0].copy_(torch.tensor(obs_critic.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2] )))
                else:
                    rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2] )))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:,:,:]))
                rollouts[i].recurrent_hidden_states.zero_()
                rollouts[i].recurrent_hidden_states_critic.zero_()
                rollouts[i].recurrent_c_states.zero_()
                rollouts[i].recurrent_c_states_critic.zero_()
            else:
                raise NotImplementedError
            rollouts[i].to(device)

        for i in range(args.num_agents):
            # save for every interval-th episode or for the last epoch
            if (episode % args.save_interval == 0 or episode == episodes - 1):            
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
            for i in range(args.num_agents):
                print("value loss of agent%i: " %i + str(value_losses[i]))
    logger.close()
if __name__ == "__main__":
    main()
