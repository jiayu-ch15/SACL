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

from envs import MGEnv, CleanupEnv, HarvestEnv, GridWorldEnv, StarCraft2Env, get_map_params
from algorithm.ppo import PPO
from algorithm.model import Policy

from config import get_config, get_config_SC
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
import shutil

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
    args = get_config_SC()

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
    num_agents = get_map_params(args.map_name)["n_agents"]
    #Policy network
    actor_critic = []
    if args.share_policy:
        ac = Policy(envs.observation_space[0], 
                    envs.action_space[0],
                    num_agents = num_agents,
                    base_kwargs={'lstm': args.lstm,
                                 'naive_recurrent': args.naive_recurrent_policy,
                                 'recurrent': args.recurrent_policy,
                                 'hidden_size': args.hidden_size})
        ac.to(device)
        for agent_id in range(num_agents):
            actor_critic.append(ac)         
    else:
        for agent_id in range(num_agents):
            ac = Policy(envs.observation_space[0], 
                      envs.action_space[0],
                      num_agents = num_agents,
                      base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                   'recurrent': args.recurrent_policy,
                                   'hidden_size': args.hidden_size})
            ac.to(device)
            actor_critic.append(ac) 
          
    agents = []
    rollouts = [] 
    for agent_id in range(num_agents):
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
        ro = RolloutStorage(num_agents,
                            agent_id,
                            args.episode_length, 
                            args.n_rollout_threads,
                            envs.observation_space[agent_id], 
                            envs.action_space[agent_id],
                            actor_critic[agent_id].recurrent_hidden_state_size)
                       
        
        agents.append(agent)
        rollouts.append(ro)
    
    # reset env 
    obs, available_actions = envs.reset()
    # rollout
    for i in range(num_agents):
        if len(envs.observation_space[0]) == 1:
            rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1)))
            rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
            rollouts[i].recurrent_hidden_states.zero_()
            rollouts[i].recurrent_hidden_states_critic.zero_()
            rollouts[i].recurrent_c_states.zero_()
            rollouts[i].recurrent_c_states_critic.zero_()
        elif len(envs.observation_space[0]) == 3:
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
    defectcoop_num = []
    gore1_num = []
    gore2_num = []
    hare1_num = []
    hare2_num = []
    collective_return = []
    apple_consumption = []
    waste_cleared = []
    sustainability = []
    fire = []
    battles_won = []
    battles_game = []
    battles_draw = []
    win_rate = []
    restarts = []

    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    all_episode = 0

    for episode in range(episodes):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(num_agents):
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
                for i in range(num_agents):
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic ,recurrent_c_states, recurrent_c_states_critic = actor_critic[i].act(rollouts[i].share_obs[step], 
                    rollouts[i].obs[step], 
                    rollouts[i].recurrent_hidden_states[step], 
                    rollouts[i].recurrent_hidden_states_critic[step],
                    rollouts[i].recurrent_c_states[step], 
                    rollouts[i].recurrent_c_states_critic[step], 
                    rollouts[i].masks[step],
                    available_actions[:,i,:])
                    
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
            bad_masks = []
            for i in range(num_agents):
                mask = []
                bad_mask = []
                for done_ in done:  
                    if done_:              
                        mask.append([0.0])
                        bad_mask.append([1.0])
                    else:
                        mask.append([1.0])
                        bad_mask.append([1.0])
                masks.append(torch.FloatTensor(mask))
                bad_masks.append(torch.FloatTensor(bad_mask))
                            
            for i in range(num_agents):
                if len(envs.observation_space[0]) == 1:
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
                elif len(envs.observation_space[0]) == 3:
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
                                        
        with torch.no_grad():
            next_values = []
            for i in range(num_agents):
                next_value = actor_critic[i].get_value(rollouts[i].share_obs[-1], 
                                                       rollouts[i].obs[-1], 
                                                       rollouts[i].recurrent_hidden_states[-1],
                                                       rollouts[i].recurrent_hidden_states_critic[-1],
                                                       rollouts[i].recurrent_c_states[-1],
                                                       rollouts[i].recurrent_c_states_critic[-1],
                                                       rollouts[i].masks[-1]).detach()
                next_values.append(next_value)

        for i in range(num_agents):
            rollouts[i].compute_returns(next_values[i], 
                                        args.use_gae, 
                                        args.gamma,
                                        args.gae_lambda, 
                                        args.use_proper_time_limits)

        # update the network
        value_losses = []
        action_losses = []
        dist_entropies = []
        for i in range(num_agents):
            value_loss, action_loss, dist_entropy = agents[i].update(rollouts[i])
            value_losses.append(value_loss)
            action_losses.append(action_loss)
            dist_entropies.append(dist_entropy)
             
        if args.env_name == "StarCraft2":
            for info in infos:
                if 'battles_won' in info.keys():
                    logger.add_scalars('battles_won',
                    {'battles_won': info['battles_won']},
                    all_episode) 
                if 'battles_game' in info.keys():
                    logger.add_scalars('battles_game',
                        {'battles_game': info['battles_game']},
                        all_episode)
                    if info['battles_game'] == 0:
                        logger.add_scalars('win_rate',
                        {'win_rate': 0},
                        all_episode)
                    else:
                        logger.add_scalars('win_rate',
                        {'win_rate': info['battles_won']/info['battles_game']},
                        all_episode)
                if 'battles_draw' in info.keys():
                    logger.add_scalars('battles_draw',
                    {'battles_draw': info['battles_draw']},
                    all_episode)
                if 'restarts' in info.keys():
                    logger.add_scalars('restarts',
                        {'restarts': info['restarts']},
                        all_episode)
                all_episode += 1     
        
        # clean the buffer and reset
        obs, available_actions = envs.reset()
        for i in range(num_agents):
            if len(envs.observation_space[0]) == 1:
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1)))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                rollouts[i].recurrent_hidden_states.zero_()
                rollouts[i].recurrent_hidden_states_critic.zero_()
                rollouts[i].recurrent_c_states.zero_()
                rollouts[i].recurrent_c_states_critic.zero_()
                rollouts[i].masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
                rollouts[i].bad_masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
            elif len(envs.observation_space[0]) == 3:
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.n_rollout_threads, -1, envs.observation_space[0][1], envs.observation_space[0][2] )))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:,:,:]))
                rollouts[i].recurrent_hidden_states.zero_()
                rollouts[i].recurrent_hidden_states_critic.zero_()
                rollouts[i].recurrent_c_states.zero_()
                rollouts[i].recurrent_c_states_critic.zero_()
                rollouts[i].masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
                rollouts[i].bad_masks[0].copy_(torch.ones(args.n_rollout_threads, 1))
            else:
                raise NotImplementedError
            rollouts[i].to(device)

        for i in range(num_agents):
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
            for i in range(num_agents):
                print("value loss of agent%i: " %i + str(value_losses[i]))
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
if __name__ == "__main__":
    main()
