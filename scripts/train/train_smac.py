#!/usr/bin/env python

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

from envs import StarCraft2Env, get_map_params
from utils.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from algorithm.model import Policy
from utils.shared_storage import SharedRolloutStorage

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
        return ShareDummyVecEnv([get_env_fn(0)])
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
    if args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(args.n_eval_rollout_threads)])

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

    model_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / args.env_name / args.map_name / args.algorithm_name
    if not model_dir.exists():
        os.makedirs(str(model_dir))

    run = wandb.init(config=args, 
            project=args.env_name, 
            entity="yuchao",
            notes=socket.gethostname(),
            name=str(args.algorithm_name) + "_seed" + str(args.seed),
            group=args.map_name,
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
    num_agents = get_map_params(args.map_name)["n_agents"]
    
    # Policy network
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
                                            'use_cat_self': args.use_cat_self
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
                                    'use_cat_self': args.use_cat_self
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
    obs, share_obs, available_actions = envs.reset()
    
    # replay buffer       
    share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, axis=1)    
    rollouts.share_obs[0] = share_obs.copy() 
    rollouts.obs[0] = obs.copy()  
    rollouts.available_actions[0] = available_actions.copy()                
    rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
    rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    
    # run
    start = time.time()
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    
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
            with torch.no_grad():
                if args.share_policy:
                    actor_critic.eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                        = actor_critic.act(torch.FloatTensor(np.concatenate(rollouts.share_obs[step])), 
                                            torch.FloatTensor(np.concatenate(rollouts.obs[step])), 
                                            torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states[step])), 
                                            torch.FloatTensor(np.concatenate(rollouts.recurrent_hidden_states_critic[step])),
                                            torch.FloatTensor(np.concatenate(rollouts.masks[step])),
                                            torch.FloatTensor(np.concatenate(rollouts.available_actions[step])))
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
                                                        torch.FloatTensor(rollouts.masks[step,:,agent_id]),
                                                        torch.FloatTensor(rollouts.available_actions[step,:,agent_id]))
                        
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
            obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)

            # insert data in buffer
            recurrent_hidden_statess[dones==True] = np.zeros(((dones==True).sum(),args.hidden_size)).astype(np.float32)
            recurrent_hidden_statess_critic[dones==True] = np.zeros(((dones==True).sum(),args.hidden_size)).astype(np.float32)
            masks = np.ones((args.n_rollout_threads, num_agents, 1)).astype(np.float32)
            masks[dones==True] = np.zeros(((dones==True).sum(), 1)).astype(np.float32)
                
            bad_masks = []
            active_masks = []
            for info in infos: 
                bad_mask = []  
                active_mask = []             
                for agent_id in range(num_agents): 
                    if info[agent_id]['bad_transition']:              
                        bad_mask.append([0.0])
                    else:
                        bad_mask.append([1.0])
                        
                    if info[agent_id]['active_masks']:              
                        active_mask.append([1.0])
                    else:
                        active_mask.append([0.0])
                bad_masks.append(bad_mask)
                active_masks.append(active_mask)
                                       
            # insert data to buffer
            share_obs = np.expand_dims(share_obs, 1).repeat(num_agents,axis=1)
            rollouts.insert(share_obs, 
                            obs, 
                            recurrent_hidden_statess, 
                            recurrent_hidden_statess_critic, 
                            actions,
                            action_log_probs, 
                            values,
                            rewards, 
                            masks, 
                            bad_masks,
                            active_masks,
                            available_actions)
                                              
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

                if np.sum(incre_battles_game) > 0:
                    wandb.log({"incre_win_rate": np.sum(incre_battles_won)/np.sum(incre_battles_game)}, step=total_num_steps)
                    print("incre win rate is {}.".format(np.sum(incre_battles_won)/np.sum(incre_battles_game)))
                else:
                    wandb.log({"incre_win_rate": 0}, step=total_num_steps)
                last_battles_game = battles_game
                last_battles_won = battles_won

        if episode % args.eval_interval == 0 and args.eval:
            eval_battles_won = 0
            eval_episode = 0
            eval_obs, eval_share_obs, eval_available_actions = eval_env.reset()
            eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(num_agents, axis=1)
            eval_recurrent_hidden_states = np.zeros((args.n_eval_rollout_threads, num_agents, args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros((args.n_eval_rollout_threads, num_agents, args.hidden_size)).astype(np.float32)
            eval_masks = np.ones((args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
            
            while True:         
                if args.share_policy:
                    actor_critic.eval()
                    _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = actor_critic.act(torch.FloatTensor(np.concatenate(eval_share_obs)), 
                                    torch.FloatTensor(np.concatenate(eval_obs)), 
                                    torch.FloatTensor(np.concatenate(eval_recurrent_hidden_states)), 
                                    torch.FloatTensor(np.concatenate(eval_recurrent_hidden_states_critic)),
                                    torch.FloatTensor(np.concatenate(eval_masks)),
                                    torch.FloatTensor(np.concatenate(eval_available_actions)),
                                    deterministic=True)
                    eval_actions = np.array(np.split(eval_action.detach().cpu().numpy(), args.n_eval_rollout_threads))
                else:
                    eval_actions = [] 
                    for agent_id in range(num_agents):
                        actor_critic[agent_id].eval()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = actor_critic[agent_id].act(torch.FloatTensor(eval_share_obs[:,agent_id]), 
                                        torch.FloatTensor(eval_obs[:,agent_id]), 
                                        torch.FloatTensor(eval_recurrent_hidden_states[:,agent_id]), 
                                        torch.FloatTensor(eval_recurrent_hidden_states_critic[:,agent_id]),
                                        torch.FloatTensor(eval_masks[:,agent_id]),
                                        torch.FloatTensor(eval_available_actions[:,agent_id,:]),
                                        deterministic=True)

                        eval_actions.append(eval_action.detach().cpu().numpy())
                        eval_recurrent_hidden_states[:,agent_id] = eval_recurrent_hidden_state.detach().cpu().numpy()
                        eval_recurrent_hidden_states_critic[:,agent_id] = eval_recurrent_hidden_state_critic.detach().cpu().numpy()
                    
                    eval_actions = np.array(eval_actions).transpose(1,0,2)
                    eval_recurrent_hidden_states = np.array(eval_recurrent_hidden_states).transpose(1,0,2)
                    eval_recurrent_hidden_states_critic = np.array(eval_recurrent_hidden_states_critic).transpose(1,0,2)
                        
                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_env.step(eval_actions)
                eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(num_agents, axis=1)
                
                eval_recurrent_hidden_states[eval_dones==True] = np.zeros(((eval_dones==True).sum(),args.hidden_size)).astype(np.float32)
                eval_recurrent_hidden_states_critic[eval_dones==True] = np.zeros(((eval_dones==True).sum(),args.hidden_size)).astype(np.float32)
                eval_masks = np.ones((args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
                eval_masks[eval_dones==True] = np.zeros(((eval_dones==True).sum(), 1)).astype(np.float32)                                  
                
                for eval_i in range(args.n_eval_rollout_threads):
                    if eval_dones[eval_i][0]:
                        eval_episode += 1
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won += 1
                
                if eval_episode >= args.eval_episodes:
                    wandb.log({"eval_win_rate": eval_battles_won/eval_episode}, step=total_num_steps)
                    print("eval win rate is {}.".format(eval_battles_won/eval_episode))
                    break

    envs.close()
    if args.eval:
        eval_env.close()
    run.finish()
if __name__ == "__main__":
    main()