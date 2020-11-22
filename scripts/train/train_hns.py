#!/usr/bin/env python
import sys
import os
import time
import copy
import glob
import shutil
import numpy as np
import wandb
from tensorboardX import SummaryWriter
import socket
import setproctitle
from functools import reduce
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from utils.util import update_linear_schedule, MultiDiscrete
from utils.shared_buffer import SharedReplayBuffer

from envs.hns.HNS_Env import HNSEnv
from envs.env_wrappers import ShareSubprocVecEnv, ChooseSubprocVecEnv, ShareDummyVecEnv, ChooseDummyVecEnv


def make_parallel_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "HideAndSeek" or all_args.env_name == "BlueprintConstruction" or all_args.env_name == "BoxLocking":
                env = HNSEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "HideAndSeek" or all_args.env_name == "BlueprintConstruction" or all_args.env_name == "BoxLocking":
                env = HNSEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='quadrant', help="Which scenario to run on")
    parser.add_argument('--floor_size', type=float,
                        default=6.0, help="size of floor")

    # transfer task
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument('--num_boxes', type=int,
                        default=4, help="number of boxes")
    parser.add_argument("--task_type", type=str, default='all')
    parser.add_argument("--objective_placement", type=str, default='center')

    # hide and seek task
    parser.add_argument("--num_seekers", type=int,
                        default=1, help="number of seekers")
    parser.add_argument("--num_hiders", type=int,
                        default=1, help="number of hiders")
    parser.add_argument("--num_ramps", type=int,
                        default=1, help="number of ramps")
    parser.add_argument("--num_food", type=int,
                        default=0, help="number of food")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (
            all_args.recurrent_policy or all_args.naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.recurrent_policy and all_args.naive_recurrent_policy) == False, (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if "mappo" in all_args.algorithm_name:
        if all_args.use_single_network:
            from algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
            from algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        else:
            from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
    elif "mappg" in all_args.algorithm_name:
        if all_args.use_single_network:
            from algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
            from algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            from algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
            from algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
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

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_parallel_env(all_args)
    if all_args.eval:
        eval_envs = make_eval_env(all_args)

    if all_args.env_name == "HideAndSeek":
        num_seekers = all_args.num_seekers
        num_hiders = all_args.num_hiders
        num_agents = num_seekers + num_hiders
    else:
        num_agents = all_args.num_agents

    # Policy network
    if all_args.use_obs_instead_of_state:
        cat_self = False
    else:
        cat_self = True
    if all_args.share_policy:
        if all_args.use_centralized_V:
            share_observation_space = envs.share_observation_space[0]
        else:
            share_observation_space = envs.observation_space[0]
        
        if all_args.model_dir == None or all_args.model_dir == "":
            policy = Policy(all_args,
                            envs.observation_space[0],
                            share_observation_space,
                            envs.action_space[0],
                            device=device,
                            cat_self=cat_self)
        else:
            policy = torch.load(
                str(all_args.model_dir) + "/agent_model.pt")['model']

        # algorithm
        trainer = TrainAlgo(all_args, policy, device=device)
        buffer = SharedReplayBuffer(all_args,
                                    num_agents,
                                    envs.observation_space[0],
                                    share_observation_space,
                                    envs.action_space[0])
    else:
        trainer = []
        for agent_id in range(num_agents):
            if all_args.use_centralized_V:
                share_observation_space = envs.share_observation_space[agent_id]
            else:
                share_observation_space = envs.observation_space[agent_id]

            if all_args.model_dir == None or all_args.model_dir == "":
                po = Policy(all_args,
                            envs.observation_space[agent_id],
                            share_observation_space,
                            envs.action_space[agent_id],
                            device=device,
                            cat_self=cat_self)
            else:
                po = torch.load(str(all_args.model_dir) +
                                "/agent" + str(agent_id) + "_model.pt")['model']
            tr = TrainAlgo(all_args, po, device=device)
            trainer.append(tr)

        # replay buffer
        buffer = SharedReplayBuffer(all_args,
                                    num_agents,
                                    envs.observation_space[agent_id],
                                    share_observation_space,
                                    envs.action_space[agent_id])

    # reset env
    obs, share_obs, _ = envs.reset()

    # replay buffer
    if not all_args.use_centralized_V:
        share_obs = obs
    buffer.share_obs[0] = share_obs.copy()
    buffer.obs[0] = obs.copy()
    buffer.recurrent_hidden_states = np.zeros(
        buffer.recurrent_hidden_states.shape).astype(np.float32)
    buffer.recurrent_hidden_states_critic = np.zeros(
        buffer.recurrent_hidden_states_critic.shape).astype(np.float32)

    # run
    start = time.time()
    episodes = int(
        all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    for episode in range(episodes):
        if all_args.use_linear_lr_decay:  # decrease learning rate linearly
            if all_args.share_policy:
                trainer.policy.lr_decay(episode, episodes)
            else:
                for agent_id in range(num_agents):
                    trainer[agent_id].policy.lr_decay(episode, episodes)
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

        for step in range(all_args.episode_length):
            # Sample actions
            with torch.no_grad():
                if all_args.share_policy:
                    trainer.prep_rollout()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                        = trainer.policy.act(torch.FloatTensor(np.concatenate(buffer.share_obs[step])),
                                           torch.FloatTensor(
                                               np.concatenate(buffer.obs[step])),
                                           torch.FloatTensor(np.concatenate(
                                               buffer.recurrent_hidden_states[step])),
                                           torch.FloatTensor(np.concatenate(
                                               buffer.recurrent_hidden_states_critic[step])),
                                           torch.FloatTensor(np.concatenate(buffer.masks[step])))
                    # [envs, agents, dim]
                    values = np.array(
                        np.split(value.detach().cpu().numpy(), all_args.n_rollout_threads))
                    actions = np.array(
                        np.split(action.detach().cpu().numpy(), all_args.n_rollout_threads))
                    action_log_probs = np.array(
                        np.split(action_log_prob.detach().cpu().numpy(), all_args.n_rollout_threads))
                    recurrent_hidden_statess = np.array(np.split(
                        recurrent_hidden_states.detach().cpu().numpy(), all_args.n_rollout_threads))
                    recurrent_hidden_statess_critic = np.array(np.split(
                        recurrent_hidden_states_critic.detach().cpu().numpy(), all_args.n_rollout_threads))
                else:
                    values = []
                    actions = []
                    action_log_probs = []
                    recurrent_hidden_statess = []
                    recurrent_hidden_statess_critic = []

                    for agent_id in range(num_agents):
                        trainer[agent_id].prep_rollout()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                            = trainer[agent_id].policy.act(torch.FloatTensor(buffer.share_obs[step, :, agent_id]),
                                                         torch.FloatTensor(
                                                             buffer.obs[step, :, agent_id]),
                                                         torch.FloatTensor(
                                                             buffer.recurrent_hidden_states[step, :, agent_id]),
                                                         torch.FloatTensor(
                                                             buffer.recurrent_hidden_states_critic[step, :, agent_id]),
                                                         torch.FloatTensor(buffer.masks[step, :, agent_id]))

                        values.append(value.detach().cpu().numpy())
                        actions.append(action.detach().cpu().numpy())
                        action_log_probs.append(
                            action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(
                            recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(
                            recurrent_hidden_states_critic.detach().cpu().numpy())

                    values = np.array(values).transpose(1, 0, 2)
                    actions = np.array(actions).transpose(1, 0, 2)
                    action_log_probs = np.array(
                        action_log_probs).transpose(1, 0, 2)
                    recurrent_hidden_statess = np.array(
                        recurrent_hidden_statess).transpose(1, 0, 2)
                    recurrent_hidden_statess_critic = np.array(
                        recurrent_hidden_statess_critic).transpose(1, 0, 2)

            # Obser reward and next obs
            obs, share_obs, rewards, dones, infos, _ = envs.step(actions)
            if not all_args.use_centralized_V:
                share_obs = obs
            if len(rewards.shape) < 3:
                rewards = rewards[:, :, np.newaxis]

            # insert data in buffer
            recurrent_hidden_statess[dones == True] = np.zeros(
                ((dones == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
            recurrent_hidden_statess_critic[dones == True] = np.zeros(
                ((dones == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
            masks = np.ones((all_args.n_rollout_threads,
                             num_agents, 1)).astype(np.float32)
            masks[dones == True] = np.zeros(
                ((dones == True).sum(), num_agents, 1)).astype(np.float32)

            for i in range(all_args.n_rollout_threads):
                if dones[i]:
                    if "discard_episode" in infos[i].keys():
                        if infos[i]['discard_episode']:
                            discard_episode += 1
                        else:
                            trials += 1
                    else:
                        trials += 1
                    # get info to tensorboard
                    if all_args.env_name == "HideAndSeek":
                        if all_args.num_boxes > 0:
                            if 'max_box_move_prep' in infos[i].keys():
                                max_box_move_prep.append(
                                    infos[i]['max_box_move_prep'])
                            if 'max_box_move' in infos[i].keys():
                                max_box_move.append(infos[i]['max_box_move'])
                            if 'num_box_lock_prep' in infos[i].keys():
                                num_box_lock_prep.append(
                                    infos[i]['num_box_lock_prep'])
                            if 'num_box_lock' in infos[i].keys():
                                num_box_lock.append(infos[i]['num_box_lock'])
                        if all_args.num_ramps > 0:
                            if 'max_ramp_move_prep' in infos[i].keys():
                                max_ramp_move_prep.append(
                                    infos[i]['max_ramp_move_prep'])
                            if 'max_ramp_move' in infos[i].keys():
                                max_ramp_move.append(infos[i]['max_ramp_move'])
                            if 'num_ramp_lock_prep' in infos[i].keys():
                                max_ramp_move.append(
                                    infos[i]['num_ramp_lock_prep'])
                            if 'num_ramp_lock' in infos[i].keys():
                                max_ramp_move.append(infos[i]['num_ramp_lock'])
                        if all_args.num_food > 0:
                            if 'food_eaten' in infos[i].keys():
                                food_eaten.append(infos[i]['food_eaten'])
                            if 'food_eaten_prep' in infos[i].keys():
                                food_eaten_prep.append(
                                    infos[i]['food_eaten_prep'])
                    if all_args.env_name == "BlueprintConstruction" or all_args.env_name == "BoxLocking":
                        if "success" in infos[i].keys():
                            if infos[i]['success']:
                                success += 1
                        if "lock_rate" in infos[i].keys():
                            lock_rate.append(infos[i]['lock_rate'])
                        if "activated_sites" in infos[i].keys():
                            activated_sites.append(infos[i]['activated_sites'])

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
                trainer.prep_rollout()
                next_value = trainer.policy.get_value(torch.FloatTensor(np.concatenate(buffer.share_obs[-1])),
                                                    torch.FloatTensor(np.concatenate(buffer.recurrent_hidden_states_critic[-1])),
                                                    torch.FloatTensor(np.concatenate(buffer.masks[-1])))
                next_values = np.array(
                    np.split(next_value.detach().cpu().numpy(), all_args.n_rollout_threads))
                buffer.shared_compute_returns(next_values,
                                              all_args.use_gae,
                                              all_args.gamma,
                                              all_args.gae_lambda,
                                              all_args.use_proper_time_limits,
                                              all_args.use_popart,
                                              trainer.value_normalizer)
            # update network
            trainer.prep_training()
            value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = trainer.shared_update(
                buffer)

        else:
            value_losses = []
            action_losses = []
            dist_entropies = []
            actor_grad_norms = []
            critic_grad_norms = []
            joint_losses = []
            joint_grad_norms = []

            for agent_id in range(num_agents):
                with torch.no_grad():
                    trainer[agent_id].prep_rollout()
                    next_value = trainer[agent_id].policy.get_value(torch.FloatTensor(buffer.share_obs[-1, :, agent_id]),
                                                                        torch.FloatTensor(
                                                                            buffer.obs[-1, :, agent_id]),
                                                                        torch.FloatTensor(
                        buffer.recurrent_hidden_states[-1, :, agent_id]),
                        torch.FloatTensor(
                            buffer.recurrent_hidden_states_critic[-1, :, agent_id]),
                        torch.FloatTensor(buffer.masks[-1, :, agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    buffer.single_compute_returns(agent_id,
                                                  next_value,
                                                  all_args.use_gae,
                                                  all_args.gamma,
                                                  all_args.gae_lambda,
                                                  all_args.use_proper_time_limits,
                                                  all_args.use_popart,
                                                  trainer[agent_id].value_normalizer)
                # update network
                trainer[agent_id].prep_training()
                value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = trainer[agent_id].single_update(
                    agent_id, buffer)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                actor_grad_norms.append(actor_grad_norm)
                critic_grad_norms.append(critic_grad_norm)
                joint_losses.append(joint_loss)
                joint_grad_norms.append(joint_grad_norm)

        # clean the buffer and reset
        buffer.after_update()

        # post process
        total_num_steps = (episode + 1) * \
            all_args.episode_length * all_args.n_rollout_threads

        # save model
        # save for every interval-th episode or for the last epoch
        if (episode % all_args.save_interval == 0 or episode == episodes - 1):
            if all_args.share_policy:
                torch.save({
                    'model': trainer.policy
                },
                    str(run_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):
                    torch.save({
                        'model': trainer[agent_id].policy
                    },
                        str(run_dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % all_args.log_interval == 0:
            end = time.time()
            print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                  .format(all_args.scenario_name,
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
                    wandb.log({"action_loss": action_loss},
                              step=total_num_steps)
                    wandb.log({"dist_entropy": dist_entropy},
                              step=total_num_steps)
                    wandb.log({"actor_grad_norm": actor_grad_norm}, step=total_num_steps)
                    wandb.log({"critic_grad_norm": critic_grad_norm}, step=total_num_steps)
                    if "mappg" in all_args.algorithm_name:
                        wandb.log({"joint_loss": joint_loss}, step=total_num_steps)
                        wandb.log({"joint_grad_norm": joint_grad_norm}, step=total_num_steps)
                else:
                    writter.add_scalars(
                        "value_loss", {"value_loss": value_loss}, total_num_steps)
                    writter.add_scalars(
                        "action_loss", {"action_loss": action_loss}, total_num_steps)
                    writter.add_scalars(
                        "dist_entropy", {"dist_entropy": dist_entropy}, total_num_steps)
                    writter.add_scalars(
                        "grad_norm", {"actor_grad_norm": actor_grad_norm}, total_num_steps)
                    writter.add_scalars(
                        "grad_norm", {"critic_grad_norm": critic_grad_norm}, total_num_steps)
                    if "mappg" in all_args.algorithm_name:
                        writter.add_scalars(
                            "joint_loss", {"joint_loss": joint_loss}, total_num_steps)
                        writter.add_scalars(
                            "joint_grad_norm", {"joint_grad_norm": joint_grad_norm}, total_num_steps)
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " %
                          agent_id + str(value_losses[agent_id]))
                    if all_args.use_wandb:
                        wandb.log(
                            {"agent%i/value_loss" % agent_id: value_losses[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/action_loss" % agent_id: action_losses[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/actor_grad_norm" % agent_id: actor_grad_norms[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/critic_grad_norm" % agent_id: critic_grad_norms[agent_id]}, step=total_num_steps)
                        if "mappg" in all_args.algorithm_name:
                            wandb.log(
                                {"agent%i/joint_loss" % agent_id: joint_losses[agent_id]}, step=total_num_steps)
                            wandb.log(
                                {"agent%i/joint_grad_norm" % agent_id: joint_grad_norms[agent_id]}, step=total_num_steps)
                    else:
                        writter.add_scalars("agent%i/value_loss" % agent_id, {
                                            "agent%i/value_loss" % agent_id: value_losses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/action_loss" % agent_id, {
                                            "agent%i/action_loss" % agent_id: action_losses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/dist_entropy" % agent_id, {
                                            "agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/actor_grad_norm" % agent_id, {
                                            "agent%i/actor_grad_norm" % agent_id: actor_grad_norms[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/critic_grad_norm" % agent_id, {
                                            "agent%i/critic_grad_norm" % agent_id: critic_grad_norms[agent_id]}, total_num_steps)
                        if "mappg" in all_args.algorithm_name:
                            writter.add_scalars("agent%i/joint_loss" % agent_id, {
                                                "agent%i/joint_loss" % agent_id: joint_losses[agent_id]}, total_num_steps)
                            writter.add_scalars("agent%i/joint_grad_norm" % agent_id, {
                                                "agent%i/joint_grad_norm" % agent_id: joint_grad_norms[agent_id]}, total_num_steps)

            if all_args.env_name == "HideAndSeek":
                for hider_id in range(num_hiders):
                    if all_args.use_wandb:
                        wandb.log({'hider%i/average_step_rewards' % hider_id: np.mean(
                            buffer.rewards[:, :, hider_id])}, step=total_num_steps)
                    else:
                        writter.add_scalars('hider%i/average_step_rewards' % hider_id, {'hider%i/average_step_rewards' % hider_id: np.mean(
                            buffer.rewards[:, :, hider_id])}, total_num_steps)
                for seeker_id in range(num_seekers):
                    if all_args.use_wandb:
                        wandb.log({'seeker%i/average_step_rewards' % seeker_id: np.mean(
                            buffer.rewards[:, :, num_hiders+seeker_id])}, step=total_num_steps)
                    else:
                        writter.add_scalars('seeker%i/average_step_rewards' % seeker_id, {'seeker%i/average_step_rewards' % seeker_id: np.mean(
                            buffer.rewards[:, :, num_hiders+seeker_id])}, total_num_steps)

                if all_args.use_wandb:
                    if all_args.num_boxes > 0:
                        if len(max_box_move_prep) > 0:
                            wandb.log({'max_box_move_prep': np.mean(
                                max_box_move_prep)}, step=total_num_steps)
                        if len(max_box_move) > 0:
                            wandb.log({'max_box_move': np.mean(
                                max_box_move)}, step=total_num_steps)
                        if len(num_box_lock_prep) > 0:
                            wandb.log({'num_box_lock_prep': np.mean(
                                num_box_lock_prep)}, step=total_num_steps)
                        if len(num_box_lock) > 0:
                            wandb.log({'num_box_lock': np.mean(
                                num_box_lock)}, step=total_num_steps)
                    if all_args.num_ramps > 0:
                        if len(max_ramp_move_prep) > 0:
                            wandb.log({'max_ramp_move_prep': np.mean(
                                max_ramp_move_prep)}, step=total_num_steps)
                        if len(max_ramp_move) > 0:
                            wandb.log({'max_ramp_move': np.mean(
                                max_ramp_move)}, step=total_num_steps)
                        if len(num_ramp_lock_prep) > 0:
                            wandb.log({'num_ramp_lock_prep': np.mean(
                                num_ramp_lock_prep)}, step=total_num_steps)
                        if len(num_ramp_lock) > 0:
                            wandb.log({'num_ramp_lock': np.mean(
                                num_ramp_lock)}, step=total_num_steps)
                    if all_args.num_food > 0:
                        if len(food_eaten) > 0:
                            wandb.log({'food_eaten': np.mean(
                                food_eaten)}, step=total_num_steps)
                        if len(food_eaten_prep) > 0:
                            wandb.log({'food_eaten_prep': np.mean(
                                food_eaten_prep)}, step=total_num_steps)

            if all_args.env_name == "BoxLocking" or all_args.env_name == "BlueprintConstruction":
                if all_args.share_policy:
                    if all_args.use_wandb:
                        wandb.log({"average_step_rewards": np.mean(
                            buffer.rewards)}, step=total_num_steps)
                    else:
                        writter.add_scalars("average_step_rewards", {
                                            "average_step_rewards": np.mean(buffer.rewards)}, total_num_steps)
                else:
                    for agent_id in range(num_agents):
                        if all_args.use_wandb:
                            wandb.log({"agent%i/average_step_rewards" % agent_id: np.mean(
                                buffer.rewards[:, :, agent_id])}, step=total_num_steps)
                        else:
                            writter.add_scalars("agent%i/average_step_rewards" % agent_id, {
                                                "agent%i/average_step_rewards" % agent_id: np.mean(buffer.rewards[:, :, agent_id])}, total_num_steps)

                if all_args.use_wandb:
                    wandb.log({'discard_episode': discard_episode},
                              step=total_num_steps)
                else:
                    writter.add_scalars('discard_episode', {
                                        'discard_episode': discard_episode}, total_num_steps)

                if trials > 0:
                    print("success rate is {}.".format(success/trials))
                    if all_args.use_wandb:
                        wandb.log({'success_rate': success/trials},
                                  step=total_num_steps)
                    else:
                        writter.add_scalars(
                            'success_rate', {'success_rate': success/trials}, total_num_steps)
                else:
                    if all_args.use_wandb:
                        wandb.log({'success_rate': 0.0}, step=total_num_steps)
                    else:
                        writter.add_scalars(
                            'success_rate', {'success_rate': 0.0}, total_num_steps)

                if len(lock_rate) > 0:
                    if all_args.use_wandb:
                        wandb.log({'lock_rate': np.mean(lock_rate)},
                                  step=total_num_steps)
                    else:
                        writter.add_scalars(
                            'lock_rate', {'lock_rate': np.mean(lock_rate)}, total_num_steps)

                if len(activated_sites) > 0:
                    if all_args.use_wandb:
                        wandb.log({'activated_sites': np.mean(
                            activated_sites)}, step=total_num_steps)
                    else:
                        writter.add_scalars('activated_sites', {
                                            'activated_sites': np.mean(activated_sites)}, total_num_steps)

        # eval
        if episode % all_args.eval_interval == 0 and all_args.eval:
            action_shape = eval_envs.action_space[0].shape
            # hide and seek
            eval_num_box_lock_prep = []
            eval_num_box_lock = []

            # transfer task
            eval_success = 0
            eval_trials = 0
            eval_lock_rate = []
            eval_activated_sites = []
            eval_episode_rewards = 0

            eval_reset_choose = np.ones(all_args.n_eval_rollout_threads) == 1.0

            eval_obs, eval_share_obs, _ = eval_envs.reset(eval_reset_choose)
            if not all_args.use_centralized_V:
                eval_share_obs = eval_obs

            eval_recurrent_hidden_states = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_masks = np.ones(
                (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
            eval_dones = np.zeros(all_args.n_eval_rollout_threads, dtype=bool)

            while True:
                eval_choose = eval_dones == False
                if ~np.any(eval_choose):
                    break
                with torch.no_grad():
                    if all_args.share_policy:
                        eval_actions = np.ones(
                            (all_args.n_eval_rollout_threads, num_agents, action_shape)).astype(np.int) * (-1)
                        trainer.prep_rollout()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer.policy.act(torch.FloatTensor(np.concatenate(eval_share_obs[eval_choose])),
                                                                                                                              torch.FloatTensor(
                                                                                                                                  np.concatenate(eval_obs[eval_choose])),
                                                                                                                              torch.FloatTensor(np.concatenate(
                                                                                                                                  eval_recurrent_hidden_states[eval_choose])),
                                                                                                                              torch.FloatTensor(np.concatenate(
                                                                                                                                  eval_recurrent_hidden_states_critic[eval_choose])),
                                                                                                                              torch.FloatTensor(np.concatenate(
                                                                                                                                  eval_masks[eval_choose])),
                                                                                                                              deterministic=True)

                        eval_actions[eval_choose] = np.array(
                            np.split(eval_action.detach().cpu().numpy(), (eval_choose == True).sum()))
                        eval_recurrent_hidden_states[eval_choose] = np.array(np.split(
                            eval_recurrent_hidden_state.detach().cpu().numpy(), (eval_choose == True).sum()))
                        eval_recurrent_hidden_states_critic[eval_choose] = np.array(np.split(
                            eval_recurrent_hidden_state_critic.detach().cpu().numpy(), (eval_choose == True).sum()))
                    else:
                        eval_actions = []
                        for agent_id in range(num_agents):
                            agent_eval_actions = np.ones(
                                (all_args.n_eval_rollout_threads, action_shape)).astype(np.int) * (-1)
                            trainer[agent_id].prep_rollout()
                            _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer[agent_id].policy.act(torch.FloatTensor(eval_share_obs[eval_choose, agent_id]),
                                                                                                                                            torch.FloatTensor(
                                                                                                                                                eval_obs[eval_choose, agent_id]),
                                                                                                                                            torch.FloatTensor(
                                                                                                                                                eval_recurrent_hidden_states[eval_choose, agent_id]),
                                                                                                                                            torch.FloatTensor(
                                eval_recurrent_hidden_states_critic[eval_choose, agent_id]),
                                torch.FloatTensor(
                                    eval_masks[eval_choose, agent_id]),
                                deterministic=True)
                            agent_eval_actions[eval_choose] = eval_action.detach().cpu().numpy()
                            eval_actions.append(agent_eval_actions)
                            eval_recurrent_hidden_states[eval_choose, agent_id] = eval_recurrent_hidden_state.detach().cpu().numpy()
                            eval_recurrent_hidden_states_critic[eval_choose, agent_id] = eval_recurrent_hidden_state_critic.detach().cpu().numpy()

                        eval_actions = np.array(
                            eval_actions).transpose(1, 0, 2)

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = eval_envs.step(
                    eval_actions)
                if not all_args.use_centralized_V:
                    eval_share_obs = eval_obs

                eval_episode_rewards += eval_rewards

                eval_recurrent_hidden_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
                eval_recurrent_hidden_states_critic[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
                eval_masks = np.ones(
                    (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
                eval_masks[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), num_agents, 1)).astype(np.float32)

                discard_reset_choose = np.zeros(
                    all_args.n_eval_rollout_threads, dtype=bool)
                for i in range(all_args.n_eval_rollout_threads):
                    for agent_id in range(num_agents):
                        if eval_dones[i]:
                            if "discard_episode" in eval_infos[i].keys():
                                if eval_infos[i]['discard_episode']:
                                    discard_reset_choose[i] = True
                                else:
                                    eval_trials += 1
                            else:
                                eval_trials += 1
                            # get info to tensorboard
                            if all_args.env_name == "HideAndSeek":
                                if all_args.num_boxes > 0:
                                    if 'num_box_lock_prep' in eval_infos[i].keys():
                                        eval_num_box_lock_prep.append(
                                            eval_infos[i]['num_box_lock_prep'])
                                    if 'num_box_lock' in eval_infos[i].keys():
                                        eval_num_box_lock.append(
                                            eval_infos[i]['num_box_lock'])
                            if all_args.env_name == "BlueprintConstruction" or all_args.env_name == "BoxLocking":
                                if "success" in eval_infos[i].keys():
                                    if eval_infos[i]['success']:
                                        eval_success += 1
                                if "lock_rate" in eval_infos[i].keys():
                                    eval_lock_rate.append(
                                        eval_infos[i]['lock_rate'])
                                if "activated_sites" in eval_infos[i].keys():
                                    eval_activated_sites.append(
                                        eval_infos[i]['activated_sites'])
                discard_obs, discard_share_obs, _ = eval_envs.reset(
                    discard_reset_choose)

                eval_obs[discard_reset_choose ==
                         True] = discard_obs[discard_reset_choose == True]
                eval_share_obs[discard_reset_choose ==
                               True] = discard_share_obs[discard_reset_choose == True]
                eval_dones[discard_reset_choose == True] = np.zeros(
                    discard_reset_choose.sum(), dtype=bool)

            if all_args.env_name == "HideAndSeek":
                for hider_id in range(num_hiders):
                    if all_args.use_wandb:
                        wandb.log({'hider%i/eval_average_episode_rewards' % hider_id: np.mean(
                            eval_episode_rewards[:, hider_id])}, step=total_num_steps)
                    else:
                        writter.add_scalars('hider%i/eval_average_episode_rewards' % hider_id, {
                                            'hider%i/eval_average_episode_rewards' % hider_id: np.mean(eval_episode_rewards[:, hider_id])}, total_num_steps)
                for seeker_id in range(num_seekers):
                    if all_args.use_wandb:
                        wandb.log({'seeker%i/eval_average_episode_rewards' % seeker_id: np.mean(
                            eval_episode_rewards[:, num_hiders+seeker_id])}, step=total_num_steps)
                    else:
                        writter.add_scalars('seeker%i/eval_average_episode_rewards' % seeker_id, {
                                            'seeker%i/eval_average_episode_rewards' % seeker_id: np.mean(eval_episode_rewards[:, num_hiders+seeker_id])}, total_num_steps)

            if all_args.env_name == "BoxLocking" or all_args.env_name == "BlueprintConstruction":
                if all_args.share_policy:
                    if all_args.use_wandb:
                        wandb.log({"eval_average_episode_rewards": np.mean(
                            eval_episode_rewards)}, step=total_num_steps)
                    else:
                        writter.add_scalars("eval_average_episode_rewards", {
                                            "eval_average_episode_rewards": np.mean(eval_episode_rewards)}, total_num_steps)
                else:
                    for agent_id in range(num_agents):
                        if all_args.use_wandb:
                            wandb.log({"agent%i/eval_average_episode_rewards" % agent_id: np.mean(
                                eval_episode_rewards[:, agent_id])}, step=total_num_steps)
                        else:
                            writter.add_scalars("agent%i/eval_average_episode_rewards" % agent_id, {
                                                "agent%i/eval_average_episode_rewards" % agent_id: np.mean(eval_episode_rewards[:, agent_id])}, total_num_steps)
                if eval_trials > 0:
                    print("eval success rate is {}.".format(
                        eval_success/eval_trials))
                    if all_args.use_wandb:
                        wandb.log({'eval_success_rate': eval_success /
                                   eval_trials}, step=total_num_steps)
                    else:
                        writter.add_scalars('eval_success_rate', {
                                            'eval_success_rate': eval_success/eval_trials}, total_num_steps)
                else:
                    if all_args.use_wandb:
                        wandb.log({'eval_success_rate': 0.0},
                                  step=total_num_steps)
                    else:
                        writter.add_scalars('eval_success_rate', {
                                            'eval_success_rate': 0.0}, total_num_steps)
                if len(eval_lock_rate) > 0:
                    if all_args.use_wandb:
                        wandb.log({'eval_lock_rate': np.mean(
                            eval_lock_rate)}, step=total_num_steps)
                    else:
                        writter.add_scalars('eval_lock_rate', {
                                            'eval_lock_rate': np.mean(eval_lock_rate)}, total_num_steps)
                if len(eval_activated_sites) > 0:
                    if all_args.use_wandb:
                        wandb.log({'eval_activated_sites': np.mean(
                            eval_activated_sites)}, step=total_num_steps)
                    else:
                        writter.add_scalars('eval_activated_sites', {
                                            'eval_activated_sites': np.mean(eval_activated_sites)}, total_num_steps)
    envs.close()
    if all_args.eval:
        eval_envs.close()
    run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
