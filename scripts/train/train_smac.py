#!/usr/bin/env python
import sys
import copy
import glob
import os
import time
import shutil
import wandb
from tensorboardX import SummaryWriter
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from utils.shared_buffer import SharedReplayBuffer
from utils.util import update_linear_schedule

from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.starcraft2.smac_maps import get_map_params
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def make_parallel_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
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
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")

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
                   0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
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
                         group=all_args.map_name,
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
        eval_env = make_eval_env(all_args)
    num_agents = get_map_params(all_args.map_name)["n_agents"]

    # Policy network
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
                            device=device)
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
                            device=device)
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
    obs, share_obs, available_actions = envs.reset()

    # replay buffer
    if all_args.use_centralized_V:
        share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, axis=1)
    else:
        share_obs = obs

    buffer.share_obs[0] = share_obs.copy()
    buffer.obs[0] = obs.copy()
    buffer.available_actions[0] = available_actions.copy()
    buffer.recurrent_hidden_states = np.zeros(
        buffer.recurrent_hidden_states.shape).astype(np.float32)
    buffer.recurrent_hidden_states_critic = np.zeros(
        buffer.recurrent_hidden_states_critic.shape).astype(np.float32)

    # run
    start = time.time()
    episodes = int(
        all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    last_battles_game = np.zeros(all_args.n_rollout_threads)
    last_battles_won = np.zeros(all_args.n_rollout_threads)

    for episode in range(episodes):
        if all_args.use_linear_lr_decay:  # decrease learning rate linearly
            if all_args.share_policy:
                trainer.policy.lr_decay(episode, episodes)
            else:
                for agent_id in range(num_agents):
                    trainer[agent_id].policy.lr_decay(episode, episodes)

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
                                           torch.FloatTensor(
                                               np.concatenate(buffer.masks[step])),
                                           torch.FloatTensor(np.concatenate(buffer.available_actions[step])))
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
                                                         torch.FloatTensor(
                                                             buffer.masks[step, :, agent_id]),
                                                         torch.FloatTensor(buffer.available_actions[step, :, agent_id]))

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
            obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)

            dones_env = np.all(dones, axis=1)

            # insert data in buffer
            recurrent_hidden_statess[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
            recurrent_hidden_statess_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
            masks = np.ones((all_args.n_rollout_threads,
                             num_agents, 1)).astype(np.float32)
            masks[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), num_agents, 1)).astype(np.float32)

            active_masks = np.ones(
                (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
            active_masks[dones == True] = np.zeros(
                ((dones == True).sum(), 1)).astype(np.float32)
            active_masks[dones_env == True] = np.ones(
                ((dones_env == True).sum(), num_agents, 1)).astype(np.float32)

            bad_masks = []
            for info in infos:
                bad_mask = []
                active_mask = []
                for agent_id in range(num_agents):
                    if info[agent_id]['bad_transition']:
                        bad_mask.append([0.0])
                    else:
                        bad_mask.append([1.0])
                bad_masks.append(bad_mask)

            # insert data to buffer
            if all_args.use_centralized_V:
                share_obs = np.expand_dims(
                    share_obs, 1).repeat(num_agents, axis=1)
            else:
                share_obs = obs
            buffer.insert(share_obs,
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

        if all_args.share_policy:
            with torch.no_grad():
                trainer.prep_rollout()
                next_value = trainer.policy.get_value(torch.FloatTensor(np.concatenate(buffer.share_obs[-1])),
                                                          torch.FloatTensor(np.concatenate(
                                                              buffer.recurrent_hidden_states_critic[-1])),
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
            value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = trainer.shared_update(buffer)

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
                                                                torch.FloatTensor(buffer.recurrent_hidden_states_critic[-1, :, agent_id]),
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
                actor_critic[agent_id].prep_training()
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
                    wandb.log({"action_loss": action_loss},
                              step=total_num_steps)
                    wandb.log({"dist_entropy": dist_entropy},
                              step=total_num_steps)
                    wandb.log({"actor_grad_norm": actor_grad_norm}, step=total_num_steps)
                    wandb.log({"critic_grad_norm": critic_grad_norm}, step=total_num_steps)
                    if "mappg" in all_args.algorithm_name:
                        wandb.log({"joint_loss": joint_loss}, step=total_num_steps)
                        wandb.log({"joint_grad_norm": joint_grad_norm}, step=total_num_steps)
                    wandb.log({"average_step_rewards": np.mean(
                        buffer.rewards)}, step=total_num_steps)
                else:
                    writter.add_scalars(
                        "value_loss", {"value_loss": value_loss}, total_num_steps)
                    writter.add_scalars(
                        "action_loss", {"action_loss": action_loss}, total_num_steps)
                    writter.add_scalars(
                        "dist_entropy", {"dist_entropy": dist_entropy}, total_num_steps)
                    writter.add_scalars(
                        "actor_grad_norm", {"actor_grad_norm": actor_grad_norm}, total_num_steps)
                    writter.add_scalars(
                        "critic_grad_norm", {"critic_grad_norm": critic_grad_norm}, total_num_steps)
                    if "mappg" in all_args.algorithm_name:
                        writter.add_scalars(
                            "joint_loss", {"joint_loss": joint_loss}, total_num_steps)
                        writter.add_scalars(
                            "joint_grad_norm", {"joint_grad_norm": joint_grad_norm}, total_num_steps)
                    writter.add_scalars("average_step_rewards", {
                                        "average_step_rewards": np.mean(buffer.rewards)}, total_num_steps)
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
                        wandb.log({"agent%i/average_step_rewards" % agent_id: np.mean(
                            buffer.rewards[:, :, agent_id])}, step=total_num_steps)
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
                        writter.add_scalars("agent%i/average_step_rewards" % agent_id, {
                                            "agent%i/average_step_rewards" % agent_id: np.mean(buffer.rewards[:, :, agent_id])}, total_num_steps)
            if all_args.env_name == "StarCraft2":
                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(
                            info[0]['battles_won']-last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(
                            info[0]['battles_game']-last_battles_game[i])

                if np.sum(incre_battles_game) > 0:
                    if all_args.use_wandb:
                        wandb.log({"incre_win_rate": np.sum(
                            incre_battles_won)/np.sum(incre_battles_game)}, step=total_num_steps)
                    else:
                        writter.add_scalars("incre_win_rate", {"incre_win_rate": np.sum(
                            incre_battles_won)/np.sum(incre_battles_game)}, total_num_steps)
                    print("incre win rate is {}.".format(
                        np.sum(incre_battles_won)/np.sum(incre_battles_game)))
                else:
                    if all_args.use_wandb:
                        wandb.log({"incre_win_rate": 0}, step=total_num_steps)
                    else:
                        writter.add_scalars("incre_win_rate", {
                                            "incre_win_rate": 0}, total_num_steps)
                last_battles_game = battles_game
                last_battles_won = battles_won

        if episode % all_args.eval_interval == 0 and all_args.eval:
            eval_battles_won = 0
            eval_episode = 0
            eval_obs, eval_share_obs, eval_available_actions = eval_env.reset()
            if all_args.use_centralized_V:
                eval_share_obs = np.expand_dims(
                    eval_share_obs, 1).repeat(num_agents, axis=1)
            else:
                eval_share_obs = eval_obs
            eval_recurrent_hidden_states = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_masks = np.ones(
                (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)

            while True:
                if all_args.share_policy:
                    trainer.prep_rollout()
                    _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer.policy.act(torch.FloatTensor(np.concatenate(eval_share_obs)),
                                                                                                                          torch.FloatTensor(
                                                                                                                              np.concatenate(eval_obs)),
                                                                                                                          torch.FloatTensor(np.concatenate(
                                                                                                                              eval_recurrent_hidden_states)),
                                                                                                                          torch.FloatTensor(np.concatenate(
                                                                                                                              eval_recurrent_hidden_states_critic)),
                                                                                                                          torch.FloatTensor(
                                                                                                                              np.concatenate(eval_masks)),
                                                                                                                          torch.FloatTensor(np.concatenate(
                                                                                                                              eval_available_actions)),
                                                                                                                          deterministic=True)
                    eval_actions = np.array(
                        np.split(eval_action.detach().cpu().numpy(), all_args.n_eval_rollout_threads))
                    eval_recurrent_hidden_states = np.array(np.split(
                        eval_recurrent_hidden_state.detach().cpu().numpy(), all_args.n_eval_rollout_threads))
                    eval_recurrent_hidden_states_critic = np.array(np.split(
                        eval_recurrent_hidden_state_critic.detach().cpu().numpy(), all_args.n_eval_rollout_threads))
                else:
                    eval_actions = []
                    for agent_id in range(num_agents):
                        trainer[agent_id].prep_rollout()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer[agent_id].policy.act(torch.FloatTensor(eval_share_obs[:, agent_id]),
                                                                                                                                        torch.FloatTensor(
                                                                                                                                            eval_obs[:, agent_id]),
                                                                                                                                        torch.FloatTensor(
                                                                                                                                            eval_recurrent_hidden_states[:, agent_id]),
                                                                                                                                        torch.FloatTensor(
                                                                                                                                            eval_recurrent_hidden_states_critic[:, agent_id]),
                                                                                                                                        torch.FloatTensor(eval_masks[:, agent_id]),
                                                                                                                                        torch.FloatTensor(
                                                                                                                                            eval_available_actions[:, agent_id, :]),
                                                                                                                                        deterministic=True)

                        eval_actions.append(eval_action.detach().cpu().numpy())
                        eval_recurrent_hidden_states[:, agent_id] = eval_recurrent_hidden_state.detach(
                        ).cpu().numpy()
                        eval_recurrent_hidden_states_critic[:, agent_id] = eval_recurrent_hidden_state_critic.detach(
                        ).cpu().numpy()

                    eval_actions = np.array(eval_actions).transpose(1, 0, 2)

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_env.step(
                    eval_actions)
                if all_args.use_centralized_V:
                    eval_share_obs = np.expand_dims(
                        eval_share_obs, 1).repeat(num_agents, axis=1)
                else:
                    eval_share_obs = eval_obs

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_recurrent_hidden_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
                eval_recurrent_hidden_states_critic[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), num_agents, all_args.hidden_size)).astype(np.float32)
                eval_masks = np.ones(
                    (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), num_agents, 1)).astype(np.float32)

                for eval_i in range(all_args.n_eval_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode += 1
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won += 1

                if eval_episode >= all_args.eval_episodes:
                    if all_args.use_wandb:
                        wandb.log({"eval_win_rate": eval_battles_won /
                                   eval_episode}, step=total_num_steps)
                    else:
                        writter.add_scalars("eval_win_rate", {
                                            "eval_win_rate": eval_battles_won/eval_episode}, total_num_steps)
                    print("eval win rate is {}.".format(
                        eval_battles_won/eval_episode))
                    break

    envs.close()
    if all_args.eval:
        eval_env.close()
    run.finish()

    if all_args.use_wandb:
        run.finish()
    else:
        writter.export_scalars_to_json(str(log_dir + '/summary.json'))
        writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
