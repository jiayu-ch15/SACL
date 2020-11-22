#!/usr/bin/env python
import sys
import os
import time
import copy
import glob
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
from utils.shared_buffer import SharedReplayBuffer
from utils.util import update_linear_schedule

from envs.hanabi.Hanabi_Env import HanabiEnv
from envs.env_wrappers import ChooseSubprocVecEnv, ChooseDummyVecEnv


def make_parallel_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert all_args.num_agents > 1 and all_args.num_agents < 6, (
                    "num_agents can be only between 2-5.")
                env = HanabiEnv(all_args, (all_args.seed + rank * 1000))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
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
                assert all_args.num_agents > 1 and all_args.num_agents < 6, (
                    "num_agents can be only between 2-5.")
                env = HanabiEnv(
                    all_args, (all_args.seed * 50000 + rank * 10000))
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
    parser.add_argument('--hanabi_name', type=str,
                        default='Hanabi-Very-Small', help="Which env to run on")
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

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
                   0] + "/results") / all_args.env_name / all_args.hanabi_name / all_args.algorithm_name / all_args.experiment_name
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
                         group=all_args.hanabi_name,
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
    reset_choose = np.ones(all_args.n_rollout_threads) == 1.0
    obs, share_obs, available_actions = envs.reset(reset_choose)
    if not all_args.use_centralized_V:
        share_obs = obs

    # replay buffer
    use_obs = obs.copy()
    use_share_obs = share_obs.copy()
    use_available_actions = available_actions.copy()

    # run
    start = time.time()
    episodes = int(
        all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    turn_obs = np.zeros((all_args.n_rollout_threads, num_agents,
                         *buffer.obs.shape[3:])).astype(np.float32)
    turn_share_obs = np.zeros(
        (all_args.n_rollout_threads, num_agents, *buffer.share_obs.shape[3:])).astype(np.float32)
    turn_available_actions = np.zeros(
        (all_args.n_rollout_threads, num_agents, *buffer.available_actions.shape[3:])).astype(np.float32)
    turn_values = np.zeros(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_actions = np.zeros(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    env_actions = np.ones(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)*(-1.0)
    turn_action_log_probs = np.zeros(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_recurrent_hidden_states = np.zeros(
        (all_args.n_rollout_threads, num_agents, *buffer.recurrent_hidden_states.shape[3:])).astype(np.float32)
    turn_recurrent_hidden_states_critic = np.zeros(
        (all_args.n_rollout_threads, num_agents, *buffer.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)
    turn_masks = np.ones(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_active_masks = np.ones(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_bad_masks = np.ones(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_rewards_since_last_action = np.zeros(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)
    turn_rewards = np.zeros(
        (all_args.n_rollout_threads, num_agents, 1)).astype(np.float32)

    for episode in range(episodes):
        if all_args.use_linear_lr_decay:  # decrease learning rate linearly
            if all_args.share_policy:
                trainer.policy.lr_decay(episode, episodes)
            else:
                for agent_id in range(num_agents):
                    trainer[agent_id].policy.lr_decay(episode, episodes)

        scores = []
        for step in range(all_args.episode_length):
            # Sample actions
            reset_choose = np.zeros(all_args.n_rollout_threads) == 1.0
            with torch.no_grad():
                for current_agent_id in range(num_agents):
                    env_actions[:, current_agent_id] = np.ones(
                        (all_args.n_rollout_threads, 1)).astype(np.float32)*(-1.0)
                    choose = np.any(
                        use_available_actions[:, current_agent_id] == 1, axis=1)
                    if ~np.any(choose):
                        reset_choose = np.ones(
                            all_args.n_rollout_threads) == 1.0
                        break
                    if all_args.share_policy:
                        trainer.prep_rollout()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = trainer.policy.act(torch.FloatTensor(use_share_obs[choose, current_agent_id]),
                                                                                                                                   torch.FloatTensor(
                                                                                                                                       use_obs[choose, current_agent_id]),
                                                                                                                                   torch.FloatTensor(
                            turn_recurrent_hidden_states[choose, current_agent_id]),
                            torch.FloatTensor(
                                turn_recurrent_hidden_states_critic[choose, current_agent_id]),
                            torch.FloatTensor(
                                turn_masks[choose, current_agent_id]),
                            torch.FloatTensor(use_available_actions[choose, current_agent_id]))
                    else:
                        trainer[current_agent_id].prep_rollout()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = trainer[current_agent_id].policy.act(torch.FloatTensor(use_share_obs[choose, current_agent_id]),
                                                                                                                                                     torch.FloatTensor(
                                                                                                                                                         use_obs[choose, current_agent_id]),
                                                                                                                                                     torch.FloatTensor(
                            turn_recurrent_hidden_states[choose, current_agent_id]),
                            torch.FloatTensor(
                                turn_recurrent_hidden_states_critic[choose, current_agent_id]),
                            torch.FloatTensor(
                                turn_masks[choose, current_agent_id]),
                            torch.FloatTensor(use_available_actions[choose, current_agent_id]))

                    turn_obs[choose, current_agent_id] = use_obs[choose,
                                                                 current_agent_id].copy()
                    turn_share_obs[choose, current_agent_id] = use_share_obs[choose,
                                                                             current_agent_id].copy()
                    turn_available_actions[choose,
                                           current_agent_id] = use_available_actions[choose, current_agent_id].copy()
                    turn_values[choose,
                                current_agent_id] = value.detach().cpu().numpy()
                    turn_actions[choose, current_agent_id] = action.detach().cpu().numpy()
                    env_actions[choose, current_agent_id] = action.detach().cpu().numpy()
                    turn_action_log_probs[choose, current_agent_id] = action_log_prob.detach().cpu().numpy()
                    turn_recurrent_hidden_states[choose, current_agent_id] = recurrent_hidden_states.detach().cpu().numpy()
                    turn_recurrent_hidden_states_critic[choose, current_agent_id] = recurrent_hidden_states_critic.detach().cpu().numpy()

                    obs, share_obs, reward, done, infos, available_actions = envs.step(
                        env_actions)
                    if not all_args.use_centralized_V:
                        share_obs = obs
                    # truly used value
                    use_obs = obs.copy()
                    use_share_obs = share_obs.copy()
                    use_available_actions = available_actions.copy()

                    # rearrange reward
                    turn_rewards_since_last_action[choose] += reward[choose]
                    turn_rewards[choose, current_agent_id] = turn_rewards_since_last_action[choose,
                                                                                            current_agent_id].copy()
                    turn_rewards_since_last_action[choose,
                                                   current_agent_id] = 0.0

                    # done==True env

                    # deal with reset_choose
                    reset_choose[done == True] = np.ones(
                        (done == True).sum(), dtype=bool)

                    # deal with all agents
                    use_available_actions[done == True] = np.zeros(((done == True).sum(
                    ), num_agents, *buffer.available_actions.shape[3:])).astype(np.float32)
                    turn_masks[done == True] = np.zeros(
                        ((done == True).sum(), num_agents, 1)).astype(np.float32)
                    turn_recurrent_hidden_states[done == True] = np.zeros(((done == True).sum(
                    ), num_agents, *buffer.recurrent_hidden_states.shape[3:])).astype(np.float32)
                    turn_recurrent_hidden_states_critic[done == True] = np.zeros(((done == True).sum(
                    ), num_agents, *buffer.recurrent_hidden_states_critic.shape[3:])).astype(np.float32)

                    # deal with current agent
                    turn_active_masks[done == True, current_agent_id] = np.ones(
                        ((done == True).sum(), 1)).astype(np.float32)

                    # deal with left agents
                    left_agent_id = current_agent_id + 1
                    left_agents_num = num_agents - left_agent_id
                    turn_active_masks[done == True, left_agent_id:] = np.zeros(
                        ((done == True).sum(), left_agents_num, 1)).astype(np.float32)
                    turn_rewards[done == True,
                                 left_agent_id:] = turn_rewards_since_last_action[done == True, left_agent_id:]
                    turn_rewards_since_last_action[done == True, left_agent_id:] = np.zeros(
                        ((done == True).sum(), left_agents_num, 1)).astype(np.float32)
                    # other variables use what at last time, action will be useless.
                    turn_values[done == True, left_agent_id:] = np.zeros(
                        ((done == True).sum(), left_agents_num, 1)).astype(np.float32)
                    turn_obs[done == True, left_agent_id:] = use_obs[done ==
                                                                     True, left_agent_id:]
                    turn_share_obs[done == True,
                                   left_agent_id:] = use_share_obs[done == True, left_agent_id:]

                    # done==False env
                    # deal with current agent
                    turn_masks[done == False, current_agent_id] = np.ones(
                        ((done == False).sum(), 1)).astype(np.float32)
                    turn_active_masks[done == False, current_agent_id] = np.ones(
                        ((done == False).sum(), 1)).astype(np.float32)

                    # done==None
                    # pass

                    for i in range(all_args.n_rollout_threads):
                        if done[i]:
                            if 'score' in infos[i].keys():
                                scores.append(infos[i]['score'])

            # insert turn data into buffer
            buffer.chooseinsert(turn_share_obs,
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
            if not all_args.use_centralized_V:
                share_obs = obs

            use_obs[reset_choose] = obs[reset_choose]
            use_share_obs[reset_choose] = share_obs[reset_choose]
            use_available_actions[reset_choose] = available_actions[reset_choose]

        buffer.share_obs[-1] = use_share_obs.copy()
        buffer.obs[-1] = use_obs.copy()
        buffer.available_actions[-1] = use_available_actions.copy()

        if all_args.share_policy:
            # compute returns
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
                actor_critic[agent_id].train()
                value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = agents[agent_id].single_update(
                    agent_id, buffer)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                actor_grad_norms.append(actor_grad_norm)
                critic_grad_norms.append(critic_grad_norm)
                joint_losses.append(joint_loss)
                joint_grad_norms.append(joint_grad_norm)

        # clean the buffer and reset
        buffer.chooseafter_update()

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
            print("\n Env {} Algos {} Exp {} Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                  .format(all_args.hanabi_name,
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
            if all_args.env_name == "Hanabi":
                if len(scores) > 0:
                    if all_args.use_wandb:
                        wandb.log({'score': np.mean(scores)},
                                  step=total_num_steps)
                    else:
                        writter.add_scalars(
                            'score', {'score': np.mean(scores)}, total_num_steps)
                    print("mean score is {}.".format(np.mean(scores)))
                else:
                    if all_args.use_wandb:
                        wandb.log({'score': 0}, step=total_num_steps)
                    else:
                        writter.add_scalars(
                            'score', {'score': 0}, total_num_steps)
                    print("Can not access mean score.")

        if episode % all_args.eval_interval == 0 and all_args.eval:
            eval_finish = False
            eval_reset_choose = np.ones(all_args.n_eval_rollout_threads) == 1.0
            eval_scores = []
            eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(
                eval_reset_choose)
            if not all_args.use_centralized_V:
                eval_share_obs = eval_obs
            eval_actions = np.ones(
                (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)*(-1.0)
            eval_recurrent_hidden_states = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_masks = np.ones(
                (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)

            while True:
                if eval_finish:
                    break
                for agent_id in range(num_agents):
                    eval_actions[:, agent_id] = np.ones(
                        (all_args.n_eval_rollout_threads, 1)).astype(np.float32)*(-1.0)
                    eval_choose = np.any(
                        eval_available_actions[:, agent_id] == 1, axis=1)

                    if ~np.any(eval_choose):
                        eval_finish = True
                        break

                    if all_args.share_policy:
                        trainer.prep_rollout()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer.policy.act(torch.FloatTensor(eval_share_obs[eval_choose, agent_id]),
                                                                                                                              torch.FloatTensor(
                                                                                                                                  eval_obs[eval_choose, agent_id]),
                                                                                                                              torch.FloatTensor(
                            eval_recurrent_hidden_states[eval_choose, agent_id]),
                            torch.FloatTensor(
                                eval_recurrent_hidden_states_critic[eval_choose, agent_id]),
                            torch.FloatTensor(
                                eval_masks[eval_choose, agent_id]),
                            torch.FloatTensor(
                                eval_available_actions[eval_choose, agent_id]),
                            deterministic=True)
                    else:
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
                            torch.FloatTensor(
                                eval_available_actions[eval_choose, agent_id]),
                            deterministic=True)

                    eval_actions[eval_choose,
                                 agent_id] = eval_action.detach().cpu().numpy()
                    eval_recurrent_hidden_states[eval_choose, agent_id] = eval_recurrent_hidden_state.detach(
                    ).cpu().numpy()
                    eval_recurrent_hidden_states_critic[eval_choose, agent_id] = eval_recurrent_hidden_state_critic.detach(
                    ).cpu().numpy()

                    # Obser reward and next obs
                    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(
                        eval_actions)
                    if not all_args.use_centralized_V:
                        eval_share_obs = eval_obs
                    eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(
                    ), num_agents, *buffer.available_actions.shape[3:])).astype(np.float32)

                    for i in range(all_args.n_eval_rollout_threads):
                        if eval_dones[i]:
                            if 'score' in eval_infos[i].keys():
                                eval_scores.append(eval_infos[i]['score'])

            print("eval mean score is {}.".format(np.mean(eval_scores)))
            if all_args.use_wandb:
                wandb.log({'eval_score': np.mean(eval_scores)},
                          step=total_num_steps)
            else:
                writter.add_scalars(
                    'eval_score', {'eval_score': np.mean(eval_scores)}, total_num_steps)

    envs.close()
    if all_args.eval:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        writter.export_scalars_to_json(str(log_dir + '/summary.json'))
        writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
