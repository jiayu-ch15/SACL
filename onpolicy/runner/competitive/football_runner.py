from collections import defaultdict, deque
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
import wandb
import pdb

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.competitive.base_runner import Runner
import copy


def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        self.all_args = config['all_args']
        self.num_red = config['all_args'].num_red
        self.num_blue = config['all_args'].num_blue
        self.save_ckpt_interval = self.all_args.save_ckpt_interval
        self.use_valuenorm = self.all_args.use_valuenorm

        self.no_info = np.ones(self.n_rollout_threads, dtype=bool)

       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            
            # train network
            red_train_infos, blue_train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # save checkpoint
            if (episode + 1) % self.save_ckpt_interval == 0:
                self.save_ckpt(total_num_steps)

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                if self.train_red:
                    red_train_infos["episode_rewards"] = np.mean(self.red_buffer.rewards) * self.episode_length
                    print("red episode rewards is {}".format(red_train_infos["episode_rewards"]))
                if self.train_blue:
                    blue_train_infos["episode_rewards"] = np.mean(self.blue_buffer.rewards) * self.episode_length
                    print("blue episode rewards is {}".format(blue_train_infos["episode_rewards"]))
                self.log_train(red_train_infos, blue_train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.no_info = np.ones(self.n_rollout_threads, dtype=bool)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # red buffer
        self.red_buffer.obs[0] = obs[:, :self.num_red].copy()
        self.red_buffer.share_obs[0] = obs[:, :self.num_red].copy()
        # blue buffer
        self.blue_buffer.obs[0] = obs[:, -self.num_blue:].copy()
        self.blue_buffer.share_obs[0] = obs[:, -self.num_blue:].copy()

    @torch.no_grad()
    def collect(self, step):
        # red trainer
        self.red_trainer.prep_rollout()
        red_values, red_actions, red_action_log_probs, red_rnn_states, red_rnn_states_critic = self.red_trainer.policy.get_actions(
            np.concatenate(self.red_buffer.share_obs[step]),
            np.concatenate(self.red_buffer.obs[step]),
            np.concatenate(self.red_buffer.rnn_states[step]),
            np.concatenate(self.red_buffer.rnn_states_critic[step]),
            np.concatenate(self.red_buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        # values:[self.envs, agents, num_critic]
        red_values = np.array(np.split(_t2n(red_values), self.n_rollout_threads))
        red_actions = np.array(np.split(_t2n(red_actions), self.n_rollout_threads))
        red_action_log_probs = np.array(np.split(_t2n(red_action_log_probs), self.n_rollout_threads))
        red_rnn_states = np.array(np.split(_t2n(red_rnn_states), self.n_rollout_threads))
        red_rnn_states_critic = np.array(np.split(_t2n(red_rnn_states_critic), self.n_rollout_threads))
        # blue trainer
        self.blue_trainer.prep_rollout()
        blue_values, blue_actions, blue_action_log_probs, blue_rnn_states, blue_rnn_states_critic = self.blue_trainer.policy.get_actions(
            np.concatenate(self.blue_buffer.share_obs[step]),
            np.concatenate(self.blue_buffer.obs[step]),
            np.concatenate(self.blue_buffer.rnn_states[step]),
            np.concatenate(self.blue_buffer.rnn_states_critic[step]),
            np.concatenate(self.blue_buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        blue_values = np.array(np.split(_t2n(blue_values), self.n_rollout_threads))
        blue_actions = np.array(np.split(_t2n(blue_actions), self.n_rollout_threads))
        blue_action_log_probs = np.array(np.split(_t2n(blue_action_log_probs), self.n_rollout_threads))
        blue_rnn_states = np.array(np.split(_t2n(blue_rnn_states), self.n_rollout_threads))
        blue_rnn_states_critic = np.array(np.split(_t2n(blue_rnn_states_critic), self.n_rollout_threads))
        # concatenate
        values = np.concatenate([red_values, blue_values], axis=1)
        actions = np.concatenate([red_actions, blue_actions], axis=1)
        action_log_probs = np.concatenate([red_action_log_probs, blue_action_log_probs], axis=1)
        rnn_states = np.concatenate([red_rnn_states, blue_rnn_states], axis=1)
        rnn_states_critic = np.concatenate([red_rnn_states_critic, blue_rnn_states_critic], axis=1)
        
        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)
        if np.any(dones_env):
            for done, info in zip(dones_env, infos):
                if done:
                    self.env_infos["goal"].append(info["score_reward"])
                    if info["score_reward"] > 0:
                        self.env_infos["win_rate"].append(1)
                    else:
                        self.env_infos["win_rate"].append(0)
                    self.env_infos["steps"].append(info["max_steps"] - info["steps_left"])

        # reset rnn and mask args for done envs
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # red buffer
        self.red_buffer.insert(
            share_obs=obs[:, :self.num_red],
            obs=obs[:, :self.num_red],
            rnn_states=rnn_states[:, :self.num_red],
            rnn_states_critic=rnn_states_critic[:, :self.num_red],
            actions=actions[:, :self.num_red],
            action_log_probs=action_log_probs[:, :self.num_red],
            value_preds=values[:, :self.num_red],
            rewards=rewards[:, :self.num_red],
            masks=masks[:, :self.num_red],
        )
        # blue buffer
        self.blue_buffer.insert(
            share_obs=obs[:, -self.num_blue:],
            obs=obs[:, -self.num_blue:],
            rnn_states=rnn_states[:, -self.num_blue:],
            rnn_states_critic=rnn_states_critic[:, -self.num_blue:],
            actions=actions[:, -self.num_blue:],
            action_log_probs=action_log_probs[:, -self.num_blue:],
            value_preds=values[:, -self.num_blue:],
            rewards=rewards[:, -self.num_blue:],
            masks=masks[:, -self.num_blue:],
        )

    def train(self):
        red_train_infos = {}
        if self.train_red:
            self.red_trainer.prep_training()
            red_train_infos = self.red_trainer.train(self.red_buffer)      
        self.red_buffer.after_update()

        blue_train_infos = {}
        if self.train_blue:
            self.blue_trainer.prep_training()
            blue_train_infos = self.blue_trainer.train(self.blue_buffer)      
        self.blue_buffer.after_update()

        # self.log_system()
        return red_train_infos, blue_train_infos

    def cross_play_restore(self, model, idx=0):
        self.red_model_dir = self.all_args.red_model_dir
        self.blue_model_dir = self.all_args.blue_model_dir
        if model == "red_model":
            red_policy_actor_state_dict = torch.load(f"{self.red_model_dir[idx]}/red_actor.pt", map_location=self.device)
            self.red_trainer.policy.actor.load_state_dict(red_policy_actor_state_dict)
        elif model == "blue_model":
            blue_policy_actor_state_dict = torch.load(f"{self.blue_model_dir[idx]}/blue_actor.pt", map_location=self.device)
            self.blue_trainer.policy.actor.load_state_dict(blue_policy_actor_state_dict)
        else:
            raise NotImplementedError(f"Model {model} is not supported.")

    @torch.no_grad()
    def eval_cross_play(self):
        num_red_models = len(self.all_args.red_model_dir)
        num_blue_models = len(self.all_args.blue_model_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        # NN initialization
        self.red_policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0],
            self.envs.action_space[0],
            device=self.device,
        )
        self.blue_policy = Policy(
            self.all_args,
            self.envs.observation_space[-1],
            self.envs.share_observation_space[-1] if self.use_centralized_V else self.envs.observation_space[-1],
            self.envs.action_space[-1],
            device=self.device,
        )

        # algorithm
        self.red_trainer = TrainAlgo(self.all_args, self.red_policy, device=self.device)
        self.blue_trainer = TrainAlgo(self.all_args, self.blue_policy, device=self.device)

        self.eval_episodes = self.all_args.eval_episodes

        cross_play_win_rate = np.zeros((num_red_models, num_blue_models), dtype=float)
        cross_play_returns = np.zeros((num_red_models, num_blue_models), dtype=float)
        for red_idx in range(num_red_models):
            self.cross_play_restore("red_model", red_idx)
            for blue_idx in range(num_blue_models):
                print(f"red model {red_idx} v.s. blue model {blue_idx}")
                self.cross_play_restore("blue_model", blue_idx)
                cross_play_win_rate[red_idx, blue_idx], cross_play_returns[red_idx, blue_idx] = self.eval_head2head()
        np.save(f"{self.log_dir}/cross_play_win_rate.npy", cross_play_win_rate)
        np.save(f"{self.log_dir}/cross_play_returns.npy", cross_play_returns)

    @torch.no_grad()
    def eval_head2head(self):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,  self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
        eval_returns = []
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            # red action
            self.red_trainer.prep_rollout()
            eval_red_actions, eval_red_rnn_states = self.red_trainer.policy.act(
                np.concatenate(eval_obs[:, :self.num_red]),
                np.concatenate(eval_rnn_states[:, :self.num_red]),
                np.concatenate(eval_masks[:, :self.num_red]),
                deterministic=self.all_args.eval_deterministic,
            )
            eval_red_actions = np.array(np.split(_t2n(eval_red_actions), self.n_eval_rollout_threads))
            eval_red_rnn_states = np.array(np.split(_t2n(eval_red_rnn_states), self.n_eval_rollout_threads))
            
            # blue action
            self.blue_trainer.prep_rollout()
            eval_blue_actions, eval_blue_rnn_states = self.blue_trainer.policy.act(
                np.concatenate(eval_obs[:, -self.num_blue:]),
                np.concatenate(eval_rnn_states[:, -self.num_blue:]),
                np.concatenate(eval_masks[:, -self.num_blue:]),
                deterministic=self.all_args.eval_deterministic,
            )
            eval_blue_actions = np.array(np.split(_t2n(eval_blue_actions), self.n_eval_rollout_threads))
            eval_blue_rnn_states = np.array(np.split(_t2n(eval_blue_rnn_states), self.n_eval_rollout_threads))
            
            # concatenate and env action
            eval_actions = np.concatenate([eval_red_actions, eval_blue_actions], axis=1)
            eval_rnn_states = np.concatenate([eval_red_rnn_states, eval_blue_rnn_states], axis=1)
            # eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_returns.append(eval_rewards[:,0,0])

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goals[num_done] = eval_infos[idx_env]["score_reward"]
                        eval_win_rates[num_done] = 1 if eval_infos[idx_env]["score_reward"] > 0 else 0
                        eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - eval_infos[idx_env]["steps_left"]
                        # print("episode {:>2d} done by env {:>2d}: {}".format(num_done, idx_env, eval_infos[idx_env]["score_reward"]))
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        # get expected goal
        eval_returns = np.mean(np.sum(np.stack(eval_returns,axis=1),axis=1))
        eval_expected_goal = np.mean(eval_goals)
        eval_expected_win_rate = np.mean(eval_win_rates)
        eval_expected_step = np.mean(eval_steps)
    
        # log and print
        print("eval expected return is {}.".format(eval_returns), "eval expected win rate is {}.".format(eval_expected_win_rate), "eval expected step is {}.\n".format(eval_expected_step))
        return eval_expected_win_rate, eval_returns

    # TODO
    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            # get actions
            self.trainer.prep_rollout()

            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=self.all_args.eval_deterministic
            )
            
            # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goals[num_done] = eval_infos[idx_env]["score_reward"]
                        eval_win_rates[num_done] = 1 if eval_infos[idx_env]["score_reward"] > 0 else 0
                        eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - eval_infos[idx_env]["steps_left"]
                        # print("episode {:>2d} done by env {:>2d}: {}".format(num_done, idx_env, eval_infos[idx_env]["score_reward"]))
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        # get expected goal
        eval_expected_goal = np.mean(eval_goals)
        eval_expected_win_rate = np.mean(eval_win_rates)
        eval_expected_step = np.mean(eval_steps)
    
        # log and print
        print("eval expected goal is {}.".format(eval_expected_goal))
        wandb.log({"expected_goal": eval_expected_goal}, step=total_num_steps)
        wandb.log({"eval_expected_win_rate": eval_expected_win_rate}, step=total_num_steps)
        wandb.log({"expected_step": eval_expected_step}, step=total_num_steps)

    @torch.no_grad()
    def render(self):        
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            # render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            # render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            red_render_rnn_states = np.zeros((self.n_rollout_threads, self.num_red, self.recurrent_N, self.hidden_size), dtype=np.float32)
            red_render_masks = np.ones((self.n_rollout_threads, self.num_red, 1), dtype=np.float32)

            blue_render_rnn_states = np.zeros((self.n_rollout_threads, self.num_blue, self.recurrent_N, self.hidden_size), dtype=np.float32)
            blue_render_masks = np.ones((self.n_rollout_threads, self.num_blue, 1), dtype=np.float32)

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            render_dones = False
            env_step = 0
            while not np.any(render_dones):
                self.red_trainer.prep_rollout()
                red_render_actions, red_render_rnn_states = self.red_trainer.policy.act(
                    np.concatenate(render_obs[:,:self.num_red]),
                    np.concatenate(red_render_rnn_states),
                    np.concatenate(red_render_masks),
                    deterministic=False
                )
                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                red_render_actions = np.array(np.split(_t2n(red_render_actions), self.n_rollout_threads))
                red_render_rnn_states = np.array(np.split(_t2n(red_render_rnn_states), self.n_rollout_threads))

                self.blue_trainer.prep_rollout()
                blue_render_actions, blue_render_rnn_states = self.blue_trainer.policy.act(
                    np.concatenate(render_obs[:,-self.num_blue:]),
                    np.concatenate(blue_render_rnn_states),
                    np.concatenate(blue_render_masks),
                    deterministic=False
                )
                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                blue_render_actions = np.array(np.split(_t2n(blue_render_actions), self.n_rollout_threads))
                blue_render_rnn_states = np.array(np.split(_t2n(blue_render_rnn_states), self.n_rollout_threads))

                render_actions = np.concatenate([red_render_actions, blue_render_actions], axis=1)

                render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)
                # print('dones', render_dones, 'ball_obs', render_obs[0,0,88:91], 'left_obs', render_obs[0,0,:22], 'right_obs', render_obs[0,0,44:66])
                env_step += 1

                # append frame
                if self.all_args.save_gifs:        
                    image = render_infos[0]["frame"]
                    frames.append(image)
            
            # print goal
            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(i_episode, render_rewards[0, 0]))

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
        
        print("expected goal: {}".format(np.mean(render_goals)))