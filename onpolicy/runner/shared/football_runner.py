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
from onpolicy.runner.shared.base_runner import Runner


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
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
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
                
                train_infos["red_episode_rewards"] = np.mean(self.buffer.rewards[:,:,:self.num_red]) * self.episode_length
                train_infos["blue_episode_rewards"] = np.mean(self.buffer.rewards[:,:,-self.num_blue::]) * self.episode_length
                print("red episode rewards is {}".format(train_infos["red_episode_rewards"]))
                print("blue episode rewards is {}".format(train_infos["blue_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # insert obs to buffer
        self.buffer.share_obs[0] = obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

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

        self.buffer.insert(
            share_obs=obs,
            obs=obs,
            rnn_states=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks
        )

    def save_ckpt(self, total_num_steps):
        million_steps = int(total_num_steps // 1000000)
        save_ckpt_dir = f"{self.save_dir}/{million_steps}M"
        assert not os.path.exists(save_ckpt_dir), (f"checkpoint path {save_ckpt_dir} already exists.")
        os.makedirs(save_ckpt_dir)
        
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), f"{save_ckpt_dir}/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), f"{save_ckpt_dir}/critic.pt")
        if self.use_valuenorm:
            value_normalizer = self.trainer.value_normalizer
            torch.save(value_normalizer.state_dict(), f"{save_ckpt_dir}/value_normalizer.pt")

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    def cross_play_restore(self, model, idx=0):
        self.red_model_dir = self.all_args.red_model_dir
        self.blue_model_dir = self.all_args.blue_model_dir
        if model == "red_model":
            red_policy_actor_state_dict = torch.load(f"{self.red_model_dir[idx]}/actor.pt", map_location=self.device)
            self.red_trainer.policy.actor.load_state_dict(red_policy_actor_state_dict)
        elif model == "blue_model":
            blue_policy_actor_state_dict = torch.load(f"{self.blue_model_dir[idx]}/actor.pt", map_location=self.device)
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

        cross_play_returns = np.zeros((num_red_models, num_blue_models), dtype=float)
        for red_idx in range(num_red_models):
            self.cross_play_restore("red_model", red_idx)
            for blue_idx in range(num_blue_models):
                print(f"red model {red_idx} v.s. blue model {blue_idx}")
                self.cross_play_restore("blue_model", blue_idx)
                cross_play_returns[red_idx, blue_idx] = self.eval_head2head()
        np.save(f"{self.log_dir}/cross_play_returns.npy", cross_play_returns)

    # todo
    @torch.no_grad()
    def eval_head2head(self):
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
        print()
        print("eval expected win rate is {}.".format(eval_expected_win_rate), "eval expected step is {}.\n".format(eval_expected_step))
        return eval_expected_goal

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
            render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
                self.trainer.prep_rollout()
                render_actions, render_rnn_states = self.trainer.policy.act(
                    np.concatenate(render_obs),
                    np.concatenate(render_rnn_states),
                    np.concatenate(render_masks),
                    deterministic=True
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)

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
