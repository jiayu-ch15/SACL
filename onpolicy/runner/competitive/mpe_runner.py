import imageio
import numpy as np
import time
import torch

from onpolicy.runner.competitive.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

        self.no_info = np.ones(self.n_rollout_threads, dtype=bool)
        self.env_infos = dict(
            initial_dist=np.zeros(self.n_rollout_threads, dtype=int), 
            start_step=np.zeros(self.n_rollout_threads, dtype=int), 
            end_step=np.zeros(self.n_rollout_threads, dtype=int),
            episode_length=np.zeros(self.n_rollout_threads, dtype=int),
            outside=np.zeros(self.n_rollout_threads, dtype=bool), 
            collision=np.zeros(self.n_rollout_threads, dtype=bool),
            escape=np.zeros(self.n_rollout_threads, dtype=bool),
            outside_per_step=np.zeros(self.n_rollout_threads, dtype=float), 
            collision_per_step=np.zeros(self.n_rollout_threads, dtype=float),
        )

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # sample actions from policy
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # env step
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # insert data into buffer
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
            # compute return and update network
            self.compute()
            red_train_infos, blue_train_infos = self.train()
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                # basic info
                end = time.time()
                print(
                    f"Env: MPE, Scenario: {self.all_args.scenario_name}, Exp: {self.experiment_name}, "
                    f"Updates: {episode}/{episodes}, Env steps: {total_num_steps}/{self.num_env_steps}, "
                    f"FSP: {int(total_num_steps / (end - start))}."
                )
                # training info
                red_value_mean, red_value_var = self.red_trainer.value_normalizer.running_mean_var()
                blue_value_mean, blue_value_var = self.blue_trainer.value_normalizer.running_mean_var()
                if self.train_red:
                    red_train_infos["step_reward"] = np.mean(self.red_buffer.rewards)
                    red_train_infos["value_normalizer_mean"] = np.mean(_t2n(red_value_mean))
                    red_train_infos["value_normalizer_var"] = np.mean(_t2n(red_value_var))
                    print(f"adv step reward: {red_train_infos['step_reward']:.2f}.")
                if self.train_blue:
                    blue_train_infos["step_reward"] = np.mean(self.blue_buffer.rewards)
                    blue_train_infos["value_normalizer_mean"] = np.mean(_t2n(blue_value_mean))
                    blue_train_infos["value_normalizer_var"] = np.mean(_t2n(blue_value_var))
                    print(f"good step reward: {blue_train_infos['step_reward']:.2f}.")
                self.log_train(red_train_infos, blue_train_infos, total_num_steps)

                # env info
                self.log_env(self.env_infos, total_num_steps)
                print(
                    f"initial distance: {np.mean(self.env_infos['initial_dist']):.2f}, "
                    f"start step: {np.mean(self.env_infos['start_step']):.2f}, "
                    f"end step: {np.mean(self.env_infos['end_step']):.2f}, "
                    f"episode length: {np.mean(self.env_infos['episode_length']):.2f}.\n"
                    f"outside: {np.mean(self.env_infos['outside']):.2f}, "
                    f"collision: {np.mean(self.env_infos['collision']):.2f}, "
                    f"escape: {np.mean(self.env_infos['escape']):.2f}.\n"
                    f"outside per step: {np.mean(self.env_infos['outside_per_step']):.2f}, "
                    f"collision per step: {np.mean(self.env_infos['collision_per_step']):.2f}.\n"
                )
                self.no_info = np.ones(self.n_rollout_threads, dtype=bool)
                self.env_infos = dict(
                    initial_dist=np.zeros(self.n_rollout_threads, dtype=int), 
                    start_step=np.zeros(self.n_rollout_threads, dtype=int), 
                    end_step=np.zeros(self.n_rollout_threads, dtype=int),
                    episode_length=np.zeros(self.n_rollout_threads, dtype=int),
                    outside=np.zeros(self.n_rollout_threads, dtype=bool), 
                    collision=np.zeros(self.n_rollout_threads, dtype=bool),
                    escape=np.zeros(self.n_rollout_threads, dtype=bool),
                    outside_per_step=np.zeros(self.n_rollout_threads, dtype=float), 
                    collision_per_step=np.zeros(self.n_rollout_threads, dtype=float),
                )

            # eval
            if self.use_eval and episode % self.eval_interval == 0:
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
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
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

        # info dict
        env_dones = np.all(dones, axis=1)
        for idx in np.arange(self.n_rollout_threads)[env_dones * self.no_info]:
            self.env_infos["initial_dist"][idx] = infos[idx][-1]["initial_dist"]
            self.env_infos["start_step"][idx] = infos[idx][-1]["start_step"]
            self.env_infos["end_step"][idx] = infos[idx][-1]["num_steps"]
            self.env_infos["episode_length"][idx] = infos[idx][-1]["episode_length"]
            self.env_infos["outside"][idx] = (infos[idx][-1]["outside_per_step"] > 0)
            self.env_infos["collision"][idx] = (infos[idx][-1]["collision_per_step"] > 0)
            self.env_infos["escape"][idx] = (not self.env_infos["outside"][idx]) and (not self.env_infos["collision"][idx])
            self.env_infos["outside_per_step"][idx] = infos[idx][-1]["outside_per_step"]
            self.env_infos["collision_per_step"][idx] = infos[idx][-1]["collision_per_step"]
            self.no_info[idx] = False

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_adv_step_reward = []
        eval_good_step_reward = []
        eval_start_step = []
        eval_end_step = []
        eval_episode_length = []
        eval_outside_per_step = []
        eval_collision_per_step = []

        episodes = self.eval_episodes // self.n_eval_rollout_threads
        for episode in range(episodes):
            eval_adv_reward = []
            eval_good_reward = []

            eval_obs = self.eval_envs.reset()
            eval_share_obs = eval_obs.copy()
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_rnn_states_critic = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                # red action
                self.red_trainer.prep_rollout()
                eval_red_values, eval_red_actions, _, eval_red_rnn_states, eval_red_rnn_states_critic = self.red_trainer.policy.get_actions(
                    np.concatenate(eval_share_obs[:, :self.num_red]),
                    np.concatenate(eval_obs[:, :self.num_red]),
                    np.concatenate(eval_rnn_states[:, :self.num_red]),
                    np.concatenate(eval_rnn_states_critic[:, :self.num_red]),
                    np.concatenate(eval_masks[:, :self.num_red]),
                    deterministic=False,
                )
                eval_red_values = np.array(np.split(_t2n(eval_red_values), self.n_eval_rollout_threads))
                eval_red_actions = np.array(np.split(_t2n(eval_red_actions), self.n_eval_rollout_threads))
                eval_red_rnn_states = np.array(np.split(_t2n(eval_red_rnn_states), self.n_eval_rollout_threads))
                eval_red_rnn_states_critic = np.array(np.split(_t2n(eval_red_rnn_states_critic), self.n_eval_rollout_threads))
                
                # blue action
                self.blue_trainer.prep_rollout()
                eval_blue_values, eval_blue_actions, _, eval_blue_rnn_states, eval_blue_rnn_states_critic = self.blue_trainer.policy.get_actions(
                    np.concatenate(eval_share_obs[:, -self.num_blue:]),
                    np.concatenate(eval_obs[:, -self.num_blue:]),
                    np.concatenate(eval_rnn_states[:, -self.num_blue:]),
                    np.concatenate(eval_rnn_states_critic[:, -self.num_blue:]),
                    np.concatenate(eval_masks[:, -self.num_blue:]),
                    deterministic=False,
                )
                eval_blue_values = np.array(np.split(_t2n(eval_blue_values), self.n_eval_rollout_threads))
                eval_blue_actions = np.array(np.split(_t2n(eval_blue_actions), self.n_eval_rollout_threads))
                eval_blue_rnn_states = np.array(np.split(_t2n(eval_blue_rnn_states), self.n_eval_rollout_threads))
                eval_blue_rnn_states_critic = np.array(np.split(_t2n(eval_blue_rnn_states_critic), self.n_eval_rollout_threads))
                
                # concatenate and env action
                eval_values = np.concatenate([eval_red_values, eval_blue_values], axis=1)
                eval_actions = np.concatenate([eval_red_actions, eval_blue_actions], axis=1)
                eval_rnn_states = np.concatenate([eval_red_rnn_states, eval_blue_rnn_states], axis=1)
                eval_rnn_states_critic = np.concatenate([eval_red_rnn_states_critic, eval_blue_rnn_states_critic], axis=1)
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)

                # env step
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

                eval_share_obs = eval_obs.copy()
                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_rnn_states_critic[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

                # reward
                eval_adv_reward.append(eval_rewards[:, 0])
                eval_good_reward.append(eval_rewards[:, -1])

            # info
            eval_adv_step_reward.extend(np.mean(eval_adv_reward, axis=0).tolist())
            eval_good_step_reward.extend(np.mean(eval_good_reward, axis=0).tolist())
            eval_start_step.extend([info[-1]["start_step"] for info in eval_infos])
            eval_end_step.extend([info[-1]["num_steps"] for info in eval_infos])
            eval_episode_length.extend([info[-1]["episode_length"] for info in eval_infos])
            eval_outside_per_step.extend([info[-1]["outside_per_step"] for info in eval_infos])
            eval_collision_per_step.extend([info[-1]["collision_per_step"] for info in eval_infos])

        eval_env_infos = dict(
            eval_red_step_reward=np.array(eval_adv_step_reward),
            eval_blue_step_reward=np.array(eval_good_step_reward),
            eval_start_step=np.array(eval_start_step),
            eval_end_step=np.array(eval_end_step),
            eval_episode_length=np.array(eval_episode_length),
            eval_outside_per_step=np.array(eval_outside_per_step),
            eval_collision_per_step=np.array(eval_collision_per_step),
        )
        self.log_env(eval_env_infos, total_num_steps)
        print(
            f"adv step reward: {np.mean(eval_adv_step_reward):.2f}, "
            f"good step reward: {np.mean(eval_good_step_reward):.2f}, "
            f"outside per step: {np.mean(eval_outside_per_step):.4f}, "
            f"collision per step: {np.mean(eval_collision_per_step):.4f}."
        )
        # print(
        #     f"start step: {np.mean(eval_start_step):.2f}, "
        #     f"end step: {np.mean(eval_end_step):.2f}, "
        #     f"episode length: {np.mean(eval_episode_length):.2f}."
        # )

    @torch.no_grad()
    def render(self):
        envs = self.envs
        
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                all_frames = []
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            step_rewards = []
            
            for step in range(self.episode_length):
                # red action
                self.red_trainer.prep_rollout()
                red_actions, red_rnn_states = self.red_trainer.policy.act(
                    np.concatenate(obs[:, :self.num_red]),
                    np.concatenate(rnn_states[:, :self.num_red]),
                    np.concatenate(masks[:, :self.num_red]),
                    deterministic=False,
                )
                red_actions = np.array(np.split(_t2n(red_actions), self.n_rollout_threads))
                red_rnn_states = np.array(np.split(_t2n(red_rnn_states), self.n_rollout_threads))
                # blue action
                self.blue_trainer.prep_rollout()
                blue_actions, blue_rnn_states = self.blue_trainer.policy.act(
                    np.concatenate(obs[:, -self.num_blue:]),
                    np.concatenate(rnn_states[:, -self.num_blue:]),
                    np.concatenate(masks[:, -self.num_blue:]),
                    deterministic=False,
                )
                blue_actions = np.array(np.split(_t2n(blue_actions), self.n_rollout_threads))
                blue_rnn_states = np.array(np.split(_t2n(blue_rnn_states), self.n_rollout_threads))
                # concatenate and env action
                actions = np.concatenate([red_actions, blue_actions], axis=1)
                rnn_states = np.concatenate([red_rnn_states, blue_rnn_states], axis=1)
                actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)

                # env step
                obs, rewards, dones, infos = envs.step(actions_env)
                step_rewards.append(rewards)

                # update
                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                # append image
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            # print result
            step_rewards = np.array(step_rewards)
            adv_step_reward = np.mean(step_rewards[:, :, :self.num_red])
            good_step_reward = np.mean(step_rewards[:, :, -self.num_blue:])

            start_step = infos[0][-1]["start_step"]
            end_step = infos[0][-1]["num_steps"]
            outside_per_step = infos[0][-1]["outside_per_step"]
            collision_per_step = infos[0][-1]["collision_per_step"]
            print(
                f"episode {episode}: adv step rewards={adv_step_reward:.2f}, good step reward={good_step_reward:.2f}, "
                f"start step={start_step}, end step={end_step}, outside per step={outside_per_step}, collision per step={collision_per_step}."
            )

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(f"{self.gif_dir}/episode{episode}.gif", all_frames[:-1], duration=self.all_args.ifi)
