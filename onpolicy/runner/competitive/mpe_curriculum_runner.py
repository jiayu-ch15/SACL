import imageio
import numpy as np
import time
import torch

from onpolicy.runner.competitive.base_runner import Runner
from onpolicy.utils.curriculum_buffer import CurriculumBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class MPECurriculumRunner(Runner):
    def __init__(self, config):
        super(MPECurriculumRunner, self).__init__(config)

        self.prob_curriculum = self.all_args.prob_curriculum
        self.sample_metric = self.all_args.sample_metric
        self.curriculum_buffer = CurriculumBuffer(
            buffer_size=self.all_args.curriculum_buffer_size,
            update_method=self.all_args.update_method,
        )

        self.env_infos = dict(
            initial_dist=np.zeros(self.n_rollout_threads, dtype=float), 
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
                # manually reset done envs
                obs = self.reset_subgames(obs, dones)
                # insert data into buffer
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
                # update env info
                if (episode + 1) % self.log_interval == 0:
                    self.update_env_infos(dones, infos)
            # compute return
            self.compute()            
            # train network
            red_train_infos, blue_train_infos = self.train()

            # update curiculum buffer
            self.update_curriculum()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if ((episode + 1) % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # save checkpoint
            if (episode + 1) % self.save_ckpt_interval == 0:
                self.save_ckpt(total_num_steps)

            # log information
            if (episode + 1) % self.log_interval == 0:
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
                red_train_infos["step_reward"] = np.mean(self.red_buffer.rewards)
                red_train_infos["value_normalizer_mean"] = np.mean(_t2n(red_value_mean))
                red_train_infos["value_normalizer_var"] = np.mean(_t2n(red_value_var))
                blue_train_infos["step_reward"] = np.mean(self.blue_buffer.rewards)
                blue_train_infos["value_normalizer_mean"] = np.mean(_t2n(blue_value_mean))
                blue_train_infos["value_normalizer_var"] = np.mean(_t2n(blue_value_var))
                self.log_train(red_train_infos, blue_train_infos, total_num_steps)
                print(
                    f"adv step reward: {red_train_infos['step_reward']:.2f}, "
                    f"good step reward: {blue_train_infos['step_reward']:.2f}."
                )
                # env info
                self.log_env(self.env_infos, total_num_steps)
                print(
                    f"initial distance: {np.mean(self.env_infos['initial_dist']):.2f}, "
                    f"start step: {np.mean(self.env_infos['start_step']):.2f}, "
                    f"end step: {np.mean(self.env_infos['end_step']):.2f}, "
                    f"episode length: {np.mean(self.env_infos['episode_length']):.2f}.\n"
                    f"outside: {np.mean(self.env_infos['outside']):.2f}, "
                    f"collision: {np.mean(self.env_infos['collision']):.2f}, "
                    f"escape: {np.mean(self.env_infos['escape']):.2f}, "
                    f"outside per step: {np.mean(self.env_infos['outside_per_step']):.4f}, "
                    f"collision per step: {np.mean(self.env_infos['collision_per_step']):.4f}.\n"
                )
                self.env_infos = dict(
                    initial_dist=np.zeros(self.n_rollout_threads, dtype=float), 
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
            if self.use_eval and (episode + 1) % self.eval_interval == 0:
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

    def reset_subgames(self, obs, dones):
        # reset subgame: env is done and p~U[0, 1] < prob_curriculum
        env_dones = np.all(dones, axis=1)
        use_curriculum = (np.random.uniform(size=self.n_rollout_threads) < self.prob_curriculum)
        env_idx = np.arange(self.n_rollout_threads)[env_dones * use_curriculum]
        # sample initial states and reset
        if len(env_idx) != 0:
            initial_states = self.curriculum_buffer.sample(len(env_idx))
            obs[env_idx] = self.envs.partial_reset(env_idx.tolist(), initial_states)
        return obs
    
    def update_curriculum(self):
        # update and get share obs
        share_obs = self.curriculum_buffer.update_states()

        # get weights according to metric
        num_weights = share_obs.shape[0]
        if self.sample_metric == "uniform":
            weights = np.ones(num_weights, dtype=np.float32)
        elif self.sample_metric == "variance":
            red_rnn_states_critic = np.zeros((num_weights * self.num_red, self.recurrent_N, self.hidden_size), dtype=np.float32)
            red_masks = np.ones((num_weights * self.num_red, 1), dtype=np.float32)
            red_values = self.red_trainer.policy.get_values(
                np.concatenate(share_obs[:, :self.num_red]),
                red_rnn_states_critic,
                red_masks,
            )
            red_values = np.array(np.split(_t2n(red_values), num_weights))
            red_values_denorm = self.red_trainer.value_normalizer.denormalize(red_values)
            weights = np.var(red_values_denorm, axis=1)[:, 0]
        elif self.sample_metric == "rb_variance":
            red_rnn_states_critic = np.zeros((num_weights * self.num_red, self.recurrent_N, self.hidden_size), dtype=np.float32)
            red_masks = np.ones((num_weights * self.num_red, 1), dtype=np.float32)
            red_values = self.red_trainer.policy.get_values(
                np.concatenate(share_obs[:, :self.num_red]),
                red_rnn_states_critic,
                red_masks,
            )
            red_values = np.array(np.split(_t2n(red_values), num_weights))
            red_values_denorm = self.red_trainer.value_normalizer.denormalize(red_values)

            blue_rnn_states_critic = np.zeros((num_weights * self.num_blue, self.recurrent_N, self.hidden_size), dtype=np.float32)
            blue_masks = np.ones((num_weights * self.num_blue, 1), dtype=np.float32)
            blue_values = self.blue_trainer.policy.get_values(
                np.concatenate(share_obs[:, -self.num_blue:]),
                blue_rnn_states_critic,
                blue_masks,
            )
            blue_values = np.array(np.split(_t2n(blue_values), num_weights))
            blue_values_denorm = self.blue_trainer.value_normalizer.denormalize(blue_values)

            values_denorm = np.concatenate([red_values_denorm, -blue_values_denorm], axis=1)
            weights = np.var(values_denorm, axis=1)[:, 0]
        # NOTE: only for test
        elif self.sample_metric == "rb_variance_mean":
            red_rnn_states_critic = np.zeros((num_weights * self.num_red, self.recurrent_N, self.hidden_size), dtype=np.float32)
            red_masks = np.ones((num_weights * self.num_red, 1), dtype=np.float32)
            red_values = self.red_trainer.policy.get_values(
                np.concatenate(share_obs[:, :self.num_red]),
                red_rnn_states_critic,
                red_masks,
            )
            red_values = np.array(np.split(_t2n(red_values), num_weights))
            red_values_denorm = self.red_trainer.value_normalizer.denormalize(red_values)
            red_values_denorm = np.mean(red_values_denorm, axis=1, keepdims=True)

            blue_rnn_states_critic = np.zeros((num_weights * self.num_blue, self.recurrent_N, self.hidden_size), dtype=np.float32)
            blue_masks = np.ones((num_weights * self.num_blue, 1), dtype=np.float32)
            blue_values = self.blue_trainer.policy.get_values(
                np.concatenate(share_obs[:, -self.num_blue:]),
                blue_rnn_states_critic,
                blue_masks,
            )
            blue_values = np.array(np.split(_t2n(blue_values), num_weights))
            blue_values_denorm = self.blue_trainer.value_normalizer.denormalize(blue_values)

            values_denorm = np.concatenate([red_values_denorm, -blue_values_denorm], axis=1)
            weights = np.var(values_denorm, axis=1)[:, 0]

        self.curriculum_buffer.update_weights(weights)

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

        # curriculum buffer
        # TODO: better way to filter bad states

        # # only red value
        # denormalized_values = self.red_trainer.value_normalizer.denormalize(values[:, :self.num_red])
        
        # # both red and blue value
        # denormalized_red_values = self.red_trainer.value_normalizer.denormalize(values[:, :self.num_red])
        # denormalized_blue_values = self.blue_trainer.value_normalizer.denormalize(values[:, -self.num_blue:])
        # denormalized_values = np.concatenate([denormalized_red_values, -denormalized_blue_values], axis=1)
        
        new_states = []
        new_share_obs = []
        for info, share_obs in zip(infos, obs):
            state = info[0]["state"]
            if np.any(np.abs(state[0:2]) > self.all_args.corner_max):
                continue
            if np.any(np.abs(state[4:6]) > self.all_args.corner_max):
                continue
            if np.any(np.abs(state[8:10]) > self.all_args.corner_max):
                continue
            if np.any(np.abs(state[12:14]) > self.all_args.corner_max):
                continue
            if np.any(state[-1] >= self.all_args.horizon):
                continue
            new_states.append(state)
            new_share_obs.append(share_obs)
        self.curriculum_buffer.insert(new_states, new_share_obs)

    def update_env_infos(self, dones, infos):
        # info dict
        env_dones = np.all(dones, axis=1)
        for idx in np.arange(self.n_rollout_threads)[env_dones]:
            self.env_infos["initial_dist"][idx] = infos[idx][-1]["initial_dist"]
            self.env_infos["start_step"][idx] = infos[idx][-1]["start_step"]
            self.env_infos["end_step"][idx] = infos[idx][-1]["num_steps"]
            self.env_infos["episode_length"][idx] = infos[idx][-1]["episode_length"]
            self.env_infos["outside"][idx] = (infos[idx][-1]["outside_per_step"] > 0)
            self.env_infos["collision"][idx] = (infos[idx][-1]["collision_per_step"] > 0)
            self.env_infos["escape"][idx] = (not self.env_infos["outside"][idx]) and (not self.env_infos["collision"][idx])
            self.env_infos["outside_per_step"][idx] = infos[idx][-1]["outside_per_step"]
            self.env_infos["collision_per_step"][idx] = infos[idx][-1]["collision_per_step"]

    # @torch.no_grad()
    # def eval(self, total_num_steps):
    #     eval_episode_rewards = []
    #     eval_obs = self.eval_envs.reset()

    #     eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
    #     eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

    #     for eval_step in range(self.episode_length):
    #         self.trainer.prep_rollout()
    #         eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
    #                                             np.concatenate(eval_rnn_states),
    #                                             np.concatenate(eval_masks),
    #                                             deterministic=True)
    #         eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
    #         eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
    #         if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
    #             for i in range(self.eval_envs.action_space[0].shape):
    #                 eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
    #                 if i == 0:
    #                     eval_actions_env = eval_uc_actions_env
    #                 else:
    #                     eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
    #         elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
    #             eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
    #         else:
    #             raise NotImplementedError

    #         # Obser reward and next obs
    #         eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
    #         eval_episode_rewards.append(eval_rewards)

    #         eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
    #         eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #         eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

    #     eval_episode_rewards = np.array(eval_episode_rewards)
    #     eval_env_infos = {}
    #     eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
    #     print("eval average episode rewards of agent: " + str(np.mean(eval_env_infos["eval_average_episode_rewards"])))
    #     self.log_env(eval_env_infos, total_num_steps)

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

            initial_dist = infos[0][-1]["initial_dist"]
            start_step = infos[0][-1]["start_step"]
            end_step = infos[0][-1]["num_steps"]
            outside_per_step = infos[0][-1]["outside_per_step"]
            collision_per_step = infos[0][-1]["collision_per_step"]
            print(
                f"episode {episode}: adv step rewards={adv_step_reward:.2f}, good step reward={good_step_reward:.2f}, "
                f"initial_dist={initial_dist}, start step={start_step}, end step={end_step}, "
                f"outside per step={outside_per_step}, collision per step={collision_per_step}."
            )

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(f"{self.gif_dir}/episode{episode}.gif", all_frames[:-1], duration=self.all_args.ifi)
