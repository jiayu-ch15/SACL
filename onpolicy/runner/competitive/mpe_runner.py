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
                print(f"saved!")
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
                red_train_infos["step_reward"] = np.mean(self.red_buffer.rewards)
                blue_train_infos["step_reward"] = np.mean(self.blue_buffer.rewards)
                self.log_train(red_train_infos, blue_train_infos, total_num_steps)
                print(
                    f"adv step reward: {red_train_infos['step_reward']:.2f}, "
                    f"good step reward: {blue_train_infos['step_reward']:.2f}."
                )
                # env info
                num_steps = np.array([info[-1]["num_steps"] for info in infos])
                num_outside = np.array([info[-1]["num_outside"] for info in infos])
                num_collision = np.array([info[-1]["num_collision"] for info in infos])
                env_infos = dict(num_steps=num_steps, num_outside=num_outside, num_collision=num_collision)
                self.log_env(env_infos, total_num_steps)
                print(
                    f"num steps: {np.mean(num_steps):.2f}, "
                    f"num outside: {np.mean(num_outside):.2f}, "
                    f"num collision: {np.mean(num_collision):.2f}.\n"
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
            num_steps = infos[0][-1]["num_steps"]
            num_outside = infos[0][-1]["num_outside"]
            num_collision = infos[0][-1]["num_collision"]
            print(
                f"episode {episode}: adv step rewards={adv_step_reward:.2f}, good step reward={good_step_reward:.2f}, "
                f"num steps={num_steps}, num outside={num_outside}, num collision={num_collision}."
            )

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(f"{self.gif_dir}/episode{episode}.gif", all_frames[:-1], duration=self.all_args.ifi)
