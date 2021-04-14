    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
from collections import defaultdict

def _t2n(x):
    return x.detach().cpu().numpy()

class HumanRunner(Runner):
    def __init__(self, config):
        super(HumanRunner, self).__init__(config)

        # load good agent model
        if self.all_args.prey_model_dir is not None and not self.all_args.use_fixed_prey:
            self.load_prey_model()

    def load_prey_model(self):
        # policy network
        from onpolicy.envs.human.prey_policy.Policy import PreyPolicy
        self.prey_policy = PreyPolicy(self.all_args,
                            self.envs.observation_space[-1],
                            self.envs.action_space[-1],
                            device = self.device)
        prey_state_dict = torch.load(str(self.all_args.prey_model_dir) + '/prey.pt')#, map_location='cpu')
        self.prey_policy.actor.load_state_dict(prey_state_dict)
        self.prey_policy.actor.eval()

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if not self.all_args.use_fixed_prey:
                self.prey_rnn_states = np.zeros((self.n_rollout_threads, self.all_args.num_good_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                self.prey_masks = np.ones((self.n_rollout_threads, self.all_args.num_good_agents, 1), dtype=np.float32)
        
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                dict_obs, dict_rewards, dones, infos = self.envs.step(actions_env)
                
                obs, rewards, self.prey_obs = self._convert(dict_obs, dict_rewards)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "Human":
                    env_infos = defaultdict(list)
                    for info in infos:
                        for agent_id in range(self.all_args.num_good_agents + self.all_args.num_adversaries):
                            for key in info[agent_id].keys():
                                agent_key = 'agent{}/{}'.format(agent_id, key)
                                env_infos[agent_key].append(info[agent_id][key])

                train_infos["average_episode_rewards"] = np.mean(np.sum(self.buffer.rewards, axis=0))
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def _convert(self, dict_obs, dict_rewards):
        prey_obs = []
        predator_obs = []
        rewards = []
        for o, r in zip(dict_obs, dict_rewards):
            prey_obs.append(np.array(o['prey']))
            predator_obs.append(np.array(o['predator']))
            rewards.append(np.array(r['predator']))
        return np.array(predator_obs), np.array(rewards), np.array(prey_obs)

    def _convert_obs(self, dict_obs):
        prey_obs = []
        predator_obs = []

        for o in dict_obs:
            prey_obs.append(np.array(o['prey']))
            predator_obs.append(np.array(o['predator']))

        return np.array(predator_obs), np.array(prey_obs)

    def warmup(self):
        # reset env
        dict_obs = self.envs.reset()
        obs, self.prey_obs = self._convert_obs(dict_obs)

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        if not self.all_args.use_fixed_prey:
            prey_action, prey_rnn_states = self.prey_policy.act(np.concatenate(self.prey_obs),
                                                        np.concatenate(self.prey_rnn_states),
                                                        np.concatenate(self.prey_masks),
                                                        deterministic=True)
            self.prey_rnn_states = np.array(np.split(_t2n(prey_rnn_states), self.n_rollout_threads))
        else:
            prey_action = torch.zeros([self.n_rollout_threads, self.all_args.num_good_agents], dtype=int).to(self.device)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        prey_actions = np.array(np.split(_t2n(prey_action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        tmp_actions = np.concatenate([actions, prey_actions], axis=1)
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[tmp_actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[tmp_actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []

        eval_dict_obs = self.eval_envs.reset()
        eval_obs, eval_prey_obs = self._convert_obs(eval_dict_obs)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        if not self.all_args.use_fixed_prey:
            eval_prey_rnn_states = np.zeros((self.n_eval_rollout_threads, self.all_args.num_good_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_prey_masks = np.ones((self.n_eval_rollout_threads, self.all_args.num_good_agents, 1), dtype=np.float32)
            
        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            if not self.all_args.use_fixed_prey:
                eval_prey_action, eval_prey_rnn_states = self.prey_policy.act(np.concatenate(eval_prey_obs),
                                                np.concatenate(eval_prey_rnn_states),
                                                np.concatenate(eval_prey_masks),
                                                deterministic=True)
                eval_prey_rnn_states = np.array(np.split(_t2n(eval_prey_rnn_states), self.n_eval_rollout_threads))
            else:
                eval_prey_action = torch.zeros([self.n_eval_rollout_threads, self.all_args.num_good_agents], dtype=int).to(self.device)
            
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_prey_actions = np.array(np.split(_t2n(eval_prey_action), self.n_eval_rollout_threads))
            tmp_eval_actions = np.concatenate([eval_actions, eval_prey_actions], axis=1)
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[tmp_eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[tmp_eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_dict_obs, eval_dict_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            
            eval_obs, eval_rewards, eval_prey_obs = self._convert(eval_dict_obs, eval_dict_rewards)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = defaultdict(list)
        for eval_info in eval_infos:
            for agent_id in range(self.all_args.num_good_agents + self.all_args.num_adversaries):
                for key in eval_info[agent_id].keys():
                    agent_key = 'agent{}/eval_{}'.format(agent_id, key)
                    eval_env_infos[agent_key].append(eval_info[agent_id][key])
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(np.mean(eval_env_infos['eval_average_episode_rewards'])))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            dict_obs = envs.reset()
            obs, prey_obs = self._convert_obs(dict_obs)

            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            if not self.all_args.use_fixed_prey:
                prey_rnn_states = np.zeros((self.n_rollout_threads, self.all_args.num_good_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                prey_masks = np.ones((self.n_rollout_threads, self.all_args.num_good_agents, 1), dtype=np.float32)
        
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                if not self.all_args.use_fixed_prey:
                    prey_action, prey_rnn_states = self.prey_policy.act(np.concatenate(prey_obs),
                                                    np.concatenate(prey_rnn_states),
                                                    np.concatenate(prey_masks),
                                                    deterministic=True)
                    prey_rnn_states = np.array(np.split(_t2n(prey_rnn_states), self.n_rollout_threads))
                else:
                    prey_action = torch.zeros([self.n_rollout_threads, self.all_args.num_good_agents], dtype=int).to(self.device)
                
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                prey_actions = np.array(np.split(_t2n(prey_action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                tmp_actions = np.concatenate([actions, prey_actions], axis=1)

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[tmp_actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[tmp_actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                dict_obs, dict_rewards, dones, infos = envs.step(actions_env)
                obs, rewards, prey_obs = self._convert(dict_obs, dict_rewards)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
