    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio
from icecream import ic
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict, deque
from onpolicy.utils.util import update_linear_schedule, get_shape_from_act_space
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class GridWorldRunner(Runner):
    def __init__(self, config):
        super(GridWorldRunner, self).__init__(config)
        self.init_hyperparameters()
        self.init_map_variables() 

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            self.env_infos = defaultdict(list)

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = self.envs.step(actions)

                data = dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

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
 
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length             
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print("average episode ratio is {}".format(np.mean(self.env_infos["merge_explored_ratio"])))
                                
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def _eval_convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size, 3), dtype=np.float32)
        obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        if self.use_merge:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
        else:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                self.eval_all_agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                if self.use_merge:
                    merge_pos_map[e , infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                    if self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.eval_all_merge_pos_map[e,  infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][self.agent_view_size : self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['global_obs'][e, agent_id, 3] = self.eval_all_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map'][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map'][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_merge_obs'][e, agent_id, 3] = self.all_merge_pos_map[e][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]

                obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]

                i = 0
                obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
                if self.use_merge:
                    for l in range(self.num_agents):
                        if l!= agent_id: 
                            i += 1 
                            obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]                  

        
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return obs

    def _convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size, 3), dtype=np.float32)
        obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        if self.use_merge:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
        else:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                self.all_agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                if self.use_merge:
                    merge_pos_map[e , infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                    if self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.all_merge_pos_map[e,  infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][self.agent_view_size : self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['global_obs'][e, agent_id, 3] = self.all_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map'][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map'][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_merge_obs'][e, agent_id, 3] = self.all_merge_pos_map[e][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]

                obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]

                i = 0
                obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
                if self.use_merge:
                    for l in range(self.num_agents):
                        if l!= agent_id: 
                            i += 1 
                            obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]                  

        
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return obs

    def warmup(self):
        # reset env
        dict_obs, info = self.envs.reset()
        obs = self._convert(dict_obs, info)
        share_obs = self._convert(dict_obs, info)

        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()

        for key in share_obs.keys():
            self.buffer.share_obs[key][0] = share_obs[key].copy()

    def init_hyperparameters(self):
        # Calculating full and local map sizes
        map_size = self.all_args.grid_size
        self.use_merge = self.all_args.use_merge
        self.agent_view_size = self.all_args.agent_view_size
        self.full_w, self.full_h = map_size + 2*self.agent_view_size, map_size + 2*self.agent_view_size
        self.visualize_input = self.all_args.visualize_input
        self.augment = 255 // (np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum())
        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(self.num_agents*3, 4, figsize=(10, 2.5), facecolor="whitesmoke")
 
    def init_map_variables(self):
        # Initializing full, merge and local map
        self.all_agent_pos_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            self.all_merge_pos_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)

    def init_eval_map_variables(self):
        # Initializing full, merge and local map
        self.eval_all_agent_pos_map = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            self.eval_all_merge_pos_map = np.zeros((self.n_eval_rollout_threads, self.full_w, self.full_h), dtype=np.float32)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][step])
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.concatenate(self.buffer.obs[key][step])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                            concat_obs,
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])

        next_values = self.trainer.policy.get_values(concat_share_obs,
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def insert(self, data):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=-1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        obs = self._convert(dict_obs, infos)
        share_obs = self._convert(dict_obs, infos)

        self.all_agent_pos_map[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            self.all_merge_pos_map[dones_env == True] = np.zeros(((dones_env == True).sum(), self.full_w, self.full_h), dtype=np.float32)
     
        for done_env, info in zip(dones_env, infos):
            if done_env:
                self.env_infos['merge_ratio_step'].append(info['merge_ratio_step'])
                self.env_infos['merge_explored_ratio'].append(info['merge_explored_ratio'])
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_ratio_step".format(agent_id)
                    self.env_infos[agent_k].append(info[agent_k])

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)
    
    def visualize_obs(self, fig, ax, obs):
        # individual
        for agent_id in range(self.num_agents * 3):
            sub_ax = ax[agent_id]
            for i in range(4):
                sub_ax[i].clear()
                sub_ax[i].set_yticks([])
                sub_ax[i].set_xticks([])
                sub_ax[i].set_yticklabels([])
                sub_ax[i].set_xticklabels([])
                if agent_id < self.num_agents:
                    sub_ax[i].imshow(obs['global_obs'][0, agent_id, i])
                elif agent_id < self.num_agents*2 and self.use_merge:
                    sub_ax[i].imshow(obs['global_merge_obs'][0, agent_id-self.num_agents, i])
                elif i<3: sub_ax[i].imshow(obs['image'][0, agent_id-self.num_agents*2, :,:,i])
                #elif i < 5:
                    #sub_ax[i].imshow(obs['global_merge_goal'][0, agent_id-self.num_agents, i-4])
                    #sub_ax[i].imshow(obs['gt_map'][0, agent_id - self.num_agents, i-4])
        plt.gcf().canvas.flush_events()
        # plt.pause(0.1)
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    @torch.no_grad()
    def eval(self, total_num_steps):
        action_shape = get_shape_from_act_space(self.eval_envs.action_space[0])
        eval_episode_rewards = []
        eval_env_infos = defaultdict(list)

        reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        self.init_eval_map_variables()
        eval_dict_obs,  eval_infos = self.eval_envs.reset(reset_choose)
        eval_obs = self._eval_convert(eval_dict_obs, eval_infos)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_dones_env = np.zeros(self.n_eval_rollout_threads, dtype=bool)

        while True:
            eval_choose = (eval_dones_env==False)

            if ~np.any(eval_choose):
                break
            eval_actions = np.ones((self.n_eval_rollout_threads, self.num_agents, action_shape)).astype(np.int) * (-1.0)
            
            self.trainer.prep_rollout()

            concat_eval_obs = {}
            for key in eval_obs.keys():
                concat_eval_obs[key] = np.concatenate(eval_obs[key][eval_choose])

            eval_action, eval_rnn_state = self.trainer.policy.act(concat_eval_obs,
                                                np.concatenate(eval_rnn_states[eval_choose]),
                                                np.concatenate(eval_masks[eval_choose]),
                                                deterministic=True)
            
            eval_actions[eval_choose] = np.array(np.split(_t2n(eval_action), (eval_choose == True).sum()))
            eval_rnn_states[eval_choose] = np.array(np.split(_t2n(eval_rnn_state), (eval_choose == True).sum()))

            # Obser reward and next obs
            eval_dict_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_dones_env = np.all(eval_dones, axis=-1)

            eval_obs = self._eval_convert(eval_dict_obs, eval_infos)

            eval_episode_rewards.append(eval_rewards)

            for eval_info, eval_done_env in zip(eval_infos, eval_dones_env):
                if eval_done_env:
                    eval_env_infos['eval_merge_ratio_step'].append(eval_info['merge_ratio_step'])
                    eval_env_infos['eval_merge_explored_ratio'].append(eval_info['merge_explored_ratio'])
                    for agent_id in range(self.num_agents):
                        agent_k = "agent{}_ratio_step".format(agent_id)
                        eval_env_infos["eval_" + agent_k].append(eval_info[agent_k])

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.eval_all_agent_pos_map[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            if self.use_merge:
                self.eval_all_merge_pos_map[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.full_w, self.full_h), dtype=np.float32)
     
        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
 
        print("eval average merge explored ratio is: " + str(np.mean(eval_env_infos['eval_merge_explored_ratio'])))
        print("eval average episode rewards of agent: " + str(np.mean(eval_env_infos['eval_average_episode_rewards'])))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        env_infos = defaultdict(list)

        envs = self.envs
        
        all_frames = []
        all_local_frames = []
        for episode in range(self.all_args.render_episodes):
            ic(episode)
            self.init_map_variables()
            reset_choose = np.ones(self.n_rollout_threads) == 1.0
            dict_obs, infos = envs.reset()
            obs = self._convert(dict_obs, infos)

            if self.all_args.save_gifs:
                image, local_image = envs.render('rgb_array')[0]
                all_frames.append(image)
                all_local_frames.append(local_image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                ic(step)
                calc_start = time.time()

                self.trainer.prep_rollout()

                concat_obs = {}
                for key in obs.keys():
                    concat_obs[key] = np.concatenate(obs[key])

                action, rnn_states = self.trainer.policy.act(concat_obs,
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # Obser reward and next obs
                dict_obs, rewards, dones, infos = envs.step(actions)

                obs = self._convert(dict_obs, infos)
                episode_rewards.append(rewards)
                
                for done, info in zip(dones, infos):
                    if np.all(done):
                        env_infos['merge_ratio_step'].append(info['merge_ratio_step'])
                        env_infos['merge_explored_ratio'].append(info['merge_explored_ratio'])
                        for agent_id in range(self.num_agents):
                            agent_k = "agent{}_ratio_step".format(agent_id)
                            env_infos[agent_k].append(info[agent_k])

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image, local_image = envs.render('rgb_array')[0]
                    all_frames.append(image)
                    all_local_frames.append(local_image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

                if np.all(dones[0]):
                    ic("end")
                    break

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            print("average merge explored ratio is: " + str(np.mean(env_infos['merge_explored_ratio'])))
            print("average merge explored step is: " + str(np.mean(env_infos['merge_ratio_step'])))

        if self.all_args.save_gifs:
            ic("rendering....")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif', all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif', all_local_frames, duration=self.all_args.ifi)
            ic("done")