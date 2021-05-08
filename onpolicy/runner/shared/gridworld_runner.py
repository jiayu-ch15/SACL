    
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

from onpolicy.utils.util import update_linear_schedule
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

            self.init_map_variables() 

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                dict_obs, rewards, dones, infos = self.envs.step(actions)

                for e in range(self.n_rollout_threads):
                    if 'merge_ratio_step' in infos[e].keys():
                        self.merge_explored_ratio_step[e] = infos[e]['merge_ratio_step']
                    for agent_id in range(self.num_agents):
                        agent_k = "agent{}_ratio_step".format(agent_id)
                        if agent_k in infos[e].keys():
                            self.agent_explored_ratio_step[e][agent_id] = infos[e][agent_k]

                data = dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)
                if step == self.episode_length-1:
                    self.init_map_variables() 

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

                if self.env_name == "GridWorld":
                    env_infos = defaultdict(list)
                    for info in infos:
                        env_infos['merge_explored_ratio'].append(info['merge_explored_ratio'])
                        env_infos['num_same_direction'].append(info['num_same_direction'])

                    print("average episode ratio is {}".format(np.mean(env_infos["merge_explored_ratio"])))

                    train_infos["merge_explored_ratio_step"] = np.mean(self.merge_explored_ratio_step)
                    
                    for agent_id in range(self.num_agents):
                        train_infos["agent{}_ratio_step".format(agent_id)] = np.mean(self.agent_explored_ratio_step[:,agent_id])
                        
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
    
    def _convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size, 3), dtype=np.float32)
        obs['vector'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents+4+8), dtype=np.float32)
        obs['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        agent_pos_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            merge_pos_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                index_a_1 = infos[e]['current_agent_pos'][agent_id][0]-1 if infos[e]['current_agent_pos'][agent_id][0]-1 > 0 else 0
                index_a_2 = infos[e]['current_agent_pos'][agent_id][0]+1 if infos[e]['current_agent_pos'][agent_id][0]+1 < self.full_w else self.full_w 
                index_b_1 = infos[e]['current_agent_pos'][agent_id][1]-1 if infos[e]['current_agent_pos'][agent_id][1]-1 > 0 else 0
                index_b_2 = infos[e]['current_agent_pos'][agent_id][1]+1 if infos[e]['current_agent_pos'][agent_id][1]+1 < self.full_h else self.full_h
                agent_pos_map[e , agent_id, int(index_a_1): int(index_a_2), int(index_b_1): int(index_b_2)] = agent_id + 1
                self.all_agent_pos_map[e , agent_id, int(index_a_1): int(index_a_2), int(index_b_1): int(index_b_2)] = agent_id + 1
                if self.use_merge:
                    merge_pos_map[e , int(index_a_1): int(index_a_2), int(index_b_1): int(index_b_2)] = agent_id + 1
                    self.all_merge_pos_map[e , int(index_a_1): int(index_a_2), int(index_b_1): int(index_b_2)] = agent_id + 1

        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                #import pdb;pdb.set_trace()
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

                obs['vector'][e, agent_id] = np.concatenate([np.eye(self.num_agents)[agent_id], np.eye(4)[infos[e]['agent_direction'][agent_id]], np.eye(8)[infos[e]['human_direction'][agent_id]]])

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
        ### Full map consists of 4 channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations

        # Calculating full and local map sizes
        map_size = self.all_args.grid_size
        self.use_merge = self.all_args.use_merge
        self.agent_view_size = self.all_args.agent_view_size
        self.full_w, self.full_h = map_size + 2*self.agent_view_size, map_size + 2*self.agent_view_size
        self.visualize_input = self.all_args.visualize_input
        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(self.num_agents*3, 4, figsize=(10, 2.5), facecolor="whitesmoke")
    
    def init_map_variables(self):
        # Initializing full, merge and local map
        self.merge_explored_ratio_step = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.episode_length
        self.agent_explored_ratio_step = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) * self.episode_length
        self.all_agent_pos_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            self.all_merge_pos_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        
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

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

        obs = self._convert(dict_obs, infos)
        share_obs = self._convert(dict_obs, infos)

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
        eval_episode_rewards = []

        reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        eval_dict_obs, eval_infos = self.eval_envs.reset()
        eval_obs = self._convert(eval_dict_obs, eval_infos)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()

            concat_eval_obs = {}
            for key in eval_obs.keys():
                concat_eval_obs[key] = np.concatenate(eval_obs[key])

            eval_action, eval_rnn_states = self.trainer.policy.act(concat_eval_obs,
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_dict_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_obs = self._convert(eval_dict_obs, eval_infos)

            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
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
   
            for info in infos:
                env_infos['merge_explored_ratio'].append(info['merge_explored_ratio'])
                env_infos['num_same_direction'].append(info['num_same_direction'])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            print("average merge explored ratio is: " + str(np.mean(env_infos['merge_explored_ratio'])))
            print("average num same direction is: " + str(np.mean(env_infos['num_same_direction'])))
            
        if self.all_args.save_gifs:
            ic("rendering....")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif', all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif', all_local_frames, duration=self.all_args.ifi)
            ic("done")