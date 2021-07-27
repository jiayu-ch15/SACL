import time
import wandb
import os
import gym
import numpy as np
import imageio
from collections import defaultdict, deque
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.envs.habitat.model.model import Neural_SLAM_Module, Local_IL_Policy
from onpolicy.envs.habitat.utils.memory import FIFOMemory
from onpolicy.envs.habitat.utils.frontier import get_frontier, nearest_frontier, max_utility_frontier, bfs_distance, rrt_global_plan
from onpolicy.algorithms.utils.util import init, check
from onpolicy.utils.apf import APF, l2distance
from icecream import ic

def _t2n(x):
    return x.detach().cpu().numpy()

def get_folders(dir, folders):
    get_dir = os.listdir(dir)
    for i in get_dir:          
        sub_dir = os.path.join(dir, i)
        if os.path.isdir(sub_dir): 
            folders.append(sub_dir) 
            get_folders(sub_dir, folders)

class HabitatRunner(Runner):
    def __init__(self, config):
        super(HabitatRunner, self).__init__(config)
        # init parameters
        self.init_hyper_parameters()
        # init variables
        self.init_map_variables() 
        # global policy
        self.init_global_policy() 
        # local policy
        self.init_local_policy()  
        # slam module
        self.init_slam_module()    
    
    def warmup(self):
        # reset env
        self.obs, infos = self.envs.reset()

        self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
        self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
        self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
        self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
        self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
        self.explorable_map = [infos[e]['explorable_map'] for e in range(self.n_rollout_threads)]
        self.merge_explored_gt = [infos[e]['merge_explored_gt'] for e in range(self.n_rollout_threads)]
        self.merge_obstacle_gt = [infos[e]['merge_obstacle_gt'] for e in range(self.n_rollout_threads)]
        for agent_id in range(self.num_agents):
            self.intrinsic_gt[ : , agent_id] = np.array(self.explorable_map)[ : , agent_id]

        # Predict map from frame 1:
        self.run_slam_module(self.obs, self.obs, infos)

        # Compute Global policy input
        self.first_compute_global_input()
        if not self.use_centralized_V:
            self.share_global_input = self.global_input.copy()

        # replay buffer
        for key in self.global_input.keys():
            self.buffer.obs[key][0] = self.global_input[key].copy()

        for key in self.share_global_input.keys():
            self.buffer.share_obs[key][0] = self.share_global_input[key].copy()

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step=0)

        # compute local input
        if self.use_merge_local:
            self.compute_local_input(self.local_merge_map)
        else:
            self.compute_local_input(self.local_map)

        # Output stores local goals as well as the the ground-truth action
        self.local_output = self.envs.get_short_term_goal(self.local_input)
        self.local_output = np.array(self.local_output, dtype = np.long)
        
        self.last_obs = self.obs.copy()
            
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
 
    def run(self):
        # map and pose
        self.init_map_and_pose()

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.max_episode_length // self.n_rollout_threads

        for episode in range(episodes):
            self.init_env_info()
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
    
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            for step in range(self.max_episode_length):

                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                del self.last_obs
                self.last_obs = self.obs.copy()

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)
                self.rewards += reward
                for e in range (self.n_rollout_threads):
                    for key in ['explored_ratio', 'explored_reward', 'repeat_area', 'merge_explored_ratio', 'merge_explored_reward','merge_repeat_area']:
                        if key in infos[e].keys():
                            ic(key)
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                    if 'overlap_ratio' in infos[e].keys():
                        self.env_info['overlap_ratio'][e] = infos[e]['overlap_ratio']
                    if 'merge_explored_ratio_step' in infos[e].keys():
                        self.env_info['merge_explored_ratio_step'][e] = infos[e]['merge_explored_ratio_step']
                    if 'merge_explored_ratio_step_0.95' in infos[e].keys():
                        self.env_info['merge_explored_ratio_step_0.95'][e] = infos[e]['merge_explored_ratio_step_0.95']
                    for agent_id in range(self.num_agents):
                        agent_k = "agent{}_explored_ratio_step".format(agent_id)
                        if agent_k in infos[e].keys():
                            self.env_info['explored_ratio_step'][e][agent_id] = infos[e][agent_k]
                              
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                # Reinitialize variables when episode ends
                if step == self.max_episode_length - 1:
                    self.init_map_and_pose()
                    del self.last_obs
                    self.last_obs = self.obs.copy()
                    self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
                    self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
                    self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
                    self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
                    self.explorable_map = [infos[e]['explorable_map'] for e in range(self.n_rollout_threads)]
                    self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
                    self.merge_explored_gt = [infos[e]['merge_explored_gt'] for e in range(self.n_rollout_threads)]
                    self.merge_obstacle_gt = [infos[e]['merge_obstacle_gt'] for e in range(self.n_rollout_threads)]
                    for agent_id in range(self.num_agents):
                        self.intrinsic_gt[:, agent_id] = np.array(self.explorable_map)[:, agent_id]

                # Neural SLAM Module
                if self.train_slam:
                    self.insert_slam_module(infos)
                
                self.run_slam_module(self.last_obs, self.obs, infos, True)
                self.update_local_map()
                self.update_map_and_pose(False)
                for agent_id in range(self.num_agents):
                    _, self.local_merge_map[:, agent_id] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, agent_id)

                # Global Policy
                if local_step == self.num_local_steps - 1:
                    # For every global step, update the full and local maps
                    self.update_map_and_pose()
                    self.compute_global_input()
                    data = dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                    # insert data into buffer
                    self.insert_global_policy(data)
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step = global_step + 1)

                # Local Policy
                if self.use_merge_local:
                    self.compute_local_input(self.local_merge_map)
                else:
                    self.compute_local_input(self.local_map)

                # Output stores local goals as well as the the ground-truth action
                self.local_output = self.envs.get_short_term_goal(self.local_input)
                self.local_output = np.array(self.local_output, dtype = np.long)

                # Start Training
                torch.set_grad_enabled(True)

                # Train Neural SLAM Module
                if self.train_slam and len(self.slam_memory) > self.slam_batch_size:
                    self.train_slam_module()
                    
                # Train Local Policy
                if self.train_local and (local_step + 1) % self.local_policy_update_freq == 0:
                    self.train_local_policy()
                    
                # Train Global Policy
                if global_step % self.episode_length == self.episode_length - 1 \
                        and local_step == self.num_local_steps - 1:
                    self.train_global_policy()
                    
                # Finish Training
                torch.set_grad_enabled(False)
                
            # post process
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            
            self.convert_info()
            print("average episode merge explored reward is {}".format(np.mean(self.env_infos["sum_merge_explored_reward"])))
            print("average episode merge explored ratio is {}".format(np.mean(self.env_infos['sum_merge_explored_ratio'])))
            print("average episode merge repeat ratio is {}".format(np.mean(self.env_infos['sum_merge_repeat_ratio'])))

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

                self.log_env(self.train_slam_infos, total_num_steps)
                self.log_env(self.train_local_infos, total_num_steps)
                self.log_env(self.train_global_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save_slam_model(total_num_steps)
                self.save_global_model(total_num_steps)
                self.save_local_model(total_num_steps)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_hyper_parameters(self):
        self.map_size_cm = self.all_args.map_size_cm
        self.map_resolution = self.all_args.map_resolution
        self.global_downscaling = self.all_args.global_downscaling

        self.frame_width = self.all_args.frame_width
       
        self.load_local = self.all_args.load_local
        self.load_slam = self.all_args.load_slam
        self.train_local = self.all_args.train_local
        self.train_slam = self.all_args.train_slam
        
        self.slam_memory_size = self.all_args.slam_memory_size
        self.slam_batch_size = self.all_args.slam_batch_size
        self.slam_iterations = self.all_args.slam_iterations
        self.slam_lr = self.all_args.slam_lr
        self.slam_opti_eps = self.all_args.slam_opti_eps

        self.use_local_recurrent_policy = self.all_args.use_local_recurrent_policy
        self.local_hidden_size = self.all_args.local_hidden_size
        self.local_lr = self.all_args.local_lr
        self.local_opti_eps = self.all_args.local_opti_eps

        self.proj_loss_coeff = self.all_args.proj_loss_coeff
        self.exp_loss_coeff = self.all_args.exp_loss_coeff
        self.pose_loss_coeff = self.all_args.pose_loss_coeff

        self.local_policy_update_freq = self.all_args.local_policy_update_freq
        self.num_local_steps = self.all_args.num_local_steps
        self.max_episode_length = self.all_args.max_episode_length
        self.render_merge = self.all_args.render_merge
        self.visualize_input = self.all_args.visualize_input
        self.use_intrinsic_reward = self.all_args.use_intrinsic_reward
        self.use_delta_reward = self.all_args.use_delta_reward
        self.use_abs_orientation = self.all_args.use_abs_orientation
        self.use_center = self.all_args.use_center
        self.use_resnet = self.all_args.use_resnet
        self.use_merge = self.all_args.use_merge
        self.use_single = self.all_args.use_single
        self.use_merge_local = self.all_args.use_merge_local
        self.use_oracle = self.all_args.use_oracle

    def init_map_variables(self):
        ### Full map consists of 4 channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations

        # Calculating full and local map sizes
        map_size = self.map_size_cm // self.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w, self.local_h = int(self.full_w / self.global_downscaling), \
                        int(self.full_h / self.global_downscaling)
        if self.use_resnet:
            self.res_w = self.res_h = 224

        # Initializing full, merge and local map
        self.full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.local_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h), dtype=np.float32)
        
        # Initial full and local pose
        self.full_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)
        self.local_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)

        # Origin of local map
        self.origins = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)

        # Local Map Boundaries
        self.lmb = np.zeros((self.n_rollout_threads, self.num_agents, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundariesinit_map_and_pose
        self.planner_pose_inputs = np.zeros((self.n_rollout_threads, self.num_agents, 7), dtype=np.float32)
        
        # each agent rotation
        self.other_agent_rotation = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        # ft
        self.ft_merge_map = np.zeros((self.n_rollout_threads, 2, self.full_w, self.full_h), dtype = np.float32) # only explored and obstacle
        self.ft_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype = np.int32)
        self.ft_pre_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype = np.int32)
        self.ft_last_merge_explored_ratio = np.zeros((self.n_rollout_threads, 1), dtype= np.float32)
        self.ft_mask = np.ones((self.full_w, self.full_h), dtype=np.int32)
        self.ft_go_steps = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype= np.int32)
        self.ft_map = None
        self.ft_lx = None
        self.ft_ly = None
          
    def init_map_and_pose(self):
        self.full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.full_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)
        self.merge_goal_trace = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        self.intrinsic_gt = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        self.full_pose[:, :, :2] = self.map_size_cm / 100.0 / 2.0

        locs = self.full_pose
        self.planner_pose_inputs[:, :, :3] = locs
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.full_map[e, a, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1

                self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                    (self.local_w, self.local_h),
                                                    (self.full_w, self.full_h))

                self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a].copy()
                self.origins[e, a] = [self.lmb[e, a, 2] * self.map_resolution / 100.0,
                                self.lmb[e, a, 0] * self.map_resolution / 100.0, 0.]

        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
                self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]

    def init_global_policy(self):
        self.best_gobal_reward = -np.inf
        length = 1
        # ppo network log info
        self.train_global_infos = {}
        self.train_global_infos['value_loss']= deque(maxlen=length)
        self.train_global_infos['policy_loss']= deque(maxlen=length)
        self.train_global_infos['dist_entropy'] = deque(maxlen=length)
        self.train_global_infos['actor_grad_norm'] = deque(maxlen=length)
        self.train_global_infos['critic_grad_norm'] = deque(maxlen=length)
        self.train_global_infos['ratio'] = deque(maxlen=length)
        
        # env info
        self.env_infos = {}
        
        self.env_infos['sum_explored_ratio'] = deque(maxlen=length)
        self.env_infos['sum_explored_reward'] = deque(maxlen=length)
        self.env_infos['sum_repeat_area'] = deque(maxlen=length)
        self.env_infos['sum_mer_explored_reward'] = deque(maxlen=length)
        self.env_infos['sum_merge_explored_ratio'] = deque(maxlen=length)
        self.env_infos['sum_merge_repeat_area'] = deque(maxlen=length)
        self.env_infos['sum_merge_explored_reward'] = deque(maxlen=length)
        self.env_infos['merge_explored_ratio_step'] = deque(maxlen=length)
        self.env_infos['merge_explored_ratio_step_0.95'] = deque(maxlen=length)
        self.env_infos['invalid_merge_explored_ratio_step_num'] = deque(maxlen=length)
        self.env_infos['invalid_merge_map_num'] = deque(maxlen=length)
        self.env_infos['merge_success_rate'] = deque(maxlen=length)
        self.env_infos['max_sum_merge_explored_ratio'] = deque(maxlen=length)
        self.env_infos['min_sum_merge_explored_ratio'] = deque(maxlen=length)
        self.env_infos['explored_ratio_step'] = deque(maxlen=length) 
        self.env_infos['overlap_ratio'] = deque(maxlen=length) 
        if self.use_eval:
            self.env_infos['sum_path_length'] = deque(maxlen=length)
            self.auc_infos = {}
            self.auc_infos['merge_auc'] = np.zeros((self.all_args.eval_episodes, self.n_rollout_threads, self.max_episode_length), dtype=np.float32)
            self.auc_infos['agent_auc'] = np.zeros((self.all_args.eval_episodes,  self.n_rollout_threads, self.num_agents, self.max_episode_length), dtype=np.float32)

        self.global_input = {}
        if self.use_resnet:
            if self.use_merge:
                self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.res_w, self.res_h), dtype=np.float32)
            # self.global_input['global_merge_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.res_w, self.res_h), dtype=np.float32)
            if self.use_single:
                self.global_input['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.res_w, self.res_h), dtype=np.float32)
        else:
            if self.use_merge:
                self.global_input['global_merge_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.local_w, self.local_h), dtype=np.float32)
            if self.use_single:
                self.global_input['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.local_w, self.local_h), dtype=np.float32)
            # self.global_input['global_merge_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.local_w, self.local_h), dtype=np.float32)
        self.global_input['global_orientation'] = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.long)
        self.global_input['other_global_orientation'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents-1), dtype=np.long)
        self.global_input['vector'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
        self.share_global_input = self.global_input.copy()
        if self.use_centralized_V:
            if self.use_resnet:
                self.share_global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.res_w, self.res_h), dtype=np.float32)
            else:
                self.share_global_input['gt_map'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, self.local_w, self.local_h), dtype=np.float32)
        
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32) 
        self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

        self.first_compute = True
        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(self.num_agents*2, 8, figsize=(10, 2.5), facecolor="whitesmoke")

    def init_local_policy(self):
        self.best_local_loss = np.inf

        self.train_local_infos = {}
        self.train_local_infos['local_policy_loss'] = deque(maxlen=1000)
        
        # Local policy
        self.local_masks = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.local_rnn_states = torch.zeros((self.n_rollout_threads, self.num_agents, self.local_hidden_size)).to(self.device)
        
        local_observation_space = gym.spaces.Box(0, 255, (3,
                                                    self.frame_width,
                                                    self.frame_width), dtype='uint8')
        local_action_space = gym.spaces.Discrete(3)

        self.local_policy = Local_IL_Policy(local_observation_space.shape, local_action_space.n,
                               recurrent=self.use_local_recurrent_policy,
                               hidden_size=self.local_hidden_size,
                               deterministic=self.all_args.use_local_deterministic,
                               device=self.device)
        
        if self.load_local != "0":
            print("Loading local {}".format(self.load_local))
            state_dict = torch.load(self.load_local, map_location='cuda:0')
            self.local_policy.load_state_dict(state_dict)

        if not self.train_local:
            self.local_policy.eval()
        else:
            self.local_policy_loss = 0
            self.local_optimizer = torch.optim.Adam(self.local_policy.parameters(), lr=self.local_lr, eps=self.local_opti_eps)

    def init_slam_module(self):
        self.best_slam_cost = 10000
        
        self.train_slam_infos = {}
        self.train_slam_infos['costs'] = deque(maxlen=1000)
        self.train_slam_infos['exp_costs'] = deque(maxlen=1000)
        self.train_slam_infos['pose_costs'] = deque(maxlen=1000)
        
        self.nslam_module = Neural_SLAM_Module(self.all_args, device=self.device)
        
        if self.load_slam != "0":
            print("Loading slam {}".format(self.load_slam))
            state_dict = torch.load(self.load_slam, map_location='cuda:0')
            self.nslam_module.load_state_dict(state_dict)
        
        if not self.train_slam:
            self.nslam_module.eval()
        else:
            self.slam_memory = FIFOMemory(self.slam_memory_size)
            self.slam_optimizer = torch.optim.Adam(self.nslam_module.parameters(), lr=self.slam_lr, eps=self.slam_opti_eps)

    def init_env_info(self):
        self.env_info = {}
        self.env_info['sum_explored_ratio'] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.env_info['sum_mer_explored_reward'] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.env_info['sum_explored_reward'] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.env_info['sum_merge_explored_ratio'] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        self.env_info['sum_repeat_area'] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.env_info['sum_merge_repeat_area'] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        self.env_info['sum_merge_explored_reward'] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        self.env_info['explored_ratio_step'] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) * self.max_episode_length
        self.env_info['merge_explored_ratio_step'] = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.max_episode_length
        self.env_info['merge_explored_ratio_step_0.95'] = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.max_episode_length
        self.env_info['overlap_ratio'] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        if self.use_eval:
            self.env_info['sum_path_length'] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
    def convert_info(self):
        for k, v in self.env_info.items():
            if k == "explored_ratio_step":
                self.env_infos[k].append(v)
                for agent_id in range(self.num_agents):
                    print("agent{}_{}: {}/{}".format(agent_id, k, np.mean(v[:, agent_id]), self.max_episode_length))
                print('minimal agent {}: {}/{}'.format(k, np.min(v), self.max_episode_length))
            elif k == "merge_explored_ratio_step":
                print('invaild {} map num: {}/{}'.format(k, (v == self.max_episode_length).sum(), self.n_rollout_threads))
                self.env_infos['invalid_merge_map_num'].append((v == self.max_episode_length).sum())
                self.env_infos['merge_success_rate'].append((v != self.max_episode_length).sum() / self.n_rollout_threads)
                if (v == self.max_episode_length).sum() > 0:
                    scene_id = np.argwhere(v == self.max_episode_length).reshape((v == self.max_episode_length).sum())
                    if self.all_args.use_same_scene:
                        print('invaild {} map id: {}'.format(k, self.scene_id[scene_id[0]]))
                    else:
                        for i in range(len(scene_id)):
                            print('invaild {} map id: {}'.format(k, self.scene_id[scene_id[i]]))
                v_copy = v.copy()
                v_copy[v == self.max_episode_length] = np.nan
                self.env_infos[k].append(v)
                print('mean valid {}: {}'.format(k, np.nanmean(v_copy)))
            else:
                self.env_infos[k].append(v)
                if k == 'sum_merge_explored_ratio':       
                    self.env_infos['max_sum_merge_explored_ratio'].append(np.max(v))
                    self.env_infos['min_sum_merge_explored_ratio'].append(np.min(v))
                    ic(np.mean(v))

    def insert_slam_module(self, infos):
        # Add frames to memory
        for agent_id in range(self.num_agents):
            for env_idx in range(self.n_rollout_threads):
                env_poses = infos[env_idx]['sensor_pose'][agent_id]
                env_gt_fp_projs = np.expand_dims(infos[env_idx]['fp_proj'][agent_id], 0)
                env_gt_fp_explored = np.expand_dims(infos[env_idx]['fp_explored'][agent_id], 0)
                env_gt_pose_err = infos[env_idx]['pose_err'][agent_id]
                self.slam_memory.push(
                    (self.last_obs[env_idx][agent_id], self.obs[env_idx][agent_id], env_poses),
                    (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

    def run_slam_module(self, last_obs, obs, infos, build_maps=True):
        for a in range(self.num_agents):
            poses = np.array([infos[e]['sensor_pose'][a] for e in range(self.n_rollout_threads)])

            _, _, self.local_map[:, a, 0, :, :], self.local_map[:, a, 1, :, :], _, self.local_pose[:, a, :] = \
                self.nslam_module(last_obs[:, a, :, :, :], obs[:, a, :, :, :], poses, 
                                self.local_map[:, a, 0, :, :],
                                self.local_map[:, a, 1, :, :], 
                                self.local_pose[:, a, :],
                                build_maps = build_maps)
    def oracle_transform(self, inputs, trans, rotation, agent_trans, agent_rotation, a):
        merge_map = np.zeros((self.n_rollout_threads, 4, self.full_w, self.full_h), dtype=np.float32)
        local_merge_map = np.zeros((self.n_rollout_threads, 4, self.local_w, self.local_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                output = torch.from_numpy(inputs[e, agent_id, 2:])
                n_rotated = F.grid_sample(output.unsqueeze(0).float(), rotation[e][agent_id].float(), align_corners=True)
                n_map = F.grid_sample(n_rotated.float(), trans[e][agent_id].float(), align_corners=True)
                agent_merge_map = n_map[0, :, :, :].numpy()

                (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[0, :, :], axis=None), agent_merge_map[0, :, :].shape)
                agent_merge_map[0, :, :] = np.zeros((self.full_h, self.full_w), dtype=np.float32)
                if self.first_compute:
                    agent_merge_map[0, index_a - 1: index_a + 2, index_b - 1: index_b + 2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                else: 
                    agent_merge_map[0, index_a - 2: index_a + 3, index_b - 2: index_b + 3] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
            
                trace = np.zeros((self.full_h, self.full_w), dtype=np.float32)
                #trace[0][agent_merge_map[0] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                #trace[1][agent_merge_map[1] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                trace[agent_merge_map[1] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                #agent_merge_map[0:2] = trace[0:2]
                agent_merge_map[1] = trace
                merge_map[e,2:] += agent_merge_map
            
            
            agent_n_trans = F.grid_sample(torch.from_numpy(merge_map[e,2:]).unsqueeze(0).float(), agent_trans[e][a].float(), align_corners=True)      
            merge_map[e,2:] = F.grid_sample(agent_n_trans.float(), agent_rotation[e][a].float(), align_corners=True)[0, :, :, :].numpy()
            merge_map[e,0] = self.merge_obstacle_gt[e,a]
            merge_map[e,1] = self.merge_explored_gt[e,a]

            local_merge_map[e, :2] = merge_map[e, :2, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]].copy()
            local_merge_map[e, 2:] = self.local_map[e, a, 2:].copy()
        return merge_map, local_merge_map
    
    def transform(self, inputs, trans, rotation, agent_trans, agent_rotation, a):
        merge_map = np.zeros((self.n_rollout_threads, 4, self.full_w, self.full_h), dtype=np.float32)
        local_merge_map = np.zeros((self.n_rollout_threads, 4, self.local_w, self.local_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                output = torch.from_numpy(inputs[e, agent_id])
                n_rotated = F.grid_sample(output.unsqueeze(0).float(), rotation[e][agent_id].float(), align_corners=True)
                n_map = F.grid_sample(n_rotated.float(), trans[e][agent_id].float(), align_corners=True)
                agent_merge_map = n_map[0, :, :, :].numpy()

                (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[2, :, :], axis=None), agent_merge_map[2, :, :].shape)
                agent_merge_map[2, :, :] = np.zeros((self.full_h, self.full_w), dtype=np.float32)
                if self.first_compute:
                    agent_merge_map[2, index_a - 1: index_a + 2, index_b - 1: index_b + 2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                else: 
                    agent_merge_map[2, index_a - 2: index_a + 3, index_b - 2: index_b + 3] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
            
                trace = np.zeros((self.full_h, self.full_w), dtype=np.float32)
                #trace[0][agent_merge_map[0] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                #trace[1][agent_merge_map[1] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                trace[agent_merge_map[3] > 0.2] = (agent_id + 1)/np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
                #agent_merge_map[0:2] = trace[0:2]
                agent_merge_map[3] = trace
                merge_map[e] += agent_merge_map
            
            
            agent_n_trans = F.grid_sample(torch.from_numpy(merge_map[e]).unsqueeze(0).float(), agent_trans[e][a].float(), align_corners=True)      
            merge_map[e] = F.grid_sample(agent_n_trans.float(), agent_rotation[e][a].float(), align_corners=True)[0, :, :, :].numpy()
            for i in range(2):
                merge_map[ e, i][merge_map[ e, i]>1] = 1
                merge_map[ e, i][merge_map[ e, i]<0.2] = 0

            local_merge_map[e, :2] = merge_map[e, :2, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]].copy()
            local_merge_map[e, 2:] = self.local_map[e, a, 2:].copy()
        return merge_map, local_merge_map

    def center_transform(self, inputs, a):
        merge_map = np.zeros((self.n_rollout_threads, 4, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            for i in range(4):
                r, c = self.full_pose[e, a,:2]
                r, c =[int(r * 100.0 / self.map_resolution), int(c * 100.0 / self.map_resolution)]
                M = np.float32([[1, 0, self.full_w//2 - r], [0, 1, self.full_h//2 - c]])
                merge_map[e, i] = cv2.warpAffine(inputs[e, i], M, (self.full_w, self.full_h))
                #current_explored_map = cv2.warpAffine(current_explored_map, M, (self.size[0]*3, self.size[0]*3))
                # M = cv2.getRotationMatrix2D((self.full_w//2, self.full_h//2), self.full_pose[e, a, 2], 1) 
                # n_map[i] = cv2.warpAffine(n_move, M, (self.full_w, self.full_h)) 
                #current_explored_map = cv2.warpAffine(current_explored_map, M, (self.size[0]*3, self.size[0]*3)) 
        return merge_map

    def center_gt_transform(self, inputs, a):
        merge_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            r, c = self.full_pose[e, a,:2]
            r, c =[int(r * 100.0 / self.map_resolution), int(c * 100.0 / self.map_resolution)]
            M = np.float32([[1, 0, self.full_w//2 - r], [0, 1, self.full_h//2 - c]])
            merge_map[e] = cv2.warpAffine(inputs[e], M, (self.full_w, self.full_h))
        return merge_map
    
    def point_transform(self, point, agent_id):
        merge_point_map = np.zeros((self.n_rollout_threads, 1, self.full_w, self.full_h), dtype=np.float32)
        
        for e in range(self.n_rollout_threads):
            merge_point_map[e, 0, int(point[e, agent_id, 0]*self.local_w+self.lmb[e, agent_id, 0]-2): int(point[e, agent_id, 0]*self.local_w+self.lmb[e, agent_id, 0]+3), \
                int(point[e, agent_id, 1]*self.local_w+self.lmb[e, agent_id, 2]-2): int(point[e, agent_id, 1]*self.local_w+self.lmb[e, agent_id, 2]+3)] += 1
            
        #self.merge_goal_trace[:, agent_id, :, :] +=  merge_point_map[:, 0]
        #merge_point_map[:, 1] = self.merge_goal_trace[:, agent_id, :, :]
        
        return merge_point_map

    '''def exp_transform(self, agent_id, inputs, trans, rotation):
        trans = check(trans)
        rotation = check(rotation)
        explorable_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        for e in range(self.n_rollout_threads):
            output = torch.from_numpy(inputs[e])
            n_rotated = F.grid_sample(output.unsqueeze(0).unsqueeze(0).float(), rotation[e][agent_id].float(), align_corners=True)
            n_map = F.grid_sample(n_rotated.float(), trans[e][agent_id].float(), align_corners=True)
            explorable_map[e] = n_map[0, 0].numpy()
        return explorable_map'''

    
    def first_compute_global_input(self):
        locs = self.local_pose
        self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        if self.use_center:
            self.transform_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.local_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h), dtype=np.float32)
        # global_goal_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.full_w, self.full_h), dtype=np.float32)
        self.trans_point = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, 2))
        for a in range(self.num_agents):
            for e in range(self.n_rollout_threads):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.local_map[e, a, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1
                self.global_input['global_orientation'][e, a, 0] = int((locs[e, a, 2] + 180.0) / 5.)
                self.other_agent_rotation[e, a, 0] = locs[e, a, 2]
                self.global_input['vector'][e, a] = np.eye(self.num_agents)[a]
            if self.use_oracle:
                if self.use_center:
                    self.transform_map[:, a], self.local_merge_map[:, a] = self.oracle_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    self.merge_map[:, a] = self.center_transform(self.transform_map[:,a], a)
                    full_map[:, a] = self.center_transform(self.full_map[:, a], a)
                else:
                    self.merge_map[:, a], self.local_merge_map[:, a] = self.oracle_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    full_map[:, a] = self.full_map[:, a].copy()
            else:
                if self.use_center:
                    self.transform_map[:, a], self.local_merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    self.merge_map[:, a] = self.center_transform(self.transform_map[:,a], a)
                    full_map[:, a] = self.center_transform(self.full_map[:, a], a)
                else:
                    self.merge_map[:, a], self.local_merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    full_map[:, a] = self.full_map[:, a].copy()
            #self.global_input['global_obs'][:, a, 0:4] = self.local_map[:, a].copy()
            #self.global_input['global_obs'][:, a, 4:8] = (nn.MaxPool2d(self.global_downscaling)(check(self.full_map[:, a]))).numpy()
            '''if self.use_center:
                xxx = self.point_transform(self.global_goal, a)
                global_goal_map[:, a] = self.center_gt_transform(xxx, a)
            else:
                global_goal_map[:, a] = self.point_transform(self.global_goal, a)'''
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                i = 0
                for l in range(self.num_agents):
                    if l!= a:   
                        if self.use_abs_orientation:
                            self.global_input['other_global_orientation'][e, a, i] = int((self.other_agent_rotation[e, l, 0] + 180.0) / 5.)
                        else:
                            self.global_input['other_global_orientation'][e, a, i] = int(((self.other_agent_rotation[e, l, 0] - self.other_agent_rotation[e, a, 0] + 360.0 + 180.0) % 360) / 5.)
                        i += 1

        for a in range(self.num_agents): # TODO @CHAO
            if self.use_resnet:
                for e in range(self.n_rollout_threads):
                    for i in range(4):
                        if self.use_merge:
                            self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.res_h, self.res_w))
                            self.global_input['global_merge_obs'][e, a, i+4] = cv2.resize(self.merge_map[e, a, i], (self.res_h, self.res_w))
                        if self.use_single:
                            self.global_input['global_obs'][e, a, i] = cv2.resize(self.local_map[e, a, i], (self.res_h, self.res_w))
                            self.global_input['global_obs'][e, a, i+4] = cv2.resize(full_map[e, a, i], (self.res_h, self.res_w))
                    # self.global_input['global_merge_goal'][:, a, 0, :, :] = (nn.MaxPool2d(self.global_downscaling)(check(global_goal_map[:, a, 1]))).numpy()
                    if self.use_centralized_V:
                        #self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(self.exp_transform(a, np.array(self.explorable_map)[:, a], np.array(self.agent_trans)[:,a], np.array(self.agent_rotation)[:,a])))).numpy()
                        if self.use_center:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(self.center_gt_transform(np.array(self.explorable_map)[e, a], a), (self.res_h, self.res_w))
                        else:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(np.array(self.explorable_map)[e, a], (self.res_h, self.res_w))
            else:
                if self.use_merge:
                    self.global_input['global_merge_obs'][:, a, 0:4] = self.local_merge_map[:, a]
                    self.global_input['global_merge_obs'][:, a, 4:] = (nn.MaxPool2d(self.global_downscaling)(check(self.merge_map[:, a]))).numpy()
                if self.use_single:
                    self.global_input['global_obs'][:, a, 0:4] = self.local_map[:, a]
                    self.global_input['global_obs'][:, a, 4:] = (nn.MaxPool2d(self.global_downscaling)(check(full_map[:, a]))).numpy()
                # self.global_input['global_merge_goal'][:, a, 0, :, :] = (nn.MaxPool2d(self.global_downscaling)(check(global_goal_map[:, a, 1]))).numpy()
                if self.use_centralized_V:
                    #self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(self.exp_transform(a, np.array(self.explorable_map)[:, a], np.array(self.agent_trans)[:,a], np.array(self.agent_rotation)[:,a])))).numpy()
                    if self.use_center:
                        self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(self.center_gt_transform(np.array(self.explorable_map)[:, a], a)))).numpy()
                    else:
                        self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(np.array(self.explorable_map)[:, a]))).numpy()
        
        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()
        
        self.first_compute = False
        
    def compute_local_action(self):
        local_action = torch.empty(self.n_rollout_threads, self.num_agents)
        for a in range(self.num_agents):
            local_goals = self.local_output[:, a, :-1]

            if self.train_local:
                torch.set_grad_enabled(True)
        
            action, action_prob, self.local_rnn_states[:, a] =\
                self.local_policy(self.obs[:, a],
                                    self.local_rnn_states[:, a],
                                    self.local_masks[:, a],
                                    extras=local_goals)

            if self.train_local:
                action_target = check(self.local_output[:, a, -1]).to(self.device)
                self.local_policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                torch.set_grad_enabled(False)
            
            local_action[:, a] = action.cpu()

        return local_action
   
    def compute_local_input(self, map):
        self.local_input = []
        for e in range(self.n_rollout_threads):
            p_input = defaultdict(list)
            for a in range(self.num_agents):
                p_input['goal'].append([int(self.global_goal[e, a][0] * self.local_w), int(self.global_goal[e, a][1] * self.local_h)])
                p_input['map_pred'].append(map[e, a, 0, :, :].copy())
                p_input['exp_pred'].append(map[e, a, 1, :, :].copy())
                p_input['pose_pred'].append(self.planner_pose_inputs[e, a].copy())
            self.local_input.append(p_input)
    
    def compute_global_input(self):
        locs = self.local_pose
        self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        if self.use_center:
            self.transform_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        self.local_merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h), dtype=np.float32)
        self.trans_point = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, 2))
        # global_goal_map = np.zeros((self.n_rollout_threads, self.num_agents, 2, self.full_w, self.full_h), dtype=np.float32)
        for a in range(self.num_agents):
            for e in range(self.n_rollout_threads):
                self.global_input['global_orientation'][e, a, 0] = int((locs[e, a, 2] + 180.0) / 5.)
                self.global_input['vector'][e, a] = np.eye(self.num_agents)[a]
                self.other_agent_rotation[e, a, 0] = locs[e, a, 2]
            if self.use_oracle:
                if self.use_center:
                    self.transform_map[:, a], self.local_merge_map[:, a] = self.oracle_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    self.merge_map[:, a] = self.center_transform(self.transform_map[:,a], a)
                    full_map[:, a] = self.center_transform(self.full_map[:, a], a)
                else:
                    self.merge_map[:, a], self.local_merge_map[:, a] = self.oracle_transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    full_map[:, a] = self.full_map[:, a].copy()
            else:
                if self.use_center:
                    self.transform_map[:, a], self.local_merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    self.merge_map[:, a] = self.center_transform(self.transform_map[:, a], a)
                    full_map[:, a] = self.center_transform(self.full_map[:, a], a)
                else:
                    self.merge_map[:, a], self.local_merge_map[:, a] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                    full_map[:, a] = self.full_map[:, a].copy()
            #self.global_input['global_obs'][:, a, 0:4] = self.local_map[:, a]
            #self.global_input['global_obs'][:, a, 4:8] = (nn.MaxPool2d(self.global_downscaling)(check(self.full_map[:, a]))).numpy()
        
            '''if self.use_center:
                xxx = self.point_transform(self.global_goal, a)
                global_goal_map[:, a] = self.center_gt_transform(xxx, a)
            else:
                global_goal_map[:, a] = self.point_transform(self.global_goal, a)'''
        
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                i = 0
                for l in range(self.num_agents):
                    if l!= a:   
                        if self.use_abs_orientation:
                            self.global_input['other_global_orientation'][e, a, i] = int((self.other_agent_rotation[e, l, 0] + 180.0) / 5.)
                        else:
                            self.global_input['other_global_orientation'][e, a, i] = int(((self.other_agent_rotation[e, l, 0] - self.other_agent_rotation[e, a, 0] + 360.0 + 180.0) % 360) / 5.)
                        i += 1 
        
        for a in range(self.num_agents):
            if self.use_resnet:
                for e in range(self.n_rollout_threads):
                    for i in range(4):
                        if self.use_merge:
                            self.global_input['global_merge_obs'][e, a, i] = cv2.resize(self.local_merge_map[e, a, i], (self.res_h, self.res_w))
                            self.global_input['global_merge_obs'][e, a, i+4] = cv2.resize(self.merge_map[e, a, i], (self.res_h, self.res_w))
                        if self.use_single:
                            self.global_input['global_obs'][e, a, i] = cv2.resize(self.local_map[e, a, i], (self.res_h, self.res_w))
                            self.global_input['global_obs'][e, a, i+4] = cv2.resize(full_map[e, a, i], (self.res_h, self.res_w))
                    # self.global_input['global_merge_goal'][:, a, 0, :, :] = (nn.MaxPool2d(self.global_downscaling)(check(global_goal_map[:, a, 1]))).numpy()
                    if self.use_centralized_V:
                        #self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(self.exp_transform(a, np.array(self.explorable_map)[:, a], np.array(self.agent_trans)[:,a], np.array(self.agent_rotation)[:,a])))).numpy()
                        if self.use_center:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(self.center_gt_transform(np.array(self.explorable_map)[e, a], a), (self.res_h, self.res_w))
                        else:
                            self.share_global_input['gt_map'][e, a, 0] = cv2.resize(np.array(self.explorable_map)[e, a], (self.res_h, self.res_w))
            else:
                if self.use_merge:
                    self.global_input['global_merge_obs'][:, a, 0:4] = self.local_merge_map[:, a]
                    self.global_input['global_merge_obs'][:, a, 4:] = (nn.MaxPool2d(self.global_downscaling)(check(self.merge_map[:, a]))).numpy()
                if self.use_single:
                    self.global_input['global_obs'][:, a, 0:4] = self.local_map[:, a]
                    self.global_input['global_obs'][:, a, 4:] = (nn.MaxPool2d(self.global_downscaling)(check(full_map[:, a]))).numpy()
                # self.global_input['global_merge_goal'][:, a, 0, :, :] = (nn.MaxPool2d(self.global_downscaling)(check(global_goal_map[:, a, 1]))).numpy()
                if self.use_centralized_V:
                    #self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(self.exp_transform(a, np.array(self.explorable_map)[:, a], np.array(self.agent_trans)[:,a], np.array(self.agent_rotation)[:,a])))).numpy()
                    if self.use_center:
                        self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(self.center_gt_transform(np.array(self.explorable_map)[:, a], a)))).numpy()
                    else:
                        self.share_global_input['gt_map'][:, a, 0] = (nn.MaxPool2d(self.global_downscaling)(check(np.array(self.explorable_map)[:, a]))).numpy()

        for key in self.global_input.keys():
            self.share_global_input[key] = self.global_input[key].copy()

        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, self.global_input)
            
    def compute_global_goal(self, step):
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
        
        # Compute planner inputs
        self.global_goal = np.array(np.split(_t2n(nn.Sigmoid()(action)), self.n_rollout_threads))
 
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def eval_compute_global_goal(self, rnn_states):      
        self.trainer.prep_rollout()

        concat_obs = {}
        for key in self.global_input.keys():
            concat_obs[key] = np.concatenate(self.global_input[key])
        
        actions, rnn_states = self.trainer.policy.act(concat_obs,
                                    np.concatenate(rnn_states),
                                    np.concatenate(self.global_masks),
                                    deterministic=True)
        
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        # Compute planner inputs
        self.global_goal = np.array(np.split(_t2n(nn.Sigmoid()(actions)), self.n_rollout_threads))
        
        return rnn_states

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        concat_share_obs = {}
        concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.concatenate(self.buffer.obs[key][-1])

        next_values = self.trainer.policy.get_values(concat_share_obs,
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def update_local_map(self):        
        locs = self.local_pose
        self.planner_pose_inputs[:, :, :3] = locs + self.origins
        self.local_map[:, :, 2, :, :].fill(0.)  # Resetting current location channel
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.local_map[e, a, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1

    def update_map_and_pose(self, envs = 1000, update = True):
        if envs > self.n_rollout_threads:
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]] = self.local_map[e, a]
                    if update:
                        self.full_pose[e, a] = self.local_pose[e, a] + self.origins[e, a]

                        locs = self.full_pose[e, a]
                        r, c = locs[1], locs[0]
                        loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                        int(c * 100.0 / self.map_resolution)]

                        self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                            (self.local_w, self.local_h),
                                                            (self.full_w, self.full_h))

                        self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a].copy()
                        self.origins[e, a] = [self.lmb[e, a][2] * self.map_resolution / 100.0,
                                        self.lmb[e, a][0] * self.map_resolution / 100.0, 0.]

                        self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
                        self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]
        else:
            for a in range(self.num_agents):
                self.full_map[envs, a, :, self.lmb[envs, a, 0]:self.lmb[envs, a, 1], self.lmb[envs, a, 2]:self.lmb[envs, a, 3]] = self.local_map[envs, a]
                if update:
                    self.full_pose[envs, a] = self.local_pose[envs, a] + self.origins[envs, a]
                    locs = self.full_pose[envs, a]
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                    int(c * 100.0 / self.map_resolution)]

                    self.lmb[envs, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                        (self.local_w, self.local_h),
                                                        (self.full_w, self.full_h))

                    self.planner_pose_inputs[envs, a, 3:] = self.lmb[envs, a].copy()
                    self.origins[envs, a] = [self.lmb[envs, a][2] * self.map_resolution / 100.0,
                                    self.lmb[envs, a][0] * self.map_resolution / 100.0, 0.]

                    self.local_map[envs, a] = self.full_map[envs, a, :, self.lmb[envs, a, 0]:self.lmb[envs, a, 1], self.lmb[envs, a, 2]:self.lmb[envs, a, 3]]
                    self.local_pose[envs, a] = self.full_pose[envs, a] - self.origins[envs, a]
            
    def insert_global_policy(self, data):
        dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        #encourage_place_unexplored_map
        for e in range(self.n_rollout_threads):
            if self.use_intrinsic_reward and self.env_info['sum_merge_explored_ratio'][e] > 0.9:
                for agent_id in range(self.num_agents):
                    intrinsic_gt = self.intrinsic_gt[e , agent_id].copy()
                    intrinsic_gt[intrinsic_gt<0.2] = -1
                    intrinsic_gt[intrinsic_gt>=0.2] = 1
                    if self.use_center:
                        reward_map = intrinsic_gt - self.transform_map[e, agent_id, 1]
                    else:
                        reward_map = intrinsic_gt - self.merge_map[e, agent_id, 1]
                    if reward_map[int(self.global_goal[e, agent_id, 0]*self.local_w+self.lmb[e, agent_id, 0]), int(self.global_goal[e, agent_id, 1]*self.local_h+self.lmb[e, agent_id, 2])] > 0.5:
                        self.rewards[e, agent_id] += 0.02
        
        if self.use_delta_reward:
            self.env_info['sum_mer_explored_reward'] += self.rewards[:,:,0]
        else:
            self.env_info['sum_mer_explored_reward'] = self.rewards[:,:,0]
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        if not self.use_centralized_V:
            self.share_global_input = self.global_input.copy()

        self.buffer.insert(self.share_global_input, self.global_input, rnn_states, rnn_states_critic, actions, action_log_probs, values, self.rewards, self.global_masks)
        
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        if self.use_delta_reward:
            self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        
    def ft_compute_global_goal(self, e):
        #print("    apf {}".format(e))
        # ft_merge_map transform
        locations = self.update_ft_merge_map(e)
        
        inputs = {
            'map_pred' : self.ft_merge_map[e,0],
            'exp_pred' : self.ft_merge_map[e,1],
            'locations' : locations
        }
        # if enough distance or steps, replan
        if self.all_args.ft_global_mode == 'apf':
           goal_mask = [(self.env_step > 0 and bfs_distance(self.ft_map, self.ft_lx, self.ft_ly, locations[agent_id], self.ft_pre_goals[e][agent_id]) > 50) and self.ft_go_steps[e][agent_id]<10 for agent_id in range(self.num_agents)] #  dist>40 and steps<30? true for not update
        elif self.all_args.ft_global_mode == 'nearest':
            # goal_mask = [(self.env_step > 0 and bfs_distance(self.ft_map, self.ft_lx, self.ft_ly, locations[agent_id], self.ft_pre_goals[e][agent_id]) > 70 and self.ft_go_steps[e][agent_id]<10) for agent_id in range(self.num_agents)]
            goal_mask = [(self.env_step > 0 and l2distance(locations[agent_id], self.ft_pre_goals[e][agent_id]) > 30 and self.ft_go_steps[e][agent_id]<10) for agent_id in range(self.num_agents)]
        elif self.all_args.ft_global_mode == 'utility':
            goal_mask = [(self.env_step > 0 and bfs_distance(self.ft_map, self.ft_lx, self.ft_ly, locations[agent_id], self.ft_pre_goals[e][agent_id]) > 50 and self.ft_go_steps[e][agent_id]<20) for agent_id in range(self.num_agents)]
        elif self.all_args.ft_global_mode == 'rrt':
            goal_mask = [(self.env_step > 0 and l2distance(locations[agent_id], self.ft_pre_goals[e][agent_id]) > 30 and self.ft_go_steps[e][agent_id]<10) for agent_id in range(self.num_agents)]
        else:
            raise NotImplementedError
        
        goals = self.ft_get_goal(inputs, goal_mask, pre_goals = self.ft_pre_goals[e])

        for agent_id in range(self.num_agents):
            if not goal_mask[agent_id] or self.all_args.ft_global_mode == 'utility':
                self.ft_pre_goals[e][agent_id] = np.array(goals[agent_id], dtype=np.int32) # goals before rotation

        self.ft_goals[e]=self.rot_ft_goals(e, goals, goal_mask)
    
    def rot_ft_goals(self, e, goals, goal_mask = None):
        if goal_mask == None:
            goal_mask = [True for _ in range(self.num_agents)]
        ft_goals = np.zeros((self.num_agents, 2), dtype = np.int32)
        for agent_id in range(self.num_agents):
            if goal_mask[agent_id]:
                ft_goals[agent_id] = self.ft_goals[e][agent_id]
                continue
            self.ft_go_steps[e][agent_id] = 0
            output = np.zeros((1, 1, self.full_w, self.full_h), dtype =np.float32)
            output[0, 0, goals[agent_id][0]-1 : goals[agent_id][0]+2, goals[agent_id][1]-1:goals[agent_id][1]+2] = 1
            agent_n_trans = F.grid_sample(torch.from_numpy(output).float(), self.agent_trans[e][agent_id].float(), align_corners = True)
            map = F.grid_sample(agent_n_trans.float(), self.agent_rotation[e][agent_id].float(), align_corners = True)[0, 0, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(map[ :, :], axis=None), map[ :, :].shape) # might be wrong!!
            ft_goals[agent_id] = np.array([index_a, index_b], dtype = np.float32)
        return ft_goals
    
    def ft_compute_local_input(self):
        assert self.all_args.use_center == False
        self.local_input = []
        for e in range(self.n_rollout_threads):
            p_input = defaultdict(list)
            for a in range(self.num_agents):
                p_input['goal'].append([int(self.ft_goals[e, a][0]), int(self.ft_goals[e,a][1])])
                p_input['map_pred'].append(self.merge_map[e, a, 0, :, :].copy())
                p_input['exp_pred'].append(self.merge_map[e, a, 1, :, :].copy())
                pose_pred = self.planner_pose_inputs[e, a].copy()
                pose_pred[3:] = np.array((0, self.full_w, 0, self.full_h))
                p_input['pose_pred'].append(pose_pred)
            self.local_input.append(p_input)
            
    def update_ft_merge_map(self, e):
        full_map = self.full_map[e]
        self.ft_merge_map[e] = np.zeros((2, self.full_w, self.full_h), dtype = np.float32) # only explored and obstacle
        locations = []
        for agent_id in range(self.num_agents):
            output = torch.from_numpy(full_map[agent_id].copy())
            n_rotated = F.grid_sample(output.unsqueeze(0).float(), self.rotation[e][agent_id].float(), align_corners=True)
            n_map = F.grid_sample(n_rotated.float(), self.trans[e][agent_id].float(), align_corners = True)
            agent_merge_map = n_map[0, :, :, :].numpy()

            (index_a, index_b) = np.unravel_index(np.argmax(agent_merge_map[2, :, :], axis=None), agent_merge_map[2, :, :].shape) # might be inaccurate !!        
            self.ft_merge_map[e] += agent_merge_map[:2]
            locations.append((index_a, index_b))

        for i in range(2):
            self.ft_merge_map[e,i][self.ft_merge_map[e,i] > 1] = 1
            self.ft_merge_map[e,i][self.ft_merge_map[e,i] < 0.2] = 0
        
        return locations
    
    def ft_get_goal(self, inputs, goal_mask, pre_goals = None):
        obstacle = inputs['map_pred']
        explored = inputs['exp_pred']
        locations = inputs['locations']

        obstacle = np.rint(obstacle).astype(np.int32)
        explored = np.rint(explored).astype(np.int32)
        explored[obstacle == 1] = 1

        H, W = explored.shape
        steps = [(-1,0),(1,0),(0,-1),(0,1)]

        map, (lx, ly), unexplored = get_frontier(obstacle, explored, locations)
        '''
        map: H x W
            - 0 for explored & available cell
            - 1 for obstacle
            - 2 for target (frontier)
        '''
        self.ft_map = map.copy()
        self.ft_lx = lx
        self.ft_ly = ly
        
        goals = []
        locations = [(x-lx, y-ly) for x, y in locations]
        if self.all_args.ft_global_mode == 'utility':
            pre_goals = pre_goals.copy()
            pre_goals[:, 0] -= lx
            pre_goals[:, 1] -= ly
            goals = max_utility_frontier(map, unexplored, locations, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, utility_radius = self.all_args.utility_radius, pre_goals = pre_goals, goal_mask = goal_mask)
            goals[:, 0] += lx
            goals[:, 1] += ly
        else:
            for agent_id in range(self.num_agents): # replan when goal is not target
                if goal_mask[agent_id]:
                    goals.append((-1,-1))
                    continue
                if self.all_args.ft_global_mode == 'apf':
                    apf = APF(self.all_args)
                    path = apf.schedule(map, locations, steps, agent_id, clear_disk = True)
                    goal = path[-1]
                elif self.all_args.ft_global_mode == 'nearest':
                    goal = nearest_frontier(map, locations, steps, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius)
                elif self.all_args.ft_global_mode == 'rrt':
                    goal = rrt_global_plan(map, unexplored, locations, agent_id, clear_radius = self.all_args.ft_clear_radius, cluster_radius = self.all_args.ft_cluster_radius, step = self.env_step, utility_radius = self.all_args.utility_radius)
                else:
                    raise NotImplementedError
                goals.append((goal[0] + lx, goal[1] + ly))

        return goals
    
    def train_slam_module(self):
        for _ in range(self.slam_iterations):
            inputs, outputs = self.slam_memory.sample(self.slam_batch_size)
            b_obs_last, b_obs, b_poses = inputs
            gt_fp_projs, gt_fp_explored, gt_pose_err = outputs
            
            b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                self.nslam_module(b_obs_last, b_obs, b_poses,
                            None, None, None,
                            build_maps=False)
            
            gt_fp_projs = check(gt_fp_projs).to(self.device)
            gt_fp_explored = check(gt_fp_explored).to(self.device)
            gt_pose_err = check(gt_pose_err).to(self.device)
            
            loss = 0
            if self.proj_loss_coeff > 0:
                proj_loss = F.binary_cross_entropy(b_proj_pred.double(), gt_fp_projs.double())
                self.train_slam_infos['costs'].append(proj_loss.item())
                loss += self.proj_loss_coeff * proj_loss

            if self.exp_loss_coeff > 0:
                exp_loss = F.binary_cross_entropy(b_fp_exp_pred.double(), gt_fp_explored.double())
                self.train_slam_infos['exp_costs'].append(exp_loss.item())
                loss += self.exp_loss_coeff * exp_loss

            if self.pose_loss_coeff > 0:
                pose_loss = torch.nn.MSELoss()(b_pose_err_pred.double(), gt_pose_err.double())
                self.train_slam_infos['pose_costs'].append(self.pose_loss_coeff * pose_loss.item())
                loss += self.pose_loss_coeff * pose_loss
            
            self.slam_optimizer.zero_grad()
            loss.backward()
            self.slam_optimizer.step()

            del b_obs_last, b_obs, b_poses
            del gt_fp_projs, gt_fp_explored, gt_pose_err
            del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

    def train_local_policy(self):
        self.local_optimizer.zero_grad()
        self.local_policy_loss.backward()
        self.train_local_infos['local_policy_loss'].append(self.local_policy_loss.item())
        self.local_optimizer.step()
        self.local_policy_loss = 0
        self.local_rnn_states = self.local_rnn_states.detach_()

    def train_global_policy(self):
        self.compute()
        train_global_infos = self.train()
        
        for k, v in train_global_infos.items():
            self.train_global_infos[k].append(v)

    def save_slam_model(self, step):
        if self.train_slam:
            if len(self.train_slam_infos['costs']) >= 1000 and np.mean(self.train_slam_infos['costs']) < self.best_slam_cost:
                self.best_slam_cost = np.mean(self.train_slam_infos['costs'])
                torch.save(self.nslam_module.state_dict(), str(self.save_dir) + "/slam_best.pt")
            torch.save(self.nslam_module.state_dict(), str(self.save_dir) + "slam_periodic_{}.pt".format(step))

    def save_local_model(self, step):
        if self.train_local:
            if len(self.train_local_infos['local_policy_loss']) >= 100 and \
                (np.mean(self.train_local_infos['local_policy_loss']) <= self.best_local_loss):
                self.best_local_loss = np.mean(self.train_local_infos['local_policy_loss'])
                torch.save(self.local_policy.state_dict(), str(self.save_dir) + "/local_best.pt")
            torch.save(self.local_policy.state_dict(), str(self.save_dir) + "local_periodic_{}.pt".format(step))
    
    def save_global_model(self, step):
        if len(self.env_infos["sum_merge_explored_reward"]) >= self.all_args.eval_episodes and \
            (np.mean(self.env_infos["sum_merge_explored_reward"]) >= self.best_gobal_reward):
            self.best_gobal_reward = np.mean(self.env_infos["sum_merge_explored_reward"])
            torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_best.pt")
            torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_best.pt")
        torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_periodic_{}.pt".format(step))
    
    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    if k == 'merge_auc':
                        for i in range(self.max_episode_length):
                            wandb.log({k: np.mean(v[:,:,i])}, step=i+1)
                    else:
                        wandb.log({k: np.nanmean(v) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.95" else np.mean(v)}, step=total_num_steps)
                else:
                    if k == 'merge_auc':
                        for i in range(self.max_episode_length):
                            self.writter.add_scalars(k, {k: np.mean(v[:,:,i])}, i+1)
                    else:
                        self.writter.add_scalars(k, {k: np.nanmean(v) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.95" else np.mean(v)}, total_num_steps)

    def log_auc(self, auc_infos):
        for k, v in auc_infos.items():
            if len(v) > 0:
                for i in range(self.max_episode_length):
                    self.writter.add_scalars(k, {k: np.mean(v[:,:,i])}, i+1)

    def log_agent(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if "merge" not in k:
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_".format(agent_id) + k
                    if self.use_wandb:
                        wandb.log({agent_k: np.mean(np.array(v)[:,:,agent_id])}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(np.array(v)[:,:,agent_id])}, total_num_steps)
                        
    def log_agent_auc(self, auc_infos):
        for k, v in auc_infos.items():
            if "merge" not in k:
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_".format(agent_id) + k
                    for i in range(self.max_episode_length):
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(np.array(v)[ :, :, agent_id])}, i+1)

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir))
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir))
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render and not self.all_args.use_eval:
                policy_critic_state_dict = torch.load(str(self.model_dir))
                self.policy.critic.load_state_dict(policy_critic_state_dict)
    
    def render_gifs(self):
        gif_dir = str(self.run_dir / 'gifs')

        folders = []
        get_folders(gif_dir, folders)
        filer_folders = [folder for folder in folders if "all" in folder or "merge" in folder]

        for folder in filer_folders:
            image_names = sorted(os.listdir(folder))

            frames = []
            for image_name in image_names:
                if image_name.split('.')[-1] == "gif":
                    continue
                image_path = os.path.join(folder, image_name)
                frame = imageio.imread(image_path)
                frames.append(frame)

            imageio.mimsave(str(folder) + '/render.gif', frames, duration=self.all_args.ifi)
    
    def visualize_obs(self, fig, ax, obs):
        # individual
        for agent_id in range(self.num_agents * 2):
            sub_ax = ax[agent_id]
            for i in range(8):
                sub_ax[i].clear()
                sub_ax[i].set_yticks([])
                sub_ax[i].set_xticks([])
                sub_ax[i].set_yticklabels([])
                sub_ax[i].set_xticklabels([])
                if agent_id < self.num_agents and i<4:
                    sub_ax[i].imshow(self.full_map[0, agent_id, i])
                elif agent_id >= self.num_agents:
                    sub_ax[i].imshow(obs['global_merge_obs'][0, agent_id-self.num_agents, i])
                #elif i < 5:
                    #sub_ax[i].imshow(obs['global_merge_goal'][0, agent_id-self.num_agents, i-4])
                    #sub_ax[i].imshow(obs['gt_map'][0, agent_id - self.num_agents, i-4])
        plt.gcf().canvas.flush_events()
        # plt.pause(0.1)
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()
    
    @torch.no_grad()
    def eval(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            # store each episode ratio or reward
            self.init_env_info()
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = [infos[e]['explorable_map'] for e in range(self.n_rollout_threads)]
            self.merge_explored_gt = [infos[e]['merge_explored_gt'] for e in range(self.n_rollout_threads)]
            self.merge_obstacle_gt = [infos[e]['merge_obstacle_gt'] for e in range(self.n_rollout_threads)]

            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)

            # Compute Global policy input
            self.first_compute_global_input()

            # Compute Global goal
            rnn_states = self.eval_compute_global_goal(rnn_states)

            # compute local input
            if self.use_merge_local:
                self.compute_local_input(self.local_merge_map)
            else:
                self.compute_local_input(self.local_map)

            # Output stores local goals as well as the the ground-truth action
            self.local_output = self.envs.get_short_term_goal(self.local_input)
            self.local_output = np.array(self.local_output, dtype = np.long)
            
            for step in range(self.max_episode_length):
                ic(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = self.obs.copy()

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    for key in ['explored_ratio', 'explored_reward', 'repeat_area', 'path_length', 'merge_explored_ratio', 'merge_explored_reward', 'merge_repeat_area']:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio' and self.use_eval:
                                self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio' and self.use_eval:
                                self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e][key])
                    if 'overlap_ratio' in infos[e].keys():
                        self.env_info['overlap_ratio'][e] = infos[e]['overlap_ratio']
                    if 'merge_explored_ratio_step' in infos[e].keys():
                        self.env_info['merge_explored_ratio_step'][e] = infos[e]['merge_explored_ratio_step']
                    if 'merge_explored_ratio_step_0.95' in infos[e].keys():
                        self.env_info['merge_explored_ratio_step_0.95'][e] = infos[e]['merge_explored_ratio_step_0.95']
                    for agent_id in range(self.num_agents):
                        agent_k = "agent{}_explored_ratio_step".format(agent_id)
                        if agent_k in infos[e].keys():
                            self.env_info['explored_ratio_step'][e][agent_id] = infos[e][agent_k]
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose(update = False)
                for agent_id in range(self.num_agents):
                    _, self.local_merge_map[:, agent_id] = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, agent_id)

                # Global Policy
                if local_step == self.num_local_steps - 1:
                    # For every global step, update the full and local maps
                    self.update_map_and_pose()
                    self.compute_global_input()
                    self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
                    # Compute Global goal
                    rnn_states = self.eval_compute_global_goal(rnn_states)
                    
                # Local Policy
                if self.use_merge_local:
                    self.compute_local_input(self.local_merge_map)
                else:
                    self.compute_local_input(self.local_map)

                # Output stores local goals as well as the the ground-truth action
                self.local_output = self.envs.get_short_term_goal(self.local_input)
                self.local_output = np.array(self.local_output, dtype = np.long)
            
            self.convert_info()
            
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            if not self.use_render:
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
                
        if self.use_eval and not self.use_wandb:
                self.log_auc(self.auc_infos)
                self.log_agent_auc(self.auc_infos)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

        if self.all_args.save_gifs:
            print("generating gifs....")
            self.render_gifs()
            print("done!")
            
    @torch.no_grad()
    def eval_apf(self):
        self.eval_infos = defaultdict(list)

        for episode in range(self.all_args.eval_episodes):
            # store each episode ratio or reward
            self.init_env_info()
            if not self.use_delta_reward:
                self.rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            # init map and pose 
            self.init_map_and_pose() 
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            # reset env
            self.obs, infos = self.envs.reset(reset_choose)
            self.trans = [infos[e]['trans'] for e in range(self.n_rollout_threads)]
            self.rotation = [infos[e]['rotation'] for e in range(self.n_rollout_threads)]
            self.scene_id = [infos[e]['scene_id'] for e in range(self.n_rollout_threads)]
            self.agent_trans = [infos[e]['agent_trans'] for e in range(self.n_rollout_threads)]
            self.agent_rotation = [infos[e]['agent_rotation'] for e in range(self.n_rollout_threads)]
            self.explorable_map = [infos[e]['explorable_map'] for e in range(self.n_rollout_threads)]
            self.merge_explored_gt = [infos[e]['merge_explored_gt'] for e in range(self.n_rollout_threads)]
            self.merge_obstacle_gt = [infos[e]['merge_obstacle_gt'] for e in range(self.n_rollout_threads)]
            self.merge_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            # Predict map from frame 1:
            self.run_slam_module(self.obs, self.obs, infos)
            self.env_step = 0

            # Compute Global policy input
            self.update_local_map()
            self.update_map_and_pose(update=False)
            
            for a in range(self.num_agents):
                self.merge_map[:, a], _ = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)

            # Compute Global goal
            for e in range(self.n_rollout_threads):
                self.ft_compute_global_goal(e)

            # compute local input
            self.ft_compute_local_input()

            # Output stores local goals as well as the the ground-truth action
            self.local_output = self.envs.get_short_term_goal(self.local_input)
            self.local_output = np.array(self.local_output, dtype = np.long)
            
            for step in range(self.max_episode_length):
                self.env_step = step
                ic(step)
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.episode_length
                eval_global_step = step // self.num_local_steps + 1

                self.last_obs = self.obs.copy()

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, reward, dones, infos = self.envs.step(actions_env)

                for e in range(self.n_rollout_threads):
                    for key in ['explored_ratio', 'explored_reward', 'repeat_area', 'path_length', 'merge_explored_ratio', 'merge_explored_reward', 'merge_repeat_area']:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            if key == 'merge_explored_ratio' and self.use_eval:
                                self.auc_infos['merge_auc'][episode, e, step] = self.auc_infos['merge_auc'][episode, e, step-1] + np.array(infos[e][key])
                            if key == 'explored_ratio' and self.use_eval:
                                self.auc_infos['agent_auc'][episode, e, :, step] = self.auc_infos['agent_auc'][episode, e, :, step-1] + np.array(infos[e][key])
                    if 'overlap_ratio' in infos[e].keys():
                        self.env_info['overlap_ratio'][e] = infos[e]['overlap_ratio']
                    if 'merge_explored_ratio_step' in infos[e].keys():
                        self.env_info['merge_explored_ratio_step'][e] = infos[e]['merge_explored_ratio_step']
                    if 'merge_explored_ratio_step_0.95' in infos[e].keys():
                        self.env_info['merge_explored_ratio_step_0.95'][e] = infos[e]['merge_explored_ratio_step_0.95']
                    for agent_id in range(self.num_agents):
                        agent_k = "agent{}_explored_ratio_step".format(agent_id)
                        if agent_k in infos[e].keys():
                            self.env_info['explored_ratio_step'][e][agent_id] = infos[e][agent_k]
                                
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.global_masks *= self.local_masks

                self.run_slam_module(self.last_obs, self.obs, infos)
                self.update_local_map()
                self.update_map_and_pose(update=False)
                for a in range(self.num_agents):
                    self.merge_map[:, a], _ = self.transform(self.full_map, self.trans, self.rotation, self.agent_trans, self.agent_rotation, a)
                # Global Policy
                self.ft_go_steps += 1
                for e in range (self.n_rollout_threads):
                    if self.env_info['sum_merge_explored_ratio'][e] - self.ft_last_merge_explored_ratio[e] > 0.01 or step % self.all_args.ft_num_local_steps == 0:
                        self.update_map_and_pose(envs = e)  
                        self.ft_last_merge_explored_ratio[e] = self.env_info['sum_merge_explored_ratio'][e]
                        self.ft_compute_global_goal(e) 
                        
                # Local Policy
                self.ft_compute_local_input()

                # Output stores local goals as well as the the ground-truth action
                self.local_output = self.envs.get_short_term_goal(self.local_input)
                self.local_output = np.array(self.local_output, dtype = np.long)
            
            self.convert_info()
            
            total_num_steps = (episode + 1) * self.max_episode_length * self.n_rollout_threads
            if not self.use_render:
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
                
        if self.use_eval and not self.use_wandb:
                self.log_auc(self.auc_infos)
                self.log_agent_auc(self.auc_infos)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.95"else np.mean(v)))

        if self.all_args.save_gifs:
            print("generating gifs....")
            self.render_gifs()
            print("done!")