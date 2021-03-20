    
import time
import wandb
import os
import gym
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
from torch.nn import functional as F

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.envs.habitat.model.model import Neural_SLAM_Module, Local_IL_Policy
from onpolicy.envs.habitat.utils.memory import FIFOMemory
from onpolicy.algorithms.utils.util import init, check

from collections import defaultdict, deque


def _t2n(x):
    return x.detach().cpu().numpy()

class HabitatRunner(Runner):
    def __init__(self, config):
        super(HabitatRunner, self).__init__(config)

        # init parameters
        self.init_hyper_parameters()
        # init variables
        self.init_map_variables()
        # map and pose
        self.init_map_and_pose() 
        # global policy
        self.init_global_policy() 
        # local policy
        self.init_local_policy()  
        # slam module
        self.init_slam_module()    
    
    def warmup(self):
        # reset env
        self.obs, infos = self.envs.reset()

        # Predict map from frame 1:
        self.run_slam_module(self.obs, self.obs, infos)

        # Compute Global policy input
        self.first_compute_global_input()
        self.share_global_input = self.global_input if self.use_centralized_V else self.global_input #! wrong

        # replay buffer
        for key in self.global_input.keys():
            self.buffer.obs[key][0] = self.global_input[key].copy()

        for key in self.share_global_input.keys():
            self.buffer.share_obs[key][0] = self.share_global_input[key].copy()

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step=0)

        # compute local input
        self.compute_local_input(self.global_input['global_obs'])

        # Output stores local goals as well as the the ground-truth action
        self.local_output = self.envs.get_short_term_goal(self.local_input)
        self.local_output = np.array(self.local_output, dtype = np.long)
        
        self.last_obs = self.obs 
            
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
 
    def run(self):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                local_step = step % self.num_local_steps
                global_step = (step // self.num_local_steps) % self.num_global_steps

                del self.last_obs
                self.last_obs = self.obs

                # Sample actions
                actions_env = self.compute_local_action()

                # Obser reward and next obs
                self.obs, rewards, dones, infos = self.envs.step(actions_env)
                              
                self.local_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.local_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                self.gobal_masks *= self.local_masks

                # Reinitialize variables when episode ends
                if step == self.episode_length - 1:
                    self.init_map_and_pose()
                    del self.last_obs
                    self.last_obs = self.obs
                
                # Neural SLAM Module
                if self.train_slam:
                    self.insert_slam_module(infos)
                
                self.run_slam_module(self.last_obs, self.obs, infos, True)
                self.update_local_map()

                # Global Policy
                if local_step == self.num_local_steps - 1:
                    # For every global step, update the full and local maps
                    self.update_map_and_pose()
                    self.compute_global_input()
                    data = rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                    # insert data into buffer
                    self.insert_global_policy(data)
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step=global_step + 1)

                # Local Policy
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
                if global_step % self.num_global_steps == self.num_global_steps - 1 \
                        and local_step == self.num_local_steps - 1:
                    self.train_global_policy()

                # Finish Training
                torch.set_grad_enabled(False)       
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
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

                if self.env_name == "Habitat":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        exp_ratio = []
                        for info in infos:
                            if 'exp_ratio' in info.keys():
                                exp_ratio.append(info['exp_ratio'][agent_id])
                        agent_k = 'agent%i/exp_ratio' % agent_id
                        env_infos[agent_k] = exp_ratio
                
                self.train_global_infos["average_episode_rewards"].append(np.mean(self.buffer.rewards) * self.episode_length)
                print("average episode rewards is {}".format(np.mean(self.train_global_infos["average_episode_rewards"])))
                self.log_train(self.train_global_infos, total_num_steps)
                self.log_train(self.train_slam_infos, total_num_steps)
                self.log_train(self.train_local_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save_slam_model(total_num_steps)
                self.save_global_model(total_num_steps)
                self.save_local_model(total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

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
            gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_hyper_parameters(self):
        self.device = torch.device("cuda:0")
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
        self.num_global_steps = self.all_args.num_global_steps
        self.num_local_steps = self.all_args.num_local_steps
        
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

        # Initializing full and local map
        self.full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h))
        self.local_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h))

        # Initial full and local pose
        self.full_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3))
        self.local_pose = np.zeros((self.n_rollout_threads, self.num_agents, 3))

        # Origin of local map
        self.origins = np.zeros((self.n_rollout_threads, self.num_agents, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.n_rollout_threads, self.num_agents, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.n_rollout_threads, self.num_agents, 7))
               
    def init_map_and_pose(self):
        # self.full_map.fill_(0.) # TODO remove this code
        # self.full_pose.fill_(0.) # TODO remove this code
        self.full_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.full_w, self.full_h))
        self.local_map = np.zeros((self.n_rollout_threads, self.num_agents, 4, self.local_w, self.local_h))

        self.full_pose[:, :, :2] = self.map_size_cm / 100.0 / 2.0

        locs = self.full_pose
        self.planner_pose_inputs[:, :, :3] = locs
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.full_map[e, a, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                    (self.local_w, self.local_h),
                                                    (self.full_w, self.full_h))

                self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a]
                self.origins[e, a] = [self.lmb[e, a, 2] * self.map_resolution / 100.0,
                                self.lmb[e, a, 0] * self.map_resolution / 100.0, 0.]

        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
                self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]

    def init_global_policy(self):
        self.best_gobal_reward = -np.inf
        self.train_global_infos = {}
        self.train_global_infos['value_loss']= deque(maxlen=1000)
        self.train_global_infos['policy_loss']= deque(maxlen=1000)
        self.train_global_infos['dist_entropy'] = deque(maxlen=1000)
        self.train_global_infos['actor_grad_norm'] = deque(maxlen=1000)
        self.train_global_infos['critic_grad_norm'] = deque(maxlen=1000)
        self.train_global_infos['ratio'] = deque(maxlen=1000)
        self.train_global_infos['average_episode_rewards'] = deque(maxlen=1000)
        self.global_input = {}
        self.global_input['global_obs'] = np.zeros((self.n_rollout_threads, self.num_agents, 8, self.local_w, self.local_h), dtype=np.float32)
        self.global_input['global_orientation'] = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.long)
        
        self.gobal_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 

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
                               deterministic=self.all_args.use_local_deterministic).to(self.device)
        
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
        
        self.nslam_module = Neural_SLAM_Module(self.all_args).to(self.device)
        
        if self.load_slam != "0":
            print("Loading slam {}".format(self.load_slam))
            state_dict = torch.load(self.load_slam, map_location='cuda:0')
            self.nslam_module.load_state_dict(state_dict)
        
        if not self.train_slam:
            self.nslam_module.eval()
        else:
            self.slam_memory = FIFOMemory(self.slam_memory_size)
            self.slam_optimizer = torch.optim.Adam(self.nslam_module.parameters(), lr=self.slam_lr, eps=self.slam_opti_eps)
    
    def insert_slam_module(self, infos):
        # Add frames to memory
        for a in range(self.num_agents):
            for env_idx in range(self.n_rollout_threads):
                env_poses = infos[env_idx]['sensor_pose'][a]
                env_gt_fp_projs = np.expand_dims(infos[env_idx]['fp_proj'][a], 0)
                env_gt_fp_explored = np.expand_dims(infos[env_idx]['fp_explored'][a], 0)
                env_gt_pose_err = infos[env_idx]['pose_err'][a]
                self.slam_memory.push(
                    (self.last_obs[env_idx][a], self.obs[env_idx][a], env_poses),
                    (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

    def run_slam_module(self, last_obs, obs, infos, build_maps=True):
        for a in range(self.num_agents):
            poses = np.array([infos[e]['sensor_pose'][a] for e in range(self.n_rollout_threads)])

            _, _, self.local_map[:, a, 0, :, :], self.local_map[:, a, 1, :, :], _, self.local_pose[:,a,:] = \
                self.nslam_module(last_obs[:,a,:,:,:], obs[:,a,:,:,:], poses, 
                                self.local_map[:, a, 0, :, :],
                                self.local_map[:, a, 1, :, :], 
                                self.local_pose[:,a,:],
                                build_maps=build_maps)

    def first_compute_global_input(self):
        locs = self.local_pose
        for a in range(self.num_agents):
            for e in range(self.n_rollout_threads):
                
                r, c = locs[e, a, 1], locs[e, a, 0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.local_map[e, a, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
                self.global_input['global_orientation'][e, a] = int((locs[e, a, 2] + 180.0) / 5.)

            self.global_input['global_obs'][:, a, 0:4, :, :] = self.local_map[:,a,:,:,:]
            self.global_input['global_obs'][:, a, 4:, :, :] = (nn.MaxPool2d(self.global_downscaling)(check(self.full_map[:,a,:,:,:]))).numpy()

    def compute_local_action(self):
        local_action = torch.empty(self.n_rollout_threads, self.num_agents)
        for a in range(self.num_agents):
            local_goals = self.local_output[:, a, :-1]

            if self.train_local:
                torch.set_grad_enabled(True)
        
            action, action_prob, self.local_rnn_states[:,a] =\
                self.local_policy(self.obs[:,a],
                                    self.local_rnn_states[:,a],
                                    self.local_masks[:,a],
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
                p_input['map_pred'].append(map[e, a, 0, :, :])
                p_input['exp_pred'].append(map[e, a, 1, :, :])
                p_input['pose_pred'].append(self.planner_pose_inputs[e, a])
            self.local_input.append(p_input)
    
    def compute_global_input(self):
        locs = self.local_pose
        for a in range(self.num_agents):
            for e in range(self.n_rollout_threads):
                self.global_input['global_orientation'][e, a] = int((locs[e, a, 2] + 180.0) / 5.)
            self.global_input['global_obs'][:, a, 0:4, :, :] = self.local_map[:,a,:,:,:]
            self.global_input['global_obs'][:, a, 4:, :, :] = (nn.MaxPool2d(self.global_downscaling)(check(self.full_map[:,a,:,:,:]))).numpy()
        
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

                self.local_map[e, a, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

    def update_map_and_pose(self):
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]] = self.local_map[e, a]
                self.full_pose[e, a] = self.local_pose[e, a] + self.origins[e, a]

                locs = self.full_pose[e, a]
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                                int(c * 100.0 / self.map_resolution)]

                self.lmb[e, a] = self.get_local_map_boundaries((loc_r, loc_c),
                                                    (self.local_w, self.local_h),
                                                    (self.full_w, self.full_h))

                self.planner_pose_inputs[e, a, 3:] = self.lmb[e, a]
                self.origins[e, a] = [self.lmb[e, a][2] * self.map_resolution / 100.0,
                                self.lmb[e, a][0] * self.map_resolution / 100.0, 0.]

                self.local_map[e, a] = self.full_map[e, a, :, self.lmb[e, a, 0]:self.lmb[e, a, 1], self.lmb[e, a, 2]:self.lmb[e, a, 3]]
                self.local_pose[e, a] = self.full_pose[e, a] - self.origins[e, a]

    def insert_global_policy(self, data):
        rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        self.share_global_input = self.global_input # ! wrong
        self.global_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32) 
        self.buffer.insert(self.share_global_input, self.global_input, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, self.global_masks)

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
        if len(self.train_slam_infos['costs']) >= 1000 and np.mean(self.train_slam_infos['costs']) < self.best_slam_cost:
            self.best_slam_cost = np.mean(self.train_slam_infos['costs'])
            torch.save(self.nslam_module.state_dict(), str(self.save_dir) + "/slam_best.pt")
        torch.save(self.nslam_module.state_dict(), str(self.save_dir) + "slam_periodic_{}.pt".format(step))

    def save_local_model(self, step):
        if len(self.train_local_infos['local_policy_loss']) >= 100 and \
                (np.mean(self.train_local_infos['local_policy_loss']) <= self.best_local_loss):
            self.best_local_loss = np.mean(self.train_local_infos['local_policy_loss'])
            torch.save(self.local_policy.state_dict(), str(self.save_dir) + "/local_best.pt")
        torch.save(self.local_policy.state_dict(), str(self.save_dir) + "local_periodic_{}.pt".format(step))
    
    def save_global_model(self, step):
        if len(self.train_global_infos["average_episode_rewards"]) >= 100 and \
            (np.mean(self.train_global_infos["average_episode_rewards"]) >= self.best_gobal_reward):
            self.best_gobal_reward = np.mean(self.train_global_infos["average_episode_rewards"])
            torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_best.pt")
            torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_best.pt")
        torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "global_actor_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "global_critic_periodic_{}.pt".format(step))

        
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array', close=False)[0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array', close=False)[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + 'render.gif', all_frames, duration=self.all_args.ifi)
