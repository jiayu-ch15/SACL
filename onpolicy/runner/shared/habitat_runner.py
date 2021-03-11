    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner

from onpolicy.envs.habitat.model.model import Neural_SLAM_Module, Local_IL_Policy
from onpolicy.envs.habitat.utils.memory import FIFOMemory

def _t2n(x):
    return x.detach().cpu().numpy()

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
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


class HabitatRunner(Runner):
    def __init__(self, config):
        super(HabitatRunner, self).__init__(config)

        self.map_resolution = self.all_args.map_resolution
        self.global_downscaling = self.all_args.global_downscaling
        # init variables
        self.init_map_variables()
        # map and pose
        self.init_map_and_pose()  
        # local policy
        self.init_local_policy()  
        # slam module
        self.init_slam()    

    def init_map_variables(self):
        ### Full map consists of 4 channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations

        # Calculating full and local map sizes
        map_size = self.all_args.map_size_cm // self.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w, self.local_h = int(self.full_w / self.global_downscaling), \
                        int(self.full_h / self.global_downscaling)

        # Initializing full and local map
        self.full_map = np.zeros(self.n_rollout_threads, 4, self.full_w, self.full_h)
        self.local_map = np.zeros(self.n_rollout_threads, 4, self.local_w, self.local_h)

        # Initial full and local pose
        self.full_pose = np.zeros(self.n_rollout_threads, 3)
        self.local_pose = np.zeros(self.n_rollout_threads, 3)

        # Origin of local map
        self.origins = np.zeros((self.n_rollout_threads, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.n_rollout_threads, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.n_rollout_threads, 7))

    def init_map_and_pose(self):
        self.full_map.fill_(0.) # TODO remove this code
        self.full_pose.fill_(0.) # TODO remove this code
        self.full_pose[:, :2] = self.all_args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.n_rollout_threads):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                            int(c * 100.0 / self.map_resolution)]

            self.full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                (self.local_w, self.local_h),
                                                (self.full_w, self.full_h))

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [self.lmb[e][2] * self.map_resolution / 100.0,
                            self.lmb[e][0] * self.map_resolution / 100.0, 0.]

        for e in range(self.n_rollout_threads):
            self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
            self.local_pose[e] = self.full_pose[e] - self.origins[e]

    def init_local_policy(self):
        # Local policy
        l_observation_space = gym.spaces.Box(0, 255, (3,
                                                    self.all_args.frame_width,
                                                    self.all_args.frame_width), dtype='uint8')
        l_action_space = gym.spaces.Discrete(3)
        self.local_policy = Local_IL_Policy(l_observation_space.shape, l_action_space.n,
                               recurrent = self.all_args.use_recurrent_policy_local,
                               hidden_size = self.all_args.hidden_size_local,
                               deterministic = self.all_args.use_deterministic_local)
        
        if self.all_args.load_local != "0":
            print("Loading local {}".format(self.all_args.load_local))
            state_dict = torch.load(self.all_args.load_local, map_location='cpu')
            self.local_policy.load_state_dict(state_dict)

        if not self.all_args.train_local:
            self.local_policy.eval()
        else:
            self.local_optimizer = torch.optim.Adam(self.local_policy.parameters(), lr=self.all_args.lr, eps=self.all_args.opti_eps, weight_decay=self.all_args.weight_decay)

    def init_slam(self):
        self.nslam_module = Neural_SLAM_Module(self.all_args)
        
        if self.all_args.load_slam != "0":
            print("Loading slam {}".format(self.all_args.load_slam))
            state_dict = torch.load(self.all_args.load_slam, map_location='cpu')
            self.nslam_module.load_state_dict(state_dict)
        
        if not self.all_args.train_slam:
            self.nslam_module.eval()
        else:
            self.slam_memory = FIFOMemory(self.all_args.slam_memory_size)
            self.slam_optimizer = torch.optim.Adam(self.nslam_module.parameters(), lr=self.all_args.lr, eps=self.all_args.opti_eps, weight_decay=self.all_args.weight_decay)

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

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, infos = self.envs.reset()

        # Predict map from frame 1:
        self.predict_map(obs, infos)

        # Compute Global policy input
        global_input, global_orientation = self.compute_global_policy_input()

        # replay buffer
        self.buffer.share_obs[0] = global_input.copy()
        self.buffer.obs[0] = global_input.copy()
        self.buffer.extras[0] = global_orientation.copy()

    def predict_map(self, obs, infos):
        poses = np.array([infos[env_idx]['sensor_pose'] for env_idx in range(self.n_rollout_threads)])

        _, _, self.local_map[:, 0, :, :], self.local_map[:, 1, :, :], _, self.local_pose = \
            self.nslam_module(obs, obs, poses, 
                            self.local_map[:, 0, :, :],
                            self.local_map[:, 1, :, :], 
                            self.local_pose)

    def compute_global_policy_input(self):
        locs = self.local_pose
        global_input = np.zeros(self.n_rollout_threads, 8, self.local_w, self.local_h)
        global_orientation = np.zeros(self.n_rollout_threads, 1).long()

        for e in range(self.n_rollout_threads):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.map_resolution),
                            int(c * 100.0 / self.map_resolution)]

            self.local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
            global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

        global_input[:, 0:4, :, :] = self.local_map
        global_input[:, 4:, :, :] = (nn.MaxPool2d(self.global_downscaling)(torch.from_numpy(self.full_map))).numpy()

        return global_input, global_orientation

    def 
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
                            
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
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
                        time.sleep(ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + 'render.gif', all_frames, duration=self.all_args.ifi)
