import math
import os
import pickle
import sys
import gym
import numpy as np
import quaternion

import torch
from torch.nn import functional as F
from torchvision import transforms

import skimage.morphology
from PIL import Image
import matplotlib
if matplotlib.get_backend() == "agg":
    print("matplot backend is {}".format(matplotlib.get_backend()))
    # matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .utils.map_builder import MapBuilder
from .utils.fmm_planner import FMMPlanner
from .utils.noisy_actions import CustomActionSpaceConfiguration
from .utils.supervision import HabitatMaps
from .utils.grid import get_grid, get_grid_full
from .utils import pose as pu
from .utils import visualizations as vu

import habitat
from habitat import logger
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat_baselines.config.default import get_config as cfg_baseline

import onpolicy

def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth * 1000.
    return depth


class Exploration_Env(habitat.RLEnv):

    def __init__(self, args, config_env, config_baseline, dataset, run_dir):

        self.args = args
        self.run_dir = run_dir

        self.num_agents = args.num_agents
        self.use_reward_penalty = args.use_reward_penalty
        self.reward_decay = args.reward_decay
        self.use_render = args.use_render
        self.render_merge = args.render_merge
        self.save_gifs = args.save_gifs
        self.map_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm

        self.num_actions = 3
        self.dt = 10
        self.reward_gamma = 1

        self.sensor_noise_fwd = \
            pickle.load(open(onpolicy.__path__[0] + "/envs/habitat/model/noise_models/sensor_noise_fwd.pkl", 'rb'))
        self.sensor_noise_right = \
            pickle.load(open(onpolicy.__path__[0] + "/envs/habitat/model/noise_models/sensor_noise_right.pkl", 'rb'))
        self.sensor_noise_left = \
            pickle.load(open(onpolicy.__path__[0] + "/envs/habitat/model/noise_models/sensor_noise_left.pkl", 'rb'))

        habitat.SimulatorActions.extend_action_space("NOISY_FORWARD")
        habitat.SimulatorActions.extend_action_space("NOISY_RIGHT")
        habitat.SimulatorActions.extend_action_space("NOISY_LEFT")

        config_env.defrost()
        config_env.SIMULATOR.ACTION_SPACE_CONFIG = "CustomActionSpaceConfiguration"
        config_env.freeze()

        super().__init__(config_env, dataset)

        self.scene_name = self.habitat_env.sim.config.SCENE
        if "replica" in self.scene_name:
            self.scene_id = self.scene_name.split("/")[-3]
        else:
            self.scene_id = self.scene_name.split("/")[-1].split(".")[0]

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')
        self.share_observation_space = gym.spaces.Box(0, 255,
                                                      (3, args.frame_height,
                                                       args.frame_width),
                                                      dtype='uint8')

        self.mapper = []
        for _ in range(self.num_agents):
            self.mapper.append(self.build_mapper())
        self.curr_loc = []
        self.last_loc = []
        self.curr_loc_gt = []
        self.last_loc_gt = []
        self.last_sim_location = []
        self.map = []
        self.explored_map = []

        self.episode_no = 0

        self.res = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((args.frame_height, args.frame_width),
                                                         interpolation=Image.NEAREST)])
        
        self.maps_dict = []
        for _ in range(self.num_agents):
            self.maps_dict.append({})

        if self.use_render:
            plt.ion()
            self.figure, self.ax = plt.subplots(self.num_agents, 3, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Scene {} Map".format(self.scene_id))
            if args.render_merge:
                self.figure_m, self.ax_m = plt.subplots(1, 2, figsize=(6*16/9, 6),
                                                    facecolor="whitesmoke",
                                                    num="Scene {} Merge Map".format(self.scene_id))

    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def save_trajectory_data(self):
        traj_dir = '{}/trajectory/{}/'.format(self.run_dir, self.scene_id)
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)

        for agent_id in range(self.num_agents):
            filepath = traj_dir + 'episode' + str(self.episode_no) +'_agent' + str(agent_id) + ".txt"
            with open(filepath, "w+") as f:
                f.write(self.scene_name + "\n")
                for state in self.trajectory_states[i]:
                    f.write(str(state)+"\n")
                    f.flush()

    def save_position(self):
        self.agent_state = []
        for agent_id in range(self.num_agents):
            self.agent_state.append(self._env.sim.get_agent_state())
            self.trajectory_states[agent_id].append([self.agent_state[agent_id].position,
                                              self.agent_state[agent_id].rotation])

    def reset(self):
        self.reward_gamma = 1
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = [[] for _ in range(self.num_agents)]
        self.explored_ratio_step = np.ones(self.num_agents) * (-1.0)
        self.merge_explored_ratio_step = -1.0
        self.explored_ratio_threshold = 0.9
        self.merge_ratio = 0
        self.ratio = np.zeros(self.num_agents)

        if self.args.randomize_env_every > 0:
            if np.mod(self.episode_no, self.args.randomize_env_every) == 0:
                self.randomize_env()

        # Get Ground Truth Map
        self.explorable_map = []

        self.n_rot = []
        self.n_trans = []
        self.init_theta = []

        self.agent_n_rot = [[] for agent_id in range(self.num_agents)]
        self.agent_n_trans = [[] for agent_id in range(self.num_agents)]
        self.agent_st = []

        obs = super().reset()
        full_map_size = self.map_size_cm//self.map_resolution
        for agent_id in range(self.num_agents):
            mapp, n_rot, n_trans, init_theta = self._get_gt_map(full_map_size, agent_id)
            self.explorable_map.append(mapp)
            self.n_rot.append(n_rot)
            self.n_trans.append(n_trans)
            self.init_theta.append(init_theta)
        for aa in range(self.num_agents):
            for a in range(self.num_agents):
                delta_st = self.agent_st[a] - self.agent_st[aa]
                delta_rot_mat, delta_trans_mat, delta_n_rot_mat, delta_n_trans_mat =\
                get_grid_full(delta_st, (1, 1, self.grid_size, self.grid_size), (1, 1, full_map_size, full_map_size), torch.device("cpu"))

                self.agent_n_rot[aa].append(delta_n_rot_mat.numpy())
                self.agent_n_trans[aa].append(delta_n_trans_mat.numpy())
        
        self.merge_pred_map = np.zeros_like(self.explored_map[0])
        self.prev_explored_area = [0. for _ in range(self.num_agents)]
        self.prev_merge_explored_area = 0

        # Preprocess observations
        rgb = [obs[agent_id]['rgb'].astype(np.uint8) for agent_id in range(self.num_agents)]
        self.obs = rgb  # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = [np.asarray(self.res(rgb[agent_id])) for agent_id in range(self.num_agents)]
        state = [rgb[agent_id].transpose(2, 0, 1) for agent_id in range(self.num_agents)]
        depth = [_preprocess_depth(obs[agent_id]['depth']) for agent_id in range(self.num_agents)]
        print('nnnnnnnnnnnnn')

        # Initialize map and pose
        self.curr_loc = []
        self.curr_loc_gt = []
        self.last_loc_gt = []
        self.last_loc = []
        self.last_sim_location = []
        for agent_id in range(self.num_agents):
            self.mapper[agent_id].reset_map(self.map_size_cm)
            self.curr_loc.append([self.map_size_cm/100.0/2.0,
                                  self.map_size_cm/100.0/2.0, 0.])
            self.curr_loc_gt.append([self.map_size_cm/100.0/2.0,
                                     self.map_size_cm/100.0/2.0, 0.])
            self.last_loc_gt.append([self.map_size_cm/100.0/2.0,
                                     self.map_size_cm/100.0/2.0, 0.])
            self.last_loc.append(self.curr_loc[agent_id])
            self.last_sim_location.append(self.get_sim_location(agent_id))

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = []
        for agent_id in range(self.num_agents):
            mapper_gt_pose.append(
                (self.curr_loc_gt[agent_id][0]*100.0,
                 self.curr_loc_gt[agent_id][1]*100.0,
                 np.deg2rad(self.curr_loc_gt[agent_id][2]))
            )

        fp_proj = []
        fp_explored = []
        self.map = []
        self.explored_map = []
        # Update ground_truth map and explored area
        for agent_id in range(self.num_agents):
            fp_proj_t, map_t, fp_explored_t, explored_map_t = \
                self.mapper[agent_id].update_map(depth[agent_id], mapper_gt_pose[agent_id])
            fp_proj.append(fp_proj_t)
            self.map.append(map_t)
            fp_explored.append(fp_explored_t)
            self.explored_map.append(explored_map_t)

        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.SCENE
        self.visited = [np.zeros(self.map[0].shape)
                        for _ in range(self.num_agents)]
        self.visited_vis = [np.zeros(self.map[0].shape)
                            for _ in range(self.num_agents)]
        self.visited_gt = [np.zeros(self.map[0].shape)
                           for _ in range(self.num_agents)]
        self.collison_map = [np.zeros(self.map[0].shape)
                             for _ in range(self.num_agents)]
        self.col_width = [1 for _ in range(self.num_agents)]

        # Set info
        self.info = {
            'time': [],
            'fp_proj': [],
            'fp_explored': [],
            'sensor_pose': [],
            'pose_err': [],
        }
        for agent_id in range(self.num_agents):
            self.info['time'].append(self.timestep)
            self.info['fp_proj'].append(fp_proj[agent_id])
            self.info['fp_explored'].append(fp_explored[agent_id])
            self.info['sensor_pose'].append([0., 0., 0.])
            self.info['pose_err'].append([0., 0., 0.])
            
        self.info['trans'] = self.n_trans
        self.info['rotation'] = self.n_rot
        self.info['theta'] = self.init_theta
        self.info['agent_trans'] = self.agent_n_trans
        self.info['agent_rotation'] = self.agent_n_rot
        self.info['explorable_map'] = self.explorable_map

        self.info['scene_id'] = self.scene_id

        self.save_position()

        return state, self.info 

    def step(self, action):

        self.timestep += 1
        noisy_action = []

        # Action remapping
        for agent_id in range(self.num_agents):
            if action[agent_id] == 2:  # Forward
                action[agent_id] = 1
                noisy_action.append(habitat.SimulatorActions.NOISY_FORWARD)
            elif action[agent_id] == 1:  # Right
                action[agent_id] = 3
                noisy_action.append(habitat.SimulatorActions.NOISY_RIGHT)
            elif action[agent_id] == 0:  # Left
                action[agent_id] = 2
                noisy_action.append(habitat.SimulatorActions.NOISY_LEFT)

        for agent_id in range(self.num_agents):
            self.last_loc[agent_id] = np.copy(self.curr_loc[agent_id])
            self.last_loc_gt[agent_id] = np.copy(self.curr_loc_gt[agent_id])

        self._previous_action = action

        obs = []
        rew = []
        done = []
        info = []
        for agent_id in range(self.num_agents):
            if self.args.noisy_actions:
                obs_t, rew_t, done_t, info_t = super().step(noisy_action[agent_id], agent_id)
            else:
                obs_t, rew_t, done_t, info_t = super().step(action[agent_id], agent_id)
            obs.append(obs_t)
            rew.append(rew_t)
            done.append(done_t)
            info.append(info_t)

        # Preprocess observations
        rgb = [obs[agent_id]['rgb'].astype(np.uint8) for agent_id in range(self.num_agents)]

        self.obs = rgb  # For visualization

        if self.args.frame_width != self.args.env_frame_width:
            rgb = [np.asarray(self.res(rgb[agent_id]))
                   for agent_id in range(self.num_agents)]

        state = [rgb[agent_id].transpose(2, 0, 1) for agent_id in range(self.num_agents)]

        depth = [_preprocess_depth(obs[agent_id]['depth']) for agent_id in range(self.num_agents)]

        # Get base sensor and ground-truth pose
        dx_gt = []
        dy_gt = []
        do_gt = []
        for agent_id in range(self.num_agents):
            dx_gt_t, dy_gt_t, do_gt_t = self.get_gt_pose_change(agent_id)
            dx_gt.append(dx_gt_t)
            dy_gt.append(dy_gt_t)
            do_gt.append(do_gt_t)

        dx_base = []
        dy_base = []
        do_base = []
        for agent_id in range(self.num_agents):
            dx_base_t, dy_base_t, do_base_t = self.get_base_pose_change(
                action[agent_id], (dx_gt[agent_id], dy_gt[agent_id], do_gt[agent_id]))
            dx_base.append(dx_base_t)
            dy_base.append(dy_base_t)
            do_base.append(do_base_t)

        for agent_id in range(self.num_agents):
            self.curr_loc[agent_id] = pu.get_new_pose(self.curr_loc[agent_id],
                                               (dx_base[agent_id], dy_base[agent_id], do_base[agent_id]))

        for agent_id in range(self.num_agents):
            self.curr_loc_gt[agent_id] = pu.get_new_pose(self.curr_loc_gt[agent_id],
                                                  (dx_gt[agent_id], dy_gt[agent_id], do_gt[agent_id]))

        if not self.args.noisy_odometry:
            self.curr_loc = self.curr_loc_gt
            dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = []
        for agent_id in range(self.num_agents):
            mapper_gt_pose.append(
                (self.curr_loc_gt[agent_id][0] * 100.0,
                 self.curr_loc_gt[agent_id][1] * 100.0,
                 np.deg2rad(self.curr_loc_gt[agent_id][2]))
            )

        fp_proj = []
        fp_explored = []
        self.map = []
        self.explored_map = []
        # Update ground_truth map and explored area
        for agent_id in range(self.num_agents):
            fp_proj_t, map_t, fp_explored_t, explored_map_t = \
                self.mapper[agent_id].update_map(depth[agent_id], mapper_gt_pose[agent_id])
            fp_proj.append(fp_proj_t)
            self.map.append(map_t)
            fp_explored.append(fp_explored_t)
            self.explored_map.append(explored_map_t)

        # Update collision map
        for agent_id in range(self.num_agents):
            if action[agent_id] == 1:
                x1, y1, t1 = self.last_loc[agent_id]
                x2, y2, t2 = self.curr_loc[agent_id]
                if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                    self.col_width[agent_id] += 2
                    self.col_width[agent_id] = min(self.col_width[agent_id], 9)
                else:
                    self.col_width[agent_id] = 1

                dist = pu.get_l2_distance(x1, x2, y1, y2)
                if dist < self.args.collision_threshold:  # Collision
                    length = 2
                    width = self.col_width[agent_id]
                    buf = 3
                    for i in range(length):
                        for j in range(width):
                            wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) +
                                            (j-width//2) * np.sin(np.deg2rad(t1)))
                            wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) -
                                            (j-width//2) * np.cos(np.deg2rad(t1)))
                            r, c = wy, wx
                            r, c = int(r*100/self.map_resolution), \
                                int(c*100/self.map_resolution)
                            [r, c] = pu.threshold_poses([r, c],
                                                        self.collison_map[agent_id].shape)
                            self.collison_map[agent_id][r, c] = 1

        # Set info
        self.info = {
            'time': [],
            'fp_proj': [],
            'fp_explored': [],
            'sensor_pose': [],
            'pose_err': [],
            'explored_reward': [],
            'explored_ratio': [],
            'merge_explored_reward': 0.0,
            'merge_explored_ratio': 0.0,
        }
        for agent_id in range(self.num_agents):
            self.info['time'].append(self.timestep)
            self.info['fp_proj'].append(fp_proj[agent_id])
            self.info['fp_explored'].append(fp_explored[agent_id])
            self.info['sensor_pose'].append([dx_base[agent_id], dy_base[agent_id], do_base[agent_id]])
            self.info['pose_err'].append([dx_gt[agent_id] - dx_base[agent_id],
                                          dy_gt[agent_id] - dy_base[agent_id],
                                          do_gt[agent_id] - do_base[agent_id]])

        if self.timestep % self.args.num_local_steps == 0:
            agent_explored_area, agent_explored_ratio, merge_explored_area, merge_explored_ratio = self.get_global_reward()
            self.info['merge_explored_reward'] = merge_explored_area
            self.info['merge_explored_ratio'] = merge_explored_ratio
            self.merge_ratio += merge_explored_ratio
            # log step
            if self.merge_ratio >= self.explored_ratio_threshold and self.merge_explored_ratio_step == -1.0:
                self.merge_explored_ratio_step = self.timestep
                self.info['merge_explored_ratio_step'] = self.timestep

            for agent_id in range(self.num_agents):
                self.info['explored_reward'].append(agent_explored_area[agent_id])
                self.info['explored_ratio'].append(agent_explored_ratio[agent_id])
                self.ratio[agent_id] += agent_explored_ratio[agent_id]
                if self.ratio[agent_id] >= self.explored_ratio_threshold and self.explored_ratio_step[agent_id] == -1.0:
                    self.explored_ratio_step[agent_id] = self.timestep
                    self.info["agent{}_explored_ratio_step".format(agent_id)] = self.timestep
        else:
            for _ in range(self.num_agents):
                self.info['explored_reward'].append(0.0)
                self.info['explored_ratio'].append(0.0)

        self.save_position()

        if self.info['time'][0] >= self.args.max_episode_length:
            done = [True for _ in range(self.num_agents)]
            if self.args.save_trajectory_data:
                self.save_trajectory_data()
        else:
            done = [False for _ in range(self.num_agents)]

        return state, rew, done, self.info

    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations, agent_id):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_global_reward(self):
        agent_explored_rewards = []
        agent_explored_ratios = []

        # calculate individual reward
        curr_merge_explored_map = np.zeros_like(self.explored_map[0]) # global
        merge_explorable_map = np.zeros_like(self.explored_map[0]) # global

        for agent_id in range(self.num_agents):
            curr_agent_explored_map = self.explored_map[agent_id] * self.explorable_map[agent_id]

            curr_merge_explored_map = np.maximum(curr_merge_explored_map, self.transform(curr_agent_explored_map, agent_id))
            merge_explorable_map = np.maximum(merge_explorable_map, self.transform(self.explorable_map[agent_id], agent_id))

            curr_agent_explored_area = curr_agent_explored_map.sum()
            agent_explored_reward = (curr_agent_explored_area - self.prev_explored_area[agent_id]) * 1.0
            self.prev_explored_area[agent_id] = curr_agent_explored_area
            # converting to m^2 * Reward Scaling 0.02 * reward time penalty
            agent_explored_rewards.append(agent_explored_reward * (25./10000) * 0.02 * self.reward_gamma) 
            
            reward_scale = self.explorable_map[agent_id].sum()
            agent_explored_ratios.append(agent_explored_reward/reward_scale)

        # calculate merge reward
        curr_merge_explored_area = curr_merge_explored_map.sum()
        merge_explored_reward_scale = merge_explorable_map.sum()
        merge_explored_reward = (curr_merge_explored_area - self.prev_merge_explored_area) * 1.0
        self.prev_merge_explored_area = curr_merge_explored_area
        merge_explored_ratio = merge_explored_reward / merge_explored_reward_scale
        merge_explored_reward = merge_explored_reward * (25./10000.) * 0.02 * self.reward_gamma

        if self.use_reward_penalty:
            self.reward_gamma *= self.reward_decay

        return agent_explored_rewards, agent_explored_ratios, merge_explored_reward, merge_explored_ratio

    def get_done(self, observations, agent_id):
        # This function is not used, Habitat-RLEnv requires this function
        return False

    def get_info(self, observations, agent_id):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info
    
    def seed(self, seed):
        self._env.seed(seed)
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] = self.args.hfov
        params['resolution'] = self.map_resolution
        params['map_size_cm'] = self.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.use_render
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.map_resolution)
        mapper = MapBuilder(params)
        return mapper

    def get_sim_location(self, agent_id):
        agent_state = super().habitat_env.sim.get_agent_state(agent_id)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2*np.pi)) < 0.1 or (axis % (2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_gt_pose_change(self, agent_id):
        curr_sim_pose = self.get_sim_location(agent_id)
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location[agent_id])
        self.last_sim_location[agent_id] = curr_sim_pose
        return dx, dy, do

    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1:  # Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 3:  # Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        elif action == 2:  # Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        else:  # Stop
            x_err, y_err, o_err = 0., 0., 0.

        x_err = x_err * self.args.noise_level
        y_err = y_err * self.args.noise_level
        o_err = o_err * self.args.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)

    def transform(self, inputs, agent_id):
        inputs = torch.from_numpy(inputs)
        n_rotated = F.grid_sample(inputs.unsqueeze(0).unsqueeze(
            0).float(), self.n_rot[agent_id].float(), align_corners=True)
        n_map = F.grid_sample(
            n_rotated.float(), self.n_trans[agent_id].float(), align_corners=True)
        n_map = n_map[0, 0, :, :].numpy()

        return n_map

    def get_short_term_goal(self, inputs):
        args = self.args

        self.extrinsic_rew = []
        self.intrinsic_rew = []
        self.relative_angle = []

        def discretize(dist):
            dist_limits = [0.25, 3, 10]
            dist_bin_size = [0.05, 0.25, 1.]
            if dist < dist_limits[0]:
                ddist = int(dist/dist_bin_size[0])
            elif dist < dist_limits[1]:
                ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
                    int(dist_limits[0]/dist_bin_size[0])
            elif dist < dist_limits[2]:
                ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
                    int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
            else:
                ddist = int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
                    int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
            return ddist

        # Get Map prediction
        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        output = [np.zeros((args.goals_size + 1))
                  for _ in range(self.num_agents)]

        for agent_id in range(self.num_agents):
            grid = np.rint(map_pred[agent_id])
            explored = np.rint(exp_pred[agent_id])

            # Get pose prediction and global policy planning window
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred'][agent_id]
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            planning_window = [gx1, gx2, gy1, gy2]

            # Get last loc
            last_start_x, last_start_y = self.last_loc[agent_id][0], self.last_loc[agent_id][1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0/self.map_resolution - gx1),
                          int(c * 100.0/self.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, grid.shape)

            # Get curr loc
            self.curr_loc[agent_id] = [start_x, start_y, start_o]
            r, c = start_y, start_x
            start = [int(r * 100.0/self.map_resolution - gx1),
                     int(c * 100.0/self.map_resolution - gy1)]
            start = pu.threshold_poses(start, grid.shape)
            # TODO: try reducing this

            self.visited[agent_id][gx1:gx2, gy1:gy2][start[0]-2:start[0]+3,
                                              start[1]-2:start[1]+3] = 1

            steps = 25 # ! wrong
            for i in range(steps):
                x = int(last_start[0] + (start[0] -
                                         last_start[0]) * (i+1) / steps)
                y = int(last_start[1] + (start[1] -
                                         last_start[1]) * (i+1) / steps)
                self.visited_vis[agent_id][gx1:gx2, gy1:gy2][x, y] = 1

            # Get last loc ground truth pose
            last_start_x, last_start_y = self.last_loc_gt[agent_id][0], self.last_loc_gt[agent_id][1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0/self.map_resolution),
                          int(c * 100.0/self.map_resolution)]
            last_start = pu.threshold_poses(
                last_start, self.visited_gt[agent_id].shape)

            # Get ground truth pose
            start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt[agent_id]

            r, c = start_y_gt, start_x_gt
            start_gt = [int(r * 100.0/self.map_resolution),
                        int(c * 100.0/self.map_resolution)]
            start_gt = pu.threshold_poses(start_gt, self.visited_gt[agent_id].shape)

            steps = 25 # ! wrong
            for i in range(steps):
                x = int(last_start[0] + (start_gt[0] -
                                         last_start[0]) * (i+1) / steps)
                y = int(last_start[1] + (start_gt[1] -
                                         last_start[1]) * (i+1) / steps)
                self.visited_gt[agent_id][x, y] = 1

            # Get goal
            goal = inputs['goal'][agent_id]
            goal = pu.threshold_poses(goal, grid.shape)

            # Get intrinsic reward for global policy
            # Negative reward for exploring explored areas i.e.
            # for choosing explored cell as long-term goal

            self.extrinsic_rew.append(-pu.get_l2_distance(10, goal[0], 10, goal[1]))
            self.intrinsic_rew.append(-exp_pred[agent_id][goal[0], goal[1]])

            # Get short-term goal
            stg = self._get_stg(grid, explored, start, np.copy(goal), planning_window, agent_id)

            # Find GT action
            if self.args.use_eval or self.args.use_render or not self.args.train_local:
                gt_action = 0
            else:
                gt_action = self._get_gt_action(1 - self.explorable_map[agent_id], start,
                                                [int(stg[0]), int(stg[1])],
                                                planning_window, start_o, agent_id)

            (stg_x, stg_y) = stg
            relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
            relative_dist = relative_dist*5./100.
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            output[agent_id][0] = int((relative_angle % 360.)/5.)
            output[agent_id][1] = discretize(relative_dist)
            output[agent_id][2] = gt_action
            self.relative_angle.append(relative_angle)

        if self.use_render:
            gif_dir = '{}/gifs/{}/episode_{}/all/'.format(self.run_dir, self.scene_id, self.episode_no)
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)

            self.render(inputs, grid, map_pred, gif_dir)

            if self.render_merge:
                gif_dir = '{}/gifs/{}/episode_{}/merge/'.format(self.run_dir, self.scene_id, self.episode_no)
                if not os.path.exists(gif_dir):
                    os.makedirs(gif_dir)
                self.render_merged_map(inputs, grid, map_pred, gif_dir)
        
        return output

    def _get_gt_map(self, full_map_size, agent_id):
        self.scene_name = self.habitat_env.sim.config.SCENE
        # logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state(agent_id).position.tolist()[1]*100.

        sim_map = self.map_obj.get_map()

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location(agent_id)
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        self.grid_size = int(scale*max(map_size))

        grid_map = np.zeros((self.grid_size, self.grid_size))

        grid_map[(self.grid_size - map_size[0])//2:
                 (self.grid_size - map_size[0])//2 + map_size[0],
                 (self.grid_size - map_size[1])//2:
                 (self.grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            self.agent_st.append(torch.tensor([[ 
                (x - range_x/2.) * 2. / (range_x * scale) \
                * map_size[1] * 1. / map_size[0],
                (y - range_y/2.) * 2. / (range_y * scale),
                180.0 + np.rad2deg(o)
            ]]))

        else:
            self.agent_st.append(torch.tensor([[
                (x - range_x/2.) * 2. / (range_x * scale),
                (y - range_y/2.) * 2. / (range_y * scale)
                * map_size[0] * 1. / map_size[1],
                180.0 + np.rad2deg(o)
            ]]))



        rot_mat, trans_mat, n_rot_mat, n_trans_mat = get_grid_full(self.agent_st[agent_id], (1, 1,
                                                                        self.grid_size, self.grid_size), (1, 1,
                                                                                                full_map_size, full_map_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat, align_corners=True)
        rotated = F.grid_sample(translated, rot_mat, align_corners=True)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > self.grid_size:
            episode_map[(full_map_size - self.grid_size)//2:
                        (full_map_size - self.grid_size)//2 + self.grid_size,
                        (full_map_size - self.grid_size)//2:
                        (full_map_size - self.grid_size)//2 + self.grid_size] = \
                rotated[0, 0]
        else:
            episode_map = rotated[0, 0,
                                  (self.grid_size - full_map_size)//2:
                                  (self.grid_size - full_map_size)//2 + full_map_size,
                                  (self.grid_size - full_map_size)//2:
                                  (self.grid_size - full_map_size)//2 + full_map_size]

        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map, n_rot_mat, n_trans_mat, 180.0 + np.rad2deg(o)

    def _get_stg(self, grid, explored, start, goal, planning_window, agent_id):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows > 0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols > 0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collison_map[agent_id]
                    [gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[agent_id]
                    [gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
                and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                        int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2, w+2))
            new_mat[1:h+1, 1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)

    def _get_gt_action(self, grid, start, goal, planning_window, start_o, agent_id):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                self.selem) != True
            traversible[self.visited[agent_id]
                        [gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                    planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action
    
    def render(self, inputs, grid, map_pred, gif_dir):
        for agent_id in range(self.num_agents):
            goal = inputs['goal'][agent_id]
            goal = pu.threshold_poses(goal, grid.shape)
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred'][agent_id]
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt[agent_id]

            # predicted map and pose
            vis_grid_local = vu.get_colored_map(np.rint(map_pred[agent_id]),
                                            self.collison_map[agent_id][gx1:gx2, gy1:gy2],
                                            self.visited_vis[agent_id][gx1:gx2, gy1:gy2],
                                            self.visited_gt[agent_id][gx1:gx2, gy1:gy2],
                                            [goal],
                                            self.explored_map[agent_id][gx1:gx2, gy1:gy2],
                                            self.explorable_map[agent_id][gx1:gx2, gy1:gy2],
                                            self.map[agent_id][gx1:gx2, gy1:gy2] *
                                            self.explored_map[agent_id][gx1:gx2, gy1:gy2])
            vis_grid_local = np.flipud(vis_grid_local)
            pos_local = (start_x - gy1 * self.map_resolution/100.0,
                            start_y - gx1 * self.map_resolution/100.0,
                            start_o)
            pos_gt_local = (start_x_gt - gy1 * self.map_resolution/100.0,
                            start_y_gt - gx1 * self.map_resolution/100.0,
                            start_o_gt)


            # ground truth map and pose
            vis_grid_gt = vu.get_colored_map(self.map[agent_id],
                                            self.collison_map[agent_id],
                                            self.visited_gt[agent_id],
                                            self.visited_gt[agent_id],
                                            [(goal[0] + gx1,
                                            goal[1] + gy1)],
                                            self.explored_map[agent_id],
                                            self.explorable_map[agent_id],
                                            self.map[agent_id]*self.explored_map[agent_id])
            vis_grid_gt = np.flipud(vis_grid_gt)
            pos = (start_x, start_y, start_o)
            pos_gt = (start_x_gt, start_y_gt, start_o_gt)

            ax = self.ax[agent_id] if self.num_agents > 1 else self.ax
            
            vu.visualize_all(agent_id, self.figure, ax, 
                            self.obs[agent_id], 
                            vis_grid_local[:, :, ::-1], 
                            vis_grid_gt[:, :, ::-1],
                            pos_local,
                            pos_gt_local,
                            pos,
                            pos_gt,
                            gif_dir, 
                            self.timestep, 
                            self.use_render, self.save_gifs)

    def render_merged_map(self, inputs, grid, map_pred, gif_dir):
        merge_map = np.zeros_like(self.explored_map[0])
        merge_collision_map = np.zeros_like(self.explored_map[0])
        merge_visited_gt = np.zeros_like(self.explored_map[0])
        merge_visited_vis = np.zeros_like(self.explored_map[0])
        merge_explored_map = np.zeros_like(self.explored_map[0])
        merge_explorable_map = np.zeros_like(self.explored_map[0])
        merge_gt_explored = np.zeros_like(self.explored_map[0])
        

        all_pos = []
        all_pos_gt = []
        all_goals = []
        for agent_id in range(self.num_agents):
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred'][agent_id]
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            goal = inputs['goal'][agent_id]
            goal = pu.threshold_poses(goal, grid.shape)
            start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt[agent_id]

            pos_map = np.zeros_like(self.explored_map[0])
            pos_gt_map = np.zeros_like(self.explored_map[0])
            goal_map = np.zeros_like(self.explored_map[0])

            pos_map[int(start_y * 100.0/5.0), int(start_x * 100.0/5.0)] = 1
            pos_gt_map[int(start_y_gt * 100.0/5.0), int(start_x_gt * 100.0/5.0)] = 1
            goal_map[int(goal[0] + gx1), int(goal[1] + gy1)] = 1

            pos_map = self.transform(pos_map, agent_id)
            pos_gt_map = self.transform(pos_gt_map, agent_id)
            goal_map = self.transform(goal_map, agent_id)

            (index_b, index_a) = np.unravel_index(np.argmax(pos_map, axis=None), pos_map.shape)
            (index_gt_b, index_gt_a) = np.unravel_index(np.argmax(pos_gt_map, axis=None), pos_gt_map.shape)
            (index_goal_a, index_goal_b) = np.unravel_index(np.argmax(goal_map, axis=None), goal_map.shape)

            pos = (index_a * 5.0/100.0, index_b * 5.0/100.0, start_o + self.init_theta[agent_id])
            pos_gt = (index_gt_a * 5.0/100.0, index_gt_b * 5.0/100.0, start_o_gt + self.init_theta[agent_id])
            goal = (index_goal_a, index_goal_b, 0)
            
            all_pos.append(pos)
            all_pos_gt.append(pos_gt)
            all_goals.append(goal)
            
            pred_map = np.zeros_like(self.explored_map[0])
            pred_map[gx1:gx2, gy1:gy2]= np.rint(map_pred[agent_id])
            self.merge_pred_map = np.maximum(self.merge_pred_map, self.transform(pred_map, agent_id))
            merge_map = np.maximum(merge_map, self.transform(self.map[agent_id], agent_id))
            merge_visited_gt = np.maximum(merge_visited_gt, self.transform(self.visited_gt[agent_id], agent_id))
            merge_visited_vis = np.maximum(merge_visited_vis, self.transform(self.visited_vis[agent_id], agent_id))
            merge_collision_map[self.transform(self.collison_map[agent_id], agent_id) == 1] = 1
            merge_explorable_map[self.transform(self.explorable_map[agent_id], agent_id) == 1] = 1
            merge_explored_map = np.maximum(merge_explored_map, self.transform(self.explored_map[agent_id], agent_id))
            merge_gt_explored = np.maximum(merge_gt_explored, self.transform(self.map[agent_id] * self.explored_map[agent_id], agent_id))

        vis_grid_gt = vu.get_colored_map(merge_map,
                                    merge_collision_map,
                                    merge_visited_gt,
                                    merge_visited_gt,
                                    all_goals,
                                    merge_explored_map,
                                    merge_explorable_map,
                                    merge_gt_explored)
        
        vis_grid_pred = vu.get_colored_map(merge_pred_map,
                                    merge_collision_map,
                                    merge_visited_vis,
                                    merge_visited_gt,
                                    all_goals,
                                    merge_explored_map,
                                    merge_explorable_map,
                                    merge_gt_explored)

        vis_grid_gt = np.flipud(vis_grid_gt)
        vis_grid_pred = np.flipud(vis_grid_pred)

        vu.visualize_map(self.figure_m, self.ax_m, vis_grid_gt[:, :, ::-1], vis_grid_pred[:, :, ::-1],
                        all_pos_gt, all_pos, gif_dir,
                        self.timestep, 
                        self.use_render,
                        self.save_gifs)