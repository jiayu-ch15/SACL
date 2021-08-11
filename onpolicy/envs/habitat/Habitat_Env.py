import numpy as np
import gym
import onpolicy
import torch
from .exploration_env import Exploration_Env
from habitat.config.default import get_config as cfg_env
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


class MultiHabitatEnv(object):
    def __init__(self, args, rank, run_dir):

        self.num_agents = args.num_agents
        self.use_resnet = args.use_resnet
        self.use_centralized_V = args.use_centralized_V
        self.use_partial_reward = args.use_partial_reward
        self.use_merge_partial_reward = args.use_merge_partial_reward
        self.use_merge = args.use_merge
        self.use_single = args.use_single
        self.use_merge_goal = args.use_merge_goal

        config_env, config_baseline, dataset = self.get_config(args, rank)

        self.env = Exploration_Env(
            args, config_env, config_baseline, dataset, run_dir)

        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / args.global_downscaling), \
            int(full_h / args.global_downscaling)

        global_observation_space = {}
        #global_observation_space['global_obs'] = gym.spaces.Box(
            #low=0, high=1, shape=(8, local_w, local_h), dtype='uint8')
        if self.use_merge:
            if self.use_resnet:
                global_observation_space['global_merge_obs'] = gym.spaces.Box(
                    low=0, high=1, shape=(8, 224, 224), dtype='uint8')
            else:
                global_observation_space['global_merge_obs'] = gym.spaces.Box(
                    low=0, high=1, shape=(8, local_w, local_h), dtype='uint8')
        if self.use_merge_goal:
            if self.use_resnet:
                global_observation_space['global_merge_goal'] = gym.spaces.Box(
                    low=0, high=1, shape=(2, 224, 224), dtype='uint8')
            else:
                global_observation_space['global_merge_goal'] = gym.spaces.Box(
                    low=0, high=1, shape=(2, local_w, local_h), dtype='uint8')
        if self.use_single:
            if self.use_resnet:
                global_observation_space['global_obs'] = gym.spaces.Box(
                    low=0, high=1, shape=(8, 224, 224), dtype='uint8')
            else:
                global_observation_space['global_obs'] = gym.spaces.Box(
                    low=0, high=1, shape=(8, local_w, local_h), dtype='uint8')
            
        global_observation_space['global_orientation'] = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype='long')
        global_observation_space['other_global_orientation'] = gym.spaces.Box(
            low=-1, high=1, shape=(self.num_agents-1,), dtype='long')
        global_observation_space['vector'] = gym.spaces.Box(
            low=-1, high=1, shape=(self.num_agents,), dtype='float')
       
        share_global_observation_space = global_observation_space.copy()
        if self.use_centralized_V:
            if self.use_resnet:
                share_global_observation_space['gt_map'] = gym.spaces.Box(
                    low=0, high=1, shape=(1, 224, 224), dtype='uint8')
            else:
                share_global_observation_space['gt_map'] = gym.spaces.Box(
                    low=0, high=1, shape=(1, local_w, local_h), dtype='uint8')
            
        global_observation_space = gym.spaces.Dict(global_observation_space)
        share_global_observation_space = gym.spaces.Dict(share_global_observation_space)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for agent_id in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(share_global_observation_space)
            self.action_space.append(gym.spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32))

    def get_config(self, args, rank):
        config_env = cfg_env(config_paths=[onpolicy.__path__[0] + "/envs/habitat/habitat-lab/configs/" + args.task_config])
        
        config_env.defrost()
        config_env.DATASET.SPLIT = args.split
        config_env.DATASET.DATA_PATH = onpolicy.__path__[0] + "/envs/habitat/data/datasets/pointnav/gibson/v1/{}/{}.json.gz".format(args.split,args.split)
        config_env.freeze()

        scenes = PointNavDatasetV1.get_scenes_to_load(config_env.DATASET)

        if len(scenes) > 0:
            assert len(scenes) >= args.n_rollout_threads, (
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )
            scene_split_size = int(
                np.floor(len(scenes) / args.n_rollout_threads))

        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[rank *
                                                       scene_split_size: (rank + 1) * scene_split_size]
        config_env.DATASET.USE_SAME_SCENE = args.use_same_scene
        if args.use_same_scene:
            config_env.DATASET.CONTENT_SCENES = scenes[args.scene_id:args.scene_id+1]
        if args.use_selected_small_scenes:
            #scene_num=[2(block),14,16,20(large),21,28,29,30,37(block),38]##8(80+),12(80+),44(good),56(small),51(good),58(80+),69(good)
            #scene_num=[14,16,21,28,29,30,38,44,51,69]
            scene_num=[8, 58, 27, 29, 26, 71, 12, 54, 57, 5]
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]
        if args.use_selected_middle_scenes:
            scene_num=[20, 16, 48, 22, 21, 43, 36, 61, 49]#40, 
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]
        if args.use_selected_large_scenes:
            scene_num=[31, 70, 9, 47, 45]
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]
            

        if rank > (args.n_rollout_threads)/2 and args.n_rollout_threads > 6:
            gpu_id = 0
        else:
            gpu_id = 0 if torch.cuda.device_count() == 1 else 1

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        config_env.SIMULATOR.NUM_AGENTS = self.num_agents
        config_env.SIMULATOR.SEED = rank * 5000 + args.seed
        config_env.SIMULATOR.USE_SAME_ROTATION = args.use_same_rotation
        config_env.SIMULATOR.USE_RANDOM_ROTATION = args.use_random_rotation
        config_env.SIMULATOR.USE_DIFFERENT_START_POS = args.use_different_start_pos
        config_env.SIMULATOR.USE_FIXED_START_POS = args.use_fixed_start_pos
        config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/state/seed{}/".format(args.seed)

        config_env.SIMULATOR.AGENT.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = 10
        
        dataset = PointNavDatasetV1(config_env.DATASET)
        config_env.defrost()

        config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id

        print("Loading {}".format(config_env.SIMULATOR.SCENE))

        config_env.freeze()

        config_baseline = cfg_baseline()

        return config_env, config_baseline, dataset

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        obs, infos = self.env.reset()
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        if self.use_partial_reward:
            rewards = 0.5 * np.expand_dims(np.array(infos['explored_reward']), axis=1) + 0.5 * np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        elif self.use_merge_partial_reward:
            rewards = 0.5 * np.expand_dims(np.array(infos['explored_merge_reward']), axis=1) + 0.5 * np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        else:
            rewards = np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_short_term_goal(self, inputs):
        outputs = self.env.get_short_term_goal(inputs)
        return outputs
