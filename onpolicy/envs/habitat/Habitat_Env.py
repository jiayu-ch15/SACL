import numpy as np
import gym
import onpolicy
from .exploration_env import Exploration_Env
from habitat.config.default import get_config as cfg_env
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


class MultiHabitatEnv(object):
    def __init__(self, args, rank, run_dir):

        self.num_agents = args.num_agents

        config_env, config_baseline, dataset = self.get_config(args, rank)

        self.env = Exploration_Env(
            args, config_env, config_baseline, dataset, run_dir)

        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / args.global_downscaling), \
            int(full_h / args.global_downscaling)

        global_observation_space = {}
        global_observation_space['global_obs'] = gym.spaces.Box(
            low=0, high=1, shape=(12, local_w, local_h), dtype='uint8')
        global_observation_space['global_orientation'] = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype='long')
        global_observation_space = gym.spaces.Dict(global_observation_space)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for agent_id in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(global_observation_space)
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

        if rank > (args.n_rollout_threads)/2 and args.n_rollout_threads > 6:
            gpu_id = 2
        else:
            gpu_id = 1

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        config_env.SIMULATOR.NUM_AGENTS = self.num_agents
        config_env.SIMULATOR.SEED = args.seed
        config_env.SIMULATOR.USE_DIFFERENT_START_POS = args.use_different_start_pos
        config_env.SIMULATOR.USE_FIXED_START_POS = args.use_fixed_start_pos

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

    def reset(self):
        obs, infos = self.env.reset()
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        rewards = np.expand_dims(np.array(infos['explored_reward']), axis=1)
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_short_term_goal(self, inputs):
        outputs = self.env.get_short_term_goal(inputs)
        return outputs
