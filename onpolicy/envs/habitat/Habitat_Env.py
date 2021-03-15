from .exploration_env import Exploration_Env
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

def construct_config(args):
    env_configs = []
    baseline_configs = []
    datasets = []

    basic_config = cfg_env(config_paths=
                           ["/home/yuchao/project/onpolicy/onpolicy/envs/habitat/habitat-lab/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    if len(scenes) > 0:
        assert len(scenes) >= args.n_rollout_threads, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.n_rollout_threads))

    for i in range(args.n_rollout_threads):
        config_env = cfg_env(config_paths=
                             ["habitat_api/configs/" + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]

        gpu_id = 0 # ! strange here
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = 10
        config_env.DATASET.SPLIT = args.split

        dataset = PointNavDatasetV1(config_env.DATASET)
        datasets.append(dataset)

        config_env.defrost()
        config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
        print("Loading {}".format(config_env.SIMULATOR.SCENE))
        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

    return env_configs, baseline_configs, datasets

class MultiHabitatEnv(object):
    def __init__(self, args, rank, config_env, config_baseline, dataset):
        self.env = Exploration_Env(args, rank, config_env, config_baseline, dataset)

        self.num_agents = args.num_agents
        
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / args.global_downscaling), \
                        int(full_h / args.global_downscaling)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        global_observation_space['global_obs'] = gym.spaces.Box(low=0, high=1, shape=(8, local_w, local_h), dtype='uint8')
        global_observation_space['global_orientation'] = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype='long')
        global_observation_space = gym.spaces.Dict(global_observation_space)
        
        for agent_id in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(global_observation_space)  
            self.action_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32))

        
    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        obs, infos = self.env.reset()
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions_env)
        rewards = [infos['exp_reward']] * self.num_agents
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_short_term_goal(self, inputs):
        outputs = self.env.get_short_term_goal(inputs)
        return outputs

