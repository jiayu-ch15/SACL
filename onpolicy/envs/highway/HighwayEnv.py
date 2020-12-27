import gym
import numpy as np
from functools import reduce

from onpolicy.envs.highway.common.factory import load_environment

class HighwayEnv(gym.core.Wrapper):
    def __init__(self, all_args):
        self.n_agents = all_args.num_agents
        self.use_centralized_V = all_args.use_centralized_V
        self.env_dict={
                    "id": all_args.scenario_name,
                    "import_module": "onpolicy.envs.highway.highway_env",
                    "controlled_vehicles": self.n_agents,
                    "duration": all_args.episode_length,

                    "action": {
                        "type": "MultiAgentAction",
                        "action_config": {
                            "type": "DiscreteMetaAction"
                        }
                    },
                    "observation": {
                        "type": "MultiAgentObservation",
                        "observation_config": {
                            "type": "Kinematics"
                        }
                    }
                }
        env = load_environment(self.env_dict)

        super().__init__(env)

        self.new_observation_space = []
        self.share_observation_space = []
        for agent_id in range(self.n_agents):
            obs_shape = list(self.observation_space[agent_id].shape)
            self.obs_dim = reduce(lambda x, y: x*y, obs_shape)
            self.share_obs_dim = self.obs_dim * self.n_agents if self.use_centralized_V else self.obs_dim
            self.new_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,)))
            self.share_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.share_obs_dim,)))
        
        self.observation_space = self.new_observation_space
        self.action_space=list(self.action_space)
    def step(self, action):
        o, r, d, infos = self.env.step(tuple(action[0]))
        obs = [np.concatenate(o[i]) for i in range(self.n_agents)]
        rewards = [[r] for i in range(self.n_agents)]
        dones = [d for i in range(self.n_agents)]
        return obs, rewards, dones, infos

    def reset(self):
        o = self.env.reset()
        obs = [np.concatenate(o[i]) for i in range(self.n_agents)]
        return obs
