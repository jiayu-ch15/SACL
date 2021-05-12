import gym
from .gym_minigrid.envs.human import HumanEnv
from onpolicy.envs.gridworld.gym_minigrid.register import register
import numpy as np
from icecream import ic


class GridWorldEnv(object):
    def __init__(self, args):

        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.agent_pos = args.agent_pos
        self.num_obstacles = args.num_obstacles
        self.use_single_reward = args.use_single_reward
        
        register(
            id = self.scenario_name,
            num_agents = self.num_agents,
            grid_size = args.grid_size,
            max_steps = args.max_steps,
            agent_view_size = args.agent_view_size,
            num_obstacles = self.num_obstacles,
            agent_pos = self.agent_pos,
            use_merge = args.use_merge,
            use_same_location = args.use_same_location,
            use_complete_reward = args.use_complete_reward,
            entry_point = 'onpolicy.envs.gridworld.gym_minigrid.envs:MultiExplorationEnv'
        )

        self.env = gym.make(self.scenario_name)
        self.max_steps = self.env.max_steps
        # print("max step is {}".format(self.max_steps))

        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        if choose:
            obs, info = self.env.reset()
        else:
            obs = {}
            for key in self.observation_space[0].spaces.keys():
                obs[key] = np.zeros((self.num_agents, *self.observation_space[0][key].shape), dtype=np.float32)
            info = {}
        return obs, info

    def step(self, actions):
        if not np.all(actions == np.ones((self.num_agents, 1)).astype(np.int) * (-1.0)):
            obs, rewards, done, infos = self.env.step(actions)
            dones = np.array([done for agent_id in range(self.num_agents)])
            if self.use_single_reward:
                rewards = 0.3 * np.expand_dims(infos['agent_explored_reward'], axis=1) + 0.7 * np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
            else:
                rewards = np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        else:
            obs = {}
            for key in self.observation_space[0].spaces.keys():
                obs[key] = np.zeros((self.num_agents, *self.observation_space[0][key].shape), dtype=np.float32)
            rewards = np.zeros((self.num_agents, 1))
            dones = np.array([None for agent_id in range(self.num_agents)])
            infos = {}
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self, mode="multiexploration"):
        if mode == "multiexploration":
            self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)