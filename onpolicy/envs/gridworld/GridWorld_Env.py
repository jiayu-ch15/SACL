import gym
from .gym_minigrid.envs.human import HumanEnv
from onpolicy.envs.gridworld.gym_minigrid.register import register
import numpy as np

class GridWorldEnv(object):
    def __init__(self, args):

        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.agent_pos = args.agent_pos
        self.num_obstacles = args.num_obstacles

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
        obs, info = self.env.reset(choose=choose)
        return obs, info

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        rewards = np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self, mode="multiexploration"):
        if mode == "multiexploration":
            self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)