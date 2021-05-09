import gym
from .gym_minigrid.envs.human import HumanEnv
from onpolicy.envs.gridworld.gym_minigrid.register import register
import numpy as np

class GridWorldEnv(object):
    def __init__(self, args):

        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.use_random_pos = args.use_random_pos
        self.agent_pos = None if self.use_random_pos else args.agent_pos
        self.num_obstacles = args.num_obstacles
        self.use_single_reward = args.use_single_reward

        register(
            id = self.scenario_name,
            grid_size = args.grid_size,
            max_steps = args.max_steps,
            agent_view_size = args.agent_view_size,
            num_agents = self.num_agents,
            num_obstacles = self.num_obstacles,
            agent_pos = self.agent_pos, 
            direction_alpha = args.direction_alpha,
            use_human_command = args.use_human_command,  
            use_merge = args.use_merge, 
            use_same_location = args.use_same_location,
            use_complete_reward = args.use_complete_reward, 
            use_direction_encoder = args.use_direction_encoder,
            use_fixed_goal_pos = args.use_fixed_goal_pos, 
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
        if self.use_single_reward:
            rewards = [[0.3 * infos['agent_explored_reward'] + 0.7 * infos['merge_explored_reward'] + rewards] for _ in range(self.num_agents)]
        else:
            rewards = [[infos['merge_explored_reward'] + rewards] for _ in range(self.num_agents)]
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self, mode="multiexploration"):
        if mode == "multiexploration":
            self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)
            