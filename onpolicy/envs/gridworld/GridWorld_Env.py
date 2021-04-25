import gym
from .gym_minigrid.envs.human import HumanEnv
from onpolicy.envs.gridworld.gym_minigrid.register import register

class GridWorldEnv(object):
    def __init__(self, args):

        self.num_agents = args.num_agents
        self.num_preies = args.num_preies
        self.num_obstacles = args.num_obstacles
        self.scenario_name = args.scenario_name
        self.direction_alpha = args.direction_alpha
        self.use_human_command = args.use_human_command

        register(
            id = self.scenario_name,
            num_agents = self.num_agents,
            num_preies = self.num_preies,
            num_obstacles = self.num_obstacles,
            direction_alpha = self.direction_alpha,
            use_human_command = self.use_human_command,
            entry_point = 'onpolicy.envs.gridworld.gym_minigrid.envs:HumanEnv'
        )
        self.env = gym.make(self.scenario_name)
        self.max_steps = self.env.max_steps
        print("max step is {}".format(self.max_steps))

        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        obs = self.env.reset(choose=choose)
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        if mode == "human":
            self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)