import gym
from .gym_minigrid.register import env_list

print('%d environments registered' % len(env_list))

class GridWorldEnv(object):
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.env = gym.make(self.scenario_name)
        self.max_steps = self.env.max_steps
        print("max step is {}".format(self.max_steps))

        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        for agent_id in range(self.num_agents):
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
            self.action_space.append( self.env.action_space)
        

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, [rewards], [dones], infos

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        if mode == "human":
            self.env.render(mode=mode)
        else:
            return self.env.render(mode=mode)