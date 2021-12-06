import gfootball.env as football_env
from gym import spaces
import numpy as np

class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args):
        self.num_agents = args.num_agents
        self.env = football_env.create_environment(
            env_name=args.scenario_name,
            stacked=args.stacked,
            representation=args.representation,
            rewards=args.rewards,
            write_goal_dumps=args.write_goal_dumps,
            write_full_episode_dumps=args.write_full_episode_dumps,
            render=args.render,
            write_video=args.save_gifs,
            dump_frequency=args.dump_frequency,
            logdir=args.log_dir,
            extra_players=None, # TODO(zelaix) chekc this arg
            number_of_left_players_agent_controls=args.num_agents,
            number_of_right_players_agent_controls=0,
            channel_dimensions=(args.smm_width, args.smm_height)
        )

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        if self.num_agents == 1:
            self.action_space.append(self.env.action_space)
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
        else:
            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[idx].n
                ))
                self.observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))
                self.share_observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward[:, np.newaxis]
        done = np.array([done] * self.num_agents)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def close(self):
        self.env.close()
