import random

import gfootball.env as football_env
from gym import spaces
import numpy as np
KEEP_INDEX = {
    'academy_3_vs_1_with_keeper': [0,   1,   2,   3,   4,   5,   6,   7,  22,  23,  24,  25,  26, \
        27,  28,  29,  44,  45,  46,  47,  66,  67,  68,  69,  88,  89, \
        90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 108, 110, 112, 113]
}


class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args):
        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.env = football_env.create_environment(
            env_name=args.scenario_name,
            stacked=args.use_stacked_frames,
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
        self.remove_redundancy = args.remove_redundancy
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        if self.remove_redundancy:
            feature_length = len(KEEP_INDEX[args.scenario_name])
        
            if self.num_agents == 1:
                self.action_space.append(self.env.action_space)
                self.observation_space.append(self.env.observation_space[:feature_length]) # ! check this
                self.share_observation_space.append(self.env.observation_space[:feature_length])
            else:
                for idx in range(self.num_agents):
                    self.action_space.append(spaces.Discrete(
                        n=self.env.action_space[idx].n
                    ))
                    self.observation_space.append(spaces.Box(
                        low=self.env.observation_space.low[idx][:feature_length],
                        high=self.env.observation_space.high[idx][:feature_length],
                        shape=(feature_length,),
                        dtype=self.env.observation_space.dtype
                    ))
                    self.share_observation_space.append(spaces.Box(
                        low=self.env.observation_space.low[idx][:feature_length],
                        high=self.env.observation_space.high[idx][:feature_length],
                        shape=(feature_length,),
                        dtype=self.env.observation_space.dtype
                    ))
        else:
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
        obs = self._obs_wrapper(obs)
        if self.remove_redundancy:
            obs = obs[:, KEEP_INDEX[self.scenario_name]]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        if self.remove_redundancy:
            obs = obs[:, KEEP_INDEX[self.scenario_name]]
        reward = reward.reshape(self.num_agents, 1)
        done = np.array([done] * self.num_agents)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs
