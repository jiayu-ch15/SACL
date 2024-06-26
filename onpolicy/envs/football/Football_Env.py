import random

import gfootball.env as football_env
from gym import spaces
import numpy as np
KEEP_INDEX = {
    'academy_3_vs_1_with_keeper': [0,   1,   2,   3,   4,   5,   6,   7,  \
        22,  23,  24,  25,  26, 27,  28,  29,  44,  45,  46,  47,  \
        66,  67,  68,  69,  \
        88,  89, 90,  91,  92,  93,  94,  95,  96,  97,  98,  99, \
        100, 108, 110, 111, 112, 113, 114],
    'academy_pass_and_shoot_with_keeper': [0,   1,   2,  3,   4,   5,  \
        22,  23,  24,  25,  26,  27,  44,  45,  46,  47, \
        66,  67,  68,  69,  \
        88,  89,  90,  91,  92,  93,  94,  95,  96, 97,  98,  99, \
        108, 110, 111, 112, 113],
    'academy_run_pass_and_shoot_with_keeper': [0,   1,  2,   3,   4,   5,  \
        22,  23,  24,  25,  26,  27,  44,  45,  46,  47, \
        66,  67,  68,  69,  \
        88,  89,  90,  91,  92,  93,  94,  95, 96,  97,  98,  99, \
        108, 110, 111, 112, 113]
}


class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args):
        self.num_agents = args.num_agents
        self.num_red = args.num_red
        self.num_blue = args.num_blue
        self.zero_sum = args.zero_sum
        self.scenario_name = args.scenario_name
        
        # make env
        if not (args.use_render and args.save_videos):
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_red,
                number_of_right_players_agent_controls=args.num_blue,
                channel_dimensions=(args.smm_width, args.smm_height),
                render=(args.use_render and args.save_gifs)
            )
        else:
            # render env and save videos
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_red,
                number_of_right_players_agent_controls=args.num_blue,
                channel_dimensions=(args.smm_width, args.smm_height),
                # video related params
                write_full_episode_dumps=True,
                render=True,
                write_video=True,
                dump_frequency=1,
                logdir=args.video_dir
            )
            
        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
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
                        n=self.env.action_space.nvec[idx]
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
                        n=self.env.action_space.nvec[idx]
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

    def reset(self, task=None):
        obs = self.env.reset(task=task)

        obs = self._obs_wrapper(obs)
        if self.remove_redundancy:
            obs = obs[:, KEEP_INDEX[self.scenario_name]]
        if self.zero_feature:
            obs[obs == -1] = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        if self.remove_redundancy:
            obs = obs[:, KEEP_INDEX[self.scenario_name]]
        if self.zero_feature:
            obs[obs == -1] = 0
        reward = reward.reshape(self.num_agents, 1)
        
        # red : sum(red), blue : sum(blue)
        if self.zero_sum:
            red_reward = np.mean(reward[:self.num_red])
            blue_reward = - red_reward
            reward = [[red_reward]] * self.num_red + [[blue_reward]] * self.num_blue

        done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)

        # get state by obs
        ball_state = obs[0, 88:91]
        left_state = obs[0, : (self.num_red + 1) * 2]
        right_state =  obs[0, 44 : 44 + (self.num_blue + 1) * 2]
        info['state'] = np.concatenate([ball_state, left_state, right_state])

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

    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info