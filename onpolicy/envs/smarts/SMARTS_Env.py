import gym
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, AgentPolicy
from typing import Callable
from dataclasses import dataclass
from smarts.core.utils.math import vec_2d, vec_to_radians, squared_dist
from functools import reduce


@dataclass
class Adapter:
    space: gym.Space
    transform: Callable


_LANE_TTC_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)

def get_dict_dim(space_dict):
    dim = 0
    for key in space_dict.spaces.keys():
        space = list(space_dict[key].shape)
        dim += reduce(lambda x, y: x*y, space)
    return dim


def _lane_ttc_observation_adapter(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    ego_ttc, ego_lane_dist = _ego_ttc_lane_dist(env_observation, closest_wp.lane_index)

    return {
        "distance_from_center": np.array([norm_dist_from_center]),
        "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
        "speed": np.array([ego.speed]),
        "steering": np.array([ego.steering]),
        "ego_ttc": np.array(ego_ttc),
        "ego_lane_dist": np.array(ego_lane_dist),
    }


lane_ttc_observation_adapter = Adapter(
    space=_LANE_TTC_OBSERVATION_SPACE, transform=_lane_ttc_observation_adapter
)

def _ego_ttc_lane_dist(env_observation, ego_lane_index):
    ttc_by_p, lane_dist_by_p = _ttc_by_path(env_observation)

    return _ego_ttc_calc(ego_lane_index, ttc_by_p, lane_dist_by_p)


def _ttc_by_path(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    neighborhood_vehicle_states = env_observation.neighborhood_vehicle_states

    # first sum up the distance between waypoints along a path
    # ie. [(wp1, path1, 0),
    #      (wp2, path1, 0 + dist(wp1, wp2)),
    #      (wp3, path1, 0 + dist(wp1, wp2) + dist(wp2, wp3))]

    wps_with_lane_dist = []
    for path_idx, path in enumerate(waypoint_paths):
        lane_dist = 0.0
        for w1, w2 in zip(path, path[1:]):
            wps_with_lane_dist.append((w1, path_idx, lane_dist))
            lane_dist += np.linalg.norm(w2.pos - w1.pos)
        wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

    # next we compute the TTC along each of the paths
    ttc_by_path_index = [1000] * len(waypoint_paths)
    lane_dist_by_path_index = [1] * len(waypoint_paths)
    if neighborhood_vehicle_states is not None:
        for v in neighborhood_vehicle_states:
            # find all waypoints that are on the same lane as this vehicle
            wps_on_lane = [
                (wp, path_idx, dist)
                for wp, path_idx, dist in wps_with_lane_dist
                if wp.lane_id == v.lane_id
            ]

            if not wps_on_lane:
                # this vehicle is not on a nearby lane
                continue

            # find the closest waypoint on this lane to this vehicle
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane, key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position))
            )

            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue

            relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            if abs(relative_speed_m_per_s) < 1e-5:
                relative_speed_m_per_s = 1e-5

            ttc = lane_dist / relative_speed_m_per_s
            ttc /= 10
            if ttc <= 0:
                # discard collisions that would have happened in the past
                continue

            lane_dist /= 100
            lane_dist_by_path_index[path_idx] = min(
                lane_dist_by_path_index[path_idx], lane_dist
            )
            ttc_by_path_index[path_idx] = min(ttc_by_path_index[path_idx], ttc)

    return ttc_by_path_index, lane_dist_by_path_index


def _ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist_by_path):
    ego_ttc = [0] * 3
    ego_lane_dist = [0] * 3

    ego_ttc[1] = ttc_by_path[ego_lane_index]
    ego_lane_dist[1] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path) - 1
    min_lane_index = 0
    if ego_lane_index + 1 > max_lane_index:
        ego_ttc[2] = 0
        ego_lane_dist[2] = 0
    else:
        ego_ttc[2] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[2] = lane_dist_by_path[ego_lane_index + 1]
    if ego_lane_index - 1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 1]
    return ego_ttc, ego_lane_dist


def observation_adapter(env_observation):
    obs = lane_ttc_observation_adapter.transform(env_observation)
    obs_flatten = np.concatenate(list(obs.values()), axis=0)
    return obs_flatten


def reward_adapter(env_obs, env_reward):
    return env_reward


def action_adapter(policy_action):
    if isinstance(policy_action, (list, tuple, np.ndarray)):
        action = np.argmax(policy_action)
    else:
        action = policy_action
    action_dict = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    return action_dict[action]

class SMARTSEnv():
    def __init__(self, all_args, seed):

        self.n_agents = all_args.num_agents
        self.obs_dim = get_dict_dim(_LANE_TTC_OBSERVATION_SPACE)
        self.act_dim = 4
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,))] * self.n_agents
        self.share_observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,))] * self.n_agents
        self.action_space = [gym.spaces.Discrete(self.act_dim)] * self.n_agents

        self.agent_ids = ["Agent %i" % i for i in range(self.n_agents)]
        self.scenarios = [(all_args.scenario_path + all_args.scenario_name)]

        self.headless = all_args.headless
        self.seed = seed

        self.agent_specs = {
            agent_id: AgentSpec(
                interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=all_args.episode_length),
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                action_adapter=action_adapter,
            )
            for agent_id in self.agent_ids
        }

        self.base_env = gym.make(
            id="smarts.env:hiway-v0",
            scenarios=self.scenarios,
            agent_specs=self.agent_specs,
            headless=self.headless,
            seed=self.seed,
        )

    @property
    def scenario_log(self):
        self.base_env.scenario_log

    def reset(self):
        try:
            self.current_observations = self.base_env.reset()
        except:
            self.base_env.close()
            self.base_env = gym.make(
                "smarts.env:hiway-v0",
                scenarios=self.scenarios,
                agent_specs=self.agent_specs,
                headless=self.headless,
                seed=self.seed,
            )
            self.current_observations = self.base_env.reset()

        return self.get_obs()

    def close(self):
        self.base_env.close()

    def step(self, action_n):
        actions = dict(zip(self.agent_ids, action_n))
        self.current_observations, rewards, dones, infos = self.base_env.step(actions)
        r_n = []
        d_n = []
        for agent_id in self.agent_ids:
            r_n.append([rewards.get(agent_id, 0.)])
            d_n.append(dones.get(agent_id, True))

        return self.get_obs(),r_n, d_n, infos

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for agent_id in self.agent_ids:
            obs_n.append(self.current_observations.get(agent_id, np.zeros(self.observation_space[agent_id].shape[0])))
        return list(obs_n)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        return np.asarray(self.get_obs()).flatten()

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size() * self.n_agents

