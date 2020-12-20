import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from typing import Callable
from dataclasses import dataclass
from smarts.core.utils.math import vec_2d
from functools import reduce
import logging
import gym
import smarts
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario
from smarts.core.utils.visdom_client import VisdomClient
from envision.client import Client as Envision
import math
from scipy.spatial import distance


@dataclass
class Adapter:
    space: gym.Space
    transform: Callable


NEIGHBOR_NUM=3

_LANE_TTC_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_to_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "neighbor":gym.spaces.Box(low=-1e3, high=1e3, shape=(NEIGHBOR_NUM * 5,)),
        "proximity":gym.spaces.Box(low=-1e10, high=1e10, shape=(8,)),
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

    if env_observation.neighborhood_vehicle_states is not None:
        neighbor=cal_neighbor(env_observation, NEIGHBOR_NUM)
    else:
        neighbor = [0] * (NEIGHBOR_NUM * 5)

    if env_observation.occupancy_grid_map is not None:
        proximity = cal_proximity(env_observation)
    else:
        proximity = [0] * 8


    return {
        "distance_to_center": np.array([norm_dist_from_center]),
        "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
        "speed": np.array([ego.speed]),
        "steering": np.array([ego.steering]),
        "ego_ttc": np.array(ego_ttc),
        "ego_lane_dist": np.array(ego_lane_dist),
        "neighbor": np.array(neighbor),
        "proximity": np.array(proximity)
    }


lane_ttc_observation_adapter = Adapter(
    space=_LANE_TTC_OBSERVATION_SPACE, transform=_lane_ttc_observation_adapter
)

def _ego_ttc_lane_dist(env_observation, ego_lane_index):
    ttc_by_p, lane_dist_by_p = _ttc_by_path(env_observation)

    return _ego_ttc_calc(ego_lane_index, ttc_by_p, lane_dist_by_p)


def _cal_angle(vec):

    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle

def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)

    return groups

def cal_neighbor(env_obs, closest_neighbor_num):
    ego = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
    # dist, speed, ttc, pos
    features = np.zeros((closest_neighbor_num, 5))
    # fill neighbor vehicles into closest_neighboor_num areas
    surrounding_vehicles = _get_closest_vehicles(
        ego, neighbor_vehicle_states, n=closest_neighbor_num
    )

    heading_angle = ego.heading + math.pi / 2.0
    ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
    for i, v in surrounding_vehicles.items():
        if v[0] is None:
            continue
        v = v[0]
        rel_pos = np.asarray(
            list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
        )

        rel_dist = np.sqrt(rel_pos.dot(rel_pos))

        v_heading_angle = math.radians(v.heading)
        v_heading_vec = np.asarray(
            [math.cos(v_heading_angle), math.sin(v_heading_angle)]
        )

        ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
        rel_pos_norm_2 = rel_pos.dot(rel_pos)
        v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)

        ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
            ego_heading_norm_2 + rel_pos_norm_2
        )

        v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
            v_heading_norm_2 + rel_pos_norm_2
        )

        rel_speed = 0
        if ego_cosin <= 0 and v_cosin > 0:
            rel_speed = 0
        else:
            rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

        ttc = min(rel_dist / max(1e-5, rel_speed), 1e3)

        features[i, :] = np.asarray(
            [rel_dist, rel_speed, ttc, rel_pos[0], rel_pos[1]]
        )

    return features.reshape((-1,))


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


def proximity_detection(OGM):
    """
    Detects other vehicles in the vicinity of the ego vehicle
    hard coded for OGM(64, 64, 0.25)
    """
    boxes = []
    boxes += [
        OGM[11:25, 23:27],  # front left
        OGM[11:25, 27:37],  # front center
        OGM[11:25, 37:41],  # front right
        OGM[25:39, 23:27],  # left
        OGM[25:39, 37:41],  # right
        OGM[41:53, 23:27],  # back left
        OGM[41:53, 27:37],  # back center
        OGM[41:53, 37:41],  # back right
    ]
    output = np.array([b.max() > 0 for b in boxes], np.float32)
    return output

def cal_proximity(env_obs):
    proximity = proximity_detection(env_obs.occupancy_grid_map[1])
    return proximity

def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center


def action_adapter(policy_action):
    if isinstance(policy_action, (list, tuple, np.ndarray)):
        action = np.argmax(policy_action)
    else:
        action = policy_action
    action_dict = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    return action_dict[action]

def observation_adapter(env_observation):
    obs = lane_ttc_observation_adapter.transform(env_observation)
    obs_flatten = np.concatenate(list(obs.values()), axis=0)
    return obs_flatten


#def reward_adapter(env_obs, env_reward):
#    return env_reward

def get_reward_adapter(adapter_type="vanilla"):
    def vanilla(env_obs, env_reward):
        return env_reward

    def single_frame(last_env_obs, env_obs, env_reward):
        penalty, bonus = 0.0, 0.0
        obs = _lane_ttc_observation_adapter(env_obs)
        last_obs = _lane_ttc_observation_adapter(last_env_obs)

        neighbor_features = obs.get("neighbor", None)
        last_neighbor_feature = last_obs.get("neighbor", None)

        # dealing with neighbor_features
        if neighbor_features is not None:
            new_neighbor_feature = neighbor_features.reshape((-1, 5))
            last_neighbor_feature = last_neighbor_feature.reshape((-1, 5))
            mean_dist = np.mean(new_neighbor_feature[:, 0])
            mean_ttc = np.mean(new_neighbor_feature[:, 2])
            mean_dist2 = np.mean(last_neighbor_feature[:, 0])
            # mean_speed2 = np.mean(last_neighbor_feature[:, 1])
            mean_ttc2 = np.mean(last_neighbor_feature[:, 2])

            # this penalty should considering the speed
            # if speed is ...
            ego_speed = env_obs.ego_vehicle_state.speed
            penalty += np.tanh(ego_speed) * (
                0.03 * (mean_dist - mean_dist2)
                # - 0.01 * (mean_speed - mean_speed2)
                + 0.01 * (mean_ttc - mean_ttc2)
            )

        # ======== Penalty: distance to goal =========
        goal = env_obs.ego_vehicle_state.mission.goal

        last_ego_2d_pos = last_env_obs.ego_vehicle_state.position[:2]
        ego_2d_pos = env_obs.ego_vehicle_state.position[:2]

        if hasattr(goal, "position"):
            goal_pos = goal.position
            last_goal_dist = distance.euclidean(last_ego_2d_pos, goal_pos)
            goal_dist = distance.euclidean(ego_2d_pos, goal_pos)
            penalty += 0.1 * (last_goal_dist - goal_dist)
        else:
            raise ValueError(f"Goal type: {type(goal)} has no attr named: position.")

        # ======== Penalty: distance to the center
        if last_obs.get("distance_to_center") is not None:
            diff_dist_to_center_penalty = np.abs(
                last_obs["distance_to_center"]
            ) - np.abs(obs["distance_to_center"])
            penalty += 0.01 * diff_dist_to_center_penalty[0]

        # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
        ego_events = env_obs.events
        # ::collision
        penalty += -50.0 if len(ego_events.collisions) > 0 else 0.0
        # ::off road
        penalty += -50.0 if ego_events.off_road else 0.0
        # ::reach goal
        if ego_events.reached_goal:
            bonus += 20.0

        # ::reached max_episode_step
        if ego_events.reached_max_episode_steps:
            penalty += -0.5
        else:
            bonus += 0.5

        # ======== Penalty: penalise sharp turns done at high speeds =======
        if env_obs.ego_vehicle_state.speed > 60:
            steering_penalty = -pow(
                (env_obs.ego_vehicle_state.speed - 60)
                / 20
                * env_obs.ego_vehicle_state.steering
                / 4,
                2,
            )
        else:
            steering_penalty = 0
        penalty += 0.1 * steering_penalty

        # ========= Bonus: environment reward (distance travelled) ==========
        bonus += 0.05 * env_reward
        return bonus + penalty

    def cruising(env_obs, env_reward):
        global lane_crash_flag
        global intersection_crash_flag

        distance_from_center = get_distance_from_center(env_obs)

        center_penalty = -np.abs(distance_from_center)

        # penalise sharp turns done at high speeds
        if env_obs.ego_vehicle_state.speed * 3.6 > 60:
            steering_penalty = -pow(
                (env_obs.ego_vehicle_state.speed * 3.6 - 60)
                / 20
                * (env_obs.ego_vehicle_state.steering)
                * 45
                / 4,
                2,
            )
        else:
            steering_penalty = 0

        # penalise close proximity to lane cars
        if lane_crash_flag:
            crash_penalty = -5
        else:
            crash_penalty = 0

        # penalise close proximity to intersection cars
        if intersection_crash_flag:
            crash_penalty -= 5

        total_reward = np.sum([1.0 * env_reward])
        total_penalty = np.sum(
            [0.1 * center_penalty, 1 * steering_penalty, 1 * crash_penalty]
        )

        return (total_reward + total_penalty) / 200.0

    return {
        "vanilla": vanilla,
        "single_frame": single_frame,
        "cruising": cruising,
    }[adapter_type]





class SMARTSEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""
    def __init__(self, all_args, seed):
        self.all_args=all_args
        self._log = logging.getLogger(self.__class__.__name__)
        smarts.core.seed(seed)
        self._dones_registered = 0
        global NEIGHBOR_NUM
        NEIGHBOR_NUM = all_args.neighbor_num


        self.rews_mode=all_args.rews_mode
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
        self._agent_specs = {
            agent_id: AgentSpec(
                interface=AgentInterface.from_type(AgentType.VulnerDis, max_episode_steps=all_args.episode_length),
                observation_adapter=observation_adapter,
                reward_adapter=get_reward_adapter(all_args.rews_mode),
                action_adapter=action_adapter,
            )
            for agent_id in self.agent_ids
        }


        self._scenarios_iterator = Scenario.scenario_variations(
            self.scenarios, list(self._agent_specs.keys()), all_args.shuffle_scenarios,
        )


        self.agent_interfaces = {
            agent_id: agent.interface for agent_id, agent in self._agent_specs.items()
        }


        self.envision_client = None
        if not all_args.headless:
            self.envision_client = Envision(
                endpoint=all_args.envision_endpoint, output_dir=all_args.envision_record_data_replay_path
            )

        self.visdom_client = None
        if all_args.visdom:
            self.visdom_client = VisdomClient()

        self._smarts = SMARTS(
            agent_interfaces=self.agent_interfaces,
            traffic_sim=SumoTrafficSimulation(
                headless=all_args.sumo_headless,
                time_resolution=all_args.timestep_sec,
                num_external_sumo_clients=all_args.num_external_sumo_clients,
                sumo_port=all_args.sumo_port,
                auto_start=all_args.sumo_auto_start,
                endless_traffic=all_args.endless_traffic,
            ),
            envision=self.envision_client,
            visdom=self.visdom_client,
            timestep_sec=all_args.timestep_sec,
            zoo_workers=all_args.zoo_workers,
            auth_key=all_args.auth_key,
        )

    @property
    def scenario_log(self):
        scenario = self._smarts.scenario
        return {
            "timestep_sec": self._smarts.timestep_sec,
            "scenario_map": scenario.name,
            "scenario_routes": scenario.route or "",
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    def base_reset(self):
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        env_observations = self._smarts.reset(scenario)
        self.last_obs = env_observations
        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }

        return observations

    def reset(self):
        try:
            self.current_observations = self.base_reset()
        except:
            self.close()
            self._smarts = SMARTS(
                agent_interfaces=self.agent_interfaces,
                traffic_sim=SumoTrafficSimulation(
                    headless=self.all_args.sumo_headless,
                    time_resolution=self.all_args.timestep_sec,
                    num_external_sumo_clients=self.all_args.num_external_sumo_clients,
                    sumo_port=self.all_args.sumo_port,
                    auto_start=self.all_args.sumo_auto_start,
                    endless_traffic=self.all_args.endless_traffic,
                ),
                envision=self.envision_client,
                visdom=self.visdom_client,
                timestep_sec=self.all_args.timestep_sec,
                zoo_workers=self.all_args.zoo_workers,
                auth_key=self.all_args.auth_key,
            )
            self.current_observations = self.base_reset()
        return self.get_obs()


    def base_step(self, agent_actions):
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        observations, rewards, agent_dones, extras = self._smarts.step(agent_actions)

        infos = {
            agent_id: {"scores": value}
            for agent_id, value in extras["scores"].items()
        }
    
        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            if self.rews_mode=="vanilla":
                rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            elif self.rews_mode=="single_frame":

                rewards[agent_id] = agent_spec.reward_adapter(self.last_obs[agent_id], observation, reward)
            elif self.rews_mode == "cruising":
                rewards[agent_id] = agent_spec.reward_adapter(observation, reward)

            self.last_obs[agent_id] = observation
            observations[agent_id] = agent_spec.observation_adapter(observation)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)


        for done in agent_dones.values():
            self._dones_registered += 1 if done else 0

        agent_dones["__all__"] = self._dones_registered == len(self._agent_specs)

        return observations, rewards, agent_dones, infos

    def step(self, action_n):
        actions = dict(zip(self.agent_ids, action_n))
        self.current_observations, rewards, dones, infos = self.base_step(actions)
        obs_n = []
        r_n = []
        d_n = []
        info_n = []
        for agent_id in self.agent_ids:
            obs_n.append(self.current_observations.get(agent_id, np.zeros(self.obs_dim)))
            r_n.append([rewards.get(agent_id, 0.)])
            d_n.append(dones.get(agent_id, True))
            info_n.append(infos.get(agent_id, {'scores':0.}))
        return obs_n, r_n, d_n, info_n


    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for i, agent_id in enumerate(self.agent_ids):
            obs_n.append(self.current_observations.get(agent_id, np.zeros(self.obs_dim)))
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

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        if self._smarts is not None:
            self._smarts.destroy()
