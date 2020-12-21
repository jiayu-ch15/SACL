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
from onpolicy.envs.smarts.obs_adapter_fc import \
    get_lan_ttc_observation_adapter
from onpolicy.envs.smarts.rew_adapter_fc import get_reward_adapter


def defin_obs_space_dict(all_args):

    if all_args.use_proximity:
        _LANE_TTC_OBSERVATION_SPACE = gym.spaces.Dict(
            {
                "distance_to_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
                "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                "neighbor":gym.spaces.Box(low=-1e3, high=1e3, shape=(all_args.neighbor_num * 5,)),
                "proximity":gym.spaces.Box(low=-1e10, high=1e10, shape=(8,)),
            }
        )
    else:
        _LANE_TTC_OBSERVATION_SPACE = gym.spaces.Dict(
            {
                "distance_to_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
                "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                "neighbor": gym.spaces.Box(low=-1e3, high=1e3, shape=(all_args.neighbor_num * 5,)),
            }
        )
    return _LANE_TTC_OBSERVATION_SPACE

def get_dict_dim(space_dict):
    dim = 0
    for key in space_dict.spaces.keys():
        space = list(space_dict[key].shape)
        dim += reduce(lambda x, y: x*y, space)
    return dim

@dataclass
class Adapter:
    space: gym.Space
    transform: Callable

def action_adapter(policy_action):
    if isinstance(policy_action, (list, tuple, np.ndarray)):
        action = np.argmax(policy_action)
    else:
        action = policy_action
    action_dict = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    return action_dict[action]


class SMARTSEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""
    def __init__(self, all_args, seed):
        self.all_args=all_args
        self._log = logging.getLogger(self.__class__.__name__)
        smarts.core.seed(seed)
        self._dones_registered = 0
        self.obs_dict=defin_obs_space_dict(all_args)
        self.neighbor_num=all_args.neighbor_num

        self.rews_mode=all_args.rews_mode
        self.n_agents = all_args.num_agents
        self.obs_dim = get_dict_dim(self.obs_dict)
        self.act_dim = 4
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,))] * self.n_agents
        self.share_observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,))] * self.n_agents
        self.action_space = [gym.spaces.Discrete(self.act_dim)] * self.n_agents

        self.agent_ids = ["Agent %i" % i for i in range(self.n_agents)]
        self.scenarios = [(all_args.scenario_path + all_args.scenario_name)]

        self.seed = seed
        self._agent_specs = {
            agent_id: AgentSpec(
                interface=AgentInterface.from_type(AgentType.VulnerDis, max_episode_steps=all_args.episode_length),
                observation_adapter=self.get_obs_adapter(),
                reward_adapter=get_reward_adapter(all_args.rews_mode,self.neighbor_num),
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

    def get_obs_adapter(self):
        def observation_adapter(env_observation):
            lane_ttc_observation_adapter = Adapter(
                space=self.obs_dict, transform=get_lan_ttc_observation_adapter(self.neighbor_num,self.all_args.use_proximity)
            )
            obs = lane_ttc_observation_adapter.transform(env_observation)
            obs_flatten = np.concatenate(list(obs.values()), axis=0)
            return obs_flatten
        return observation_adapter

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
            elif self.rews_mode=="standard":

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
