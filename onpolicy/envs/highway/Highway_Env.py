import gym
import numpy as np
from functools import reduce
import torch
from onpolicy.envs.highway.common.factory import load_environment
from copy import deepcopy
from pathlib import Path
import os

class HighwayEnv(gym.core.Wrapper):
    def __init__(self, all_args):
        self.all_args = all_args

        # render parameters
        self.use_offscreen_render = all_args.use_offscreen_render
        self.use_render_vulnerability = all_args.use_render_vulnerability

        # type parameters
        self.task_type = all_args.task_type

        # [vi] ValueIteration
        # [rvi] RobustValueIteration
        # [mcts] MonteCarloTreeSearchDeterministic
        # [d3qn] duel_ddqn
        self.dummy_agent_type = all_args.dummy_agent_type

        # [d3qn] duel_ddqn
        # [ppo] onpolicy
        self.other_agent_type = all_args.other_agent_type
        self.use_same_other_policy = all_args.use_same_other_policy

        # task parameters
        self.scenario_name = all_args.scenario_name
        self.horizon = all_args.horizon
        self.use_centralized_V = all_args.use_centralized_V
        self.simulation_frequency = all_args.simulation_frequency
        self.collision_reward = all_args.collision_reward
        self.dt = all_args.dt

        self.n_defenders = all_args.n_defenders
        self.n_attackers = all_args.n_attackers
        self.n_dummies = all_args.n_dummies
        if self.task_type == "attack":
            self.n_agents = self.n_attackers
            self.n_other_agents = self.n_defenders
            self.load_start_idx = 0
            self.train_start_idx = self.n_defenders
        elif self.task_type == "defend":
            self.n_agents = self.n_defenders
            self.n_other_agents = self.n_attackers
            self.load_start_idx = self.n_defenders
            self.train_start_idx = 0
        elif self.task_type == "all":
            self.n_agents = self.n_defenders + self.n_attackers
            self.n_other_agents = 0
            self.load_start_idx = self.n_defenders + self.n_attackers
            self.train_start_idx = 0
        else:
            raise NotImplementedError

        self.env_dict={
            "id": self.scenario_name,
            "import_module": "onpolicy.envs.highway.highway_env",
            # u must keep this order!!! can not change that!!!
            "controlled_vehicles": self.n_defenders + self.n_attackers + self.n_dummies,
            "n_defenders": self.n_defenders,
            "n_attackers": self.n_attackers,
            "n_dummies": self.n_dummies,
            "duration": self.horizon,
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction"
                }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "other_vehicles_type": "onpolicy.envs.highway.highway_env.vehicle.behavior.IDMVehicle",
            # other vehicles could also set as "onpolicy.envs.highway.highway_env.vehicle.dummy.DummyVehicle" 
            # Dummy Vehicle is the vehicle keeping lane with the speed of 25 m/s.
            # While IDM Vehicle is the vehicle which is able to change lane and speed based on the obs of its front & rear vehicle
            "vehicles_count": 50,
            "offscreen_rendering": self.use_offscreen_render,
            "collision_reward": self.collision_reward,
            "simulation_frequency": self.simulation_frequency,
            "dt": self.dt
        }
        
        self.env_init = load_environment(self.env_dict)

        super().__init__(self.env_init)

        # get new obs and action space
        self.all_observation_space = [] 
        self.all_action_space = []
        for agent_id in range(self.n_attackers + self.n_defenders + self.n_dummies):
            obs_shape = list(self.observation_space[agent_id].shape)
            self.obs_dim = reduce(lambda x, y: x*y, obs_shape)
            self.all_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,)))
            self.all_action_space.append(self.action_space[agent_id])
        
        # here we load other agents and dummies, can not change the order of the following code!!
        if self.n_other_agents > 0:
            self.load_other_agents()

        if self.n_dummies > 0:
            self.load_dummies()
        
        # get new obs and action space
        self.new_observation_space = [] # just for store
        self.new_action_space = [] # just for store
        self.share_observation_space = []
        for agent_id in range(self.n_agents):
            obs_shape = list(self.observation_space[self.train_start_idx + agent_id].shape)
            self.obs_dim = reduce(lambda x, y: x*y, obs_shape)
            self.share_obs_dim = self.obs_dim * self.n_agents if self.use_centralized_V else self.obs_dim
            self.new_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,)))
            self.share_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.share_obs_dim,)))
            self.new_action_space.append(self.action_space[self.train_start_idx + agent_id])
        
        self.observation_space = self.new_observation_space
        self.action_space = self.new_action_space

        self.cache_frames = []

    def load_dummies(self):
        self.dummies = []

        if self.dummy_agent_type in ["vi", "rvi"]:
            if self.dummy_agent_type == "vi":
                from .agents.dynamic_programming.value_iteration import ValueIterationAgent as DummyAgent
            else:
                from .agents.dynamic_programming.robust_value_iteration import RobustValueIterationAgent as DummyAgent
            
            agent_config = {
                "env_preprocessors": [{"method":"simplify"}],
                "budget": 50
            }

        elif self.dummy_agent_type == "mcts":
            from .agents.tree_search.mcts import MCTSAgent as DummyAgent 

            agent_config = {
                "max_depth": 1,
                "budget": 200,
                "temperature": 200
            }  

        elif self.dummy_agent_type == "d3qn":
            from .agents.policy_pool.dqn.policy import actor as DummyAgent
            agent_config ={
                "model": {
                    "type": "DuelingNetwork",
                    "base_module": {
                        "layers": [256, 128]
                    },
                    "value": {
                        "layers": [128]
                    },
                    "advantage": {
                        "layers": [128]
                    }
                }
            }

        else:
            raise NotImplementedError

        for dummy_id in range(self.n_dummies):
            dummy = DummyAgent(env = self.env_init, 
                                config = agent_config,                
                                vehicle_id = dummy_id + self.n_attackers + self.n_defenders)
            if self.dummy_agent_type == "d3qn":
                policy_path = self.all_args.policy_path # ! need to review this code, policy path
                policy_state_dict = torch.load(policy_path, map_location='cpu')
                dummy.load_state_dict(policy_state_dict)
                dummy.eval()

            self.dummies.append(dummy)

    def load_other_agents(self):
        """
            Load trained agent which serves as the defender/attacker based on type of the task 
        """
        if self.other_agent_type == "d3qn":
            from .agents.policy_pool.dqn.policy import actor as Policy
        elif self.other_agent_type == "ppo":
            from .agents.policy_pool.ppo.policy import actor as Policy
        else:
            raise NotImplementedError
        
        if self.use_same_other_policy:
            policy_path = self.all_args.other_agent_policy_path
            self.other_agents = Policy(self.all_args,
                                self.all_observation_space[self.load_start_idx],
                                self.all_action_space[self.load_start_idx],
                                hidden_size = self.all_args.hidden_size, # re-structure this!
                                use_recurrent_policy = self.all_args.use_recurrent_policy) # cpu is fine actually, keep it for now.
            policy_state_dict = torch.jit.load(policy_path, map_location='cpu')
            self.other_agents.load_state_dict(policy_state_dict)
            self.other_agents.eval()
        else:
            # TODO: need to support different models in the future.
            policy_path = self.all_args.other_agent_policy_path # ! should be a list or other ways in this case
            self.other_agents = []           
            for agent_id in range(self.n_other_agents):
                policy = Policy(self.all_args,
                                self.all_observation_space[self.load_start_idx + agent_id],
                                self.all_action_space[self.load_start_idx + agent_id],
                                hidden_size = self.all_args.hidden_size, # re-structure this!
                                use_recurrent_policy = self.all_args.use_recurrent_policy) # cpu is fine actually, keep it for now.
                policy_state_dict = torch.load(policy_path, map_location='cpu')
                policy.load_state_dict(policy_state_dict)
                policy.eval()
                self.other_agents.append(policy)

    def step(self, action):
        if not np.all(action == np.ones((self.n_agents, 1)).astype(np.int) * (-1)):
            # we need to get actions of other agents
            if self.n_other_agents > 0:
                if self.use_same_other_policy:
                    other_actions, self.rnn_states \
                        = self.other_agents(self.other_obs, 
                                            self.rnn_states, 
                                            self.masks, 
                                            deterministic=True)
                    other_actions = other_actions.detach().numpy()

                else:
                    other_actions = []
                    for agent_id in range(self.n_other_agents):
                        self.other_agents[agent_id].eval()
                        other_action, rnn_state = \
                            self.other_agents[agent_id](self.other_obs[agent_id,:], 
                                                        self.rnn_states[agent_id,:], 
                                                        self.masks[agent_id,:], 
                                                        deterministic=True)
                        other_actions.append(other_action.detach().numpy())
                        self.rnn_states[agent_id] = rnn_state.detach().numpy()
                if self.train_start_idx == 0:
                    action = np.concatenate([action, other_actions])
                else:
                    action = np.concatenate([other_actions, action])

            # then we need to get actions of dummies
            if self.n_dummies > 0:
                dummy_actions = [[self.dummies[dummy_id].act(self.dummy_obs[dummy_id])] for dummy_id in range(self.n_dummies)]
                action = np.concatenate([action, dummy_actions])

            # for discrete action, drop the unneeded axis
            action = np.squeeze(action, axis=-1)

            all_obs, all_rewards, all_dones, infos = self.env.step(tuple(action))

            self.current_step += 1
            
            # obs
            # 1. train obs
            obs = np.array([np.concatenate(all_obs[self.train_start_idx + agent_id]) for agent_id in range(self.n_agents)])
            # 2. other obs
            self.other_obs = np.array([np.concatenate(all_obs[self.load_start_idx + agent_id]) \
                                    for agent_id in range(self.n_other_agents)])
            # 3. dummy obs
            self.dummy_obs = np.array([np.concatenate(all_obs[self.n_attackers + self.n_defenders + agent_id]) \
                                        for agent_id in range(self.n_dummies)])

            # rewards
            # ! @zhuo if agent is dead, rewards need to be set zero!
            # 1. train rewards
            rewards = [[all_rewards[self.train_start_idx + agent_id]] for agent_id in range(self.n_agents)]
            self.episode_rewards.append(rewards)
            # 2. other rewards
            other_rewards = [[all_rewards[self.load_start_idx + agent_id]] \
                                    for agent_id in range(self.n_other_agents)]
            self.episode_other_rewards.append(other_rewards)
            # 3. dummy rewards
            dummy_rewards = [[all_rewards[self.n_attackers + self.n_defenders + dummy_id]] \
                                    for dummy_id in range(self.n_dummies)]
            self.episode_dummy_rewards.append(dummy_rewards)

            # update info
            speeds=[infos["speed"][agent_id] for agent_id in range(self.n_defenders + self.n_attackers)]
            self.episode_speeds.append(speeds)
            for i, s in enumerate(np.mean(self.episode_speeds, axis=0)):
                if i <self.n_defenders:
                    infos.update({"defender_{}_speed".format(i): s})
                else:
                    infos.update({"attacker_{}_speed".format(i): s})

            # ! @zhuo u need to use this one!
            # 1. train dones
            dones = [all_dones[self.train_start_idx + agent_id] for agent_id in range(self.n_agents)]
            # 2. other dones
            other_dones = [all_dones[self.load_start_idx + agent_id] for agent_id in range(self.n_other_agents)]
            # 3. dummy dones
            dummy_dones = [all_dones[self.n_attackers + self.n_defenders + dummy_id] for dummy_id in range(self.n_dummies)]

            if np.all(dones, axis=-1):
                crashs = [infos["crashed"][agent_id] for agent_id in range(self.n_defenders + self.n_attackers)]
                for i, c in enumerate(crashs):
                    if i < self.n_defenders:
                        infos.update({"defender_{}_crash".format(i): c})
                    else:
                        infos.update({"attacker_{}_crash".format(i): c})
                infos.update({"episode_len": self.current_step})

            if self.use_render_vulnerability:
                self.cache_frames.append(self.render('rgb_array'))
                if np.all(dones): # save gif
                    self.pick_frames.append(self.render_vulnerability(self.current_step))
                    infos.update({"frames": self.pick_frames})

            if self.task_type == "attack":
                adv_rew = self.env.unwrapped.adv_rew()
            else:
                adv_rew = 0

            infos.update({"episode_rewards": np.sum(self.episode_rewards, axis=0),
                        "episode_other_rewards": np.sum(self.episode_other_rewards, axis=0) if self.n_other_agents > 0 else 0.0,
                        "episode_dummy_rewards": np.sum(self.episode_dummy_rewards, axis=0) if self.n_dummies > 0 else 0.0,
                        "adv_rew": adv_rew})
        else:
            obs = np.zeros((self.n_agents, self.obs_dim))
            rewards = np.zeros((self.n_agents, 1))
            dones = [None for agent_id in range(self.n_agents)]
            infos = {}
        return obs, rewards, dones, infos

    def reset(self, choose = True):
        if choose:
            self.episode_speeds = []
            self.episode_rewards = []
            self.episode_dummy_rewards = []
            self.episode_other_rewards = []
            self.current_step = 0
            self.cache_frames = []
            self.pick_frames = []

            all_obs = self.env.reset()

            # ? dummy needs to take all obs ?
            self.dummy_obs = np.array([np.concatenate(all_obs[self.n_attackers + self.n_defenders + agent_id]) \
                    for agent_id in range(self.n_dummies)])
            # deal with other agents
            self.rnn_states = np.zeros((self.n_other_agents, self.all_args.hidden_size), dtype=np.float32)
            # o = [np.concatenate(all_obs[i]) for i in range(len(all_obs))]

            self.other_obs = np.array([np.concatenate(all_obs[self.load_start_idx + agent_id]) \
                                    for agent_id in range(self.n_other_agents)])
            self.masks = np.ones((self.n_other_agents, 1), dtype=np.float32)

            # deal with agents that need to train
            obs = np.array([np.concatenate(all_obs[self.train_start_idx + agent_id]) for agent_id in range(self.n_agents)])
            if self.use_render_vulnerability:
                self.cache_frames.append(self.render('rgb_array'))
                self.current_step += 1
                                                
        else:
            obs = np.zeros((self.n_agents, self.obs_dim))
        return obs

    def render_vulnerability(self, end_idx, T = 10):
        '''
        assume we find a crash at step t, it could be a vulunerability, then we need to record the full process of the crash.
        start_state is the state at step t-10 (if step t < 10, then we get state at step t=0).
        T is the render length, which is default 10.
        '''
        start_idx = end_idx - T
        if start_idx < 0:
            start_idx = 0
        return self.cache_frames[start_idx:end_idx]
