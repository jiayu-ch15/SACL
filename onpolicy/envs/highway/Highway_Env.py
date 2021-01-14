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
        self.dummy_agent_type = all_args.dummy_agent_type
        # [d3qn] duel_ddqn
        self.use_same_dummy_policy = all_args.use_same_dummy_policy
        self.dummy_agent_policy_path = all_args.dummy_agent_policy_path
        if self.dummy_agent_type in ["vi", "rvi","mcts"]:
            assert self.use_same_dummy_policy == False, ("can not use True here!")

        # [d3qn] duel_ddqn
        # [ppo] onpolicy
        self.other_agent_type = all_args.other_agent_type
        self.use_same_other_policy = all_args.use_same_other_policy
        self.other_agent_policy_path = all_args.other_agent_policy_path

        # task parameters
        self.scenario_name = all_args.scenario_name
        self.horizon = all_args.horizon
        self.vehicles_count = all_args.vehicles_count
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
            "vehicles_count": self.vehicles_count,
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
        if self.dummy_agent_type in ["vi", "rvi"]:
            if self.dummy_agent_type == "vi":
                from .agents.dynamic_programming.value_iteration import ValueIterationAgent as DummyAgent
            else:
                from .agents.dynamic_programming.robust_value_iteration import RobustValueIterationAgent as DummyAgent
            agent_config = {"env_preprocessors": [{"method":"simplify"}], "budget": 50}
        elif self.dummy_agent_type == "mcts":
            from .agents.tree_search.mcts import MCTSAgent as DummyAgent 
            agent_config = {"max_depth": 1, "budget": 200, "temperature": 200}  
        elif self.dummy_agent_type == "d3qn":
            from .agents.policy_pool.dqn.policy import actor as DummyAgent
            agent_config = {"hidden_size": [256, 128]}
        else:
            raise NotImplementedError

        if self.dummy_agent_type == "d3qn":
            if self.use_same_dummy_policy:
                self.dummies = DummyAgent(self.all_args,
                                self.all_observation_space[self.n_attackers + self.n_defenders],
                                self.all_action_space[self.n_attackers + self.n_defenders],
                                hidden_size = agent_config['hidden_size']) # re-structure this!
                policy_state_dict = torch.load(self.dummy_agent_policy_path, map_location='cpu')
                self.dummies.load_state_dict(policy_state_dict)
                self.dummies.eval()
            else:
                # TODO: need to support different models in the future. 
                self.dummies = []
                for dummy_id in range(self.n_dummies):
                    dummy = DummyAgent(self.all_args,
                                self.all_observation_space[dummy_id + self.n_attackers + self.n_defenders],
                                self.all_action_space[dummy_id + self.n_attackers + self.n_defenders],
                                hidden_size = agent_config['hidden_size']) # re-structure this!
                    policy_state_dict = torch.load(self.dummy_agent_policy_path, map_location='cpu')
                    dummy.load_state_dict(policy_state_dict)
                    dummy.eval()
                    self.dummies.append(dummy)
        else:
            self.dummies = []
            for dummy_id in range(self.n_dummies):
                dummy = DummyAgent(env = self.env_init, 
                                config = agent_config,                
                                vehicle_id = dummy_id + self.n_attackers + self.n_defenders)
                self.dummies.append(dummy)

    def load_other_agents(self):
        """
            Load trained agent which serves as the defender/attacker based on type of the task 
        """
        if self.other_agent_type == "d3qn":
            from .agents.policy_pool.dqn.policy import actor as Policy
            agent_config = {"hidden_size": [256, 128]}
        elif self.other_agent_type == "ppo":
            from .agents.policy_pool.ppo.policy import actor as Policy
            agent_config = {"hidden_size": 64}
        else:
            raise NotImplementedError
        
        if self.use_same_other_policy:
            self.other_agents = Policy(self.all_args,
                                self.all_observation_space[self.load_start_idx],
                                self.all_action_space[self.load_start_idx],
                                hidden_size = agent_config['hidden_size'], # re-structure this!
                                use_recurrent_policy = self.all_args.use_recurrent_policy) # cpu is fine actually, keep it for now.
            policy_state_dict = torch.load(self.other_agent_policy_path, map_location='cpu')
            self.other_agents.load_state_dict(policy_state_dict)
            self.other_agents.eval()
        else:
            # TODO: need to support different models in the future. 
            self.other_agents = []           
            for agent_id in range(self.n_other_agents):
                policy = Policy(self.all_args,
                                self.all_observation_space[self.load_start_idx + agent_id],
                                self.all_action_space[self.load_start_idx + agent_id],
                                hidden_size = agent_config['hidden_size'], # re-structure this!
                                use_recurrent_policy = self.all_args.use_recurrent_policy) # cpu is fine actually, keep it for now.
                policy_state_dict = torch.load(self.other_agent_policy_path, map_location='cpu') # ! should be a list or other ways in this case
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
                if self.use_same_dummy_policy:
                    dummy_actions = self.dummies.act(self.dummy_obs)
                else:
                    dummy_actions = []
                    for dummy_id in range(self.n_dummies):
                        dummy_action = self.dummies[dummy_id].act(self.dummy_obs[dummy_id])
                        if type(dummy_action) == int:
                            dummy_action = [dummy_action]
                        else:
                            if len(dummy_action.shape) > 1:
                                dummy_action = dummy_action.squeeze(-1) 
                            elif len(dummy_action.shape) < 1:  
                                dummy_action = [dummy_action]
                            else:
                                pass                    
                        dummy_actions.append(dummy_action)
                action = np.concatenate([action, dummy_actions])

            # for discrete action, drop the unneeded axis
            action = np.squeeze(action, axis=-1)

            all_obs, all_rewards, all_dones, infos = self.env.step(tuple(action))

            if self.current_step == 1:
                self.init_position = [deepcopy(infos)["position"][agent_id] for agent_id in range(self.n_defenders + self.n_attackers)]

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

            # ! @zhuo u need to use this one!
            # 1. train dones
            dones = [all_dones[self.train_start_idx + agent_id] for agent_id in range(self.n_agents)]
            # 2. other dones
            other_dones = [all_dones[self.load_start_idx + agent_id] for agent_id in range(self.n_other_agents)]
            # 3. dummy dones
            dummy_dones = [all_dones[self.n_attackers + self.n_defenders + dummy_id] for dummy_id in range(self.n_dummies)]

            # update info
            speeds = [infos["speed"][agent_id] for agent_id in range(self.n_defenders + self.n_attackers)]
            self.episode_speeds.append(speeds)
            for i, s in enumerate(np.mean(self.episode_speeds, axis=0)):
                if i < self.n_defenders:
                    infos.update({"defender_{}_speed".format(i): s})
                else:
                    infos.update({"attacker_{}_speed".format(i): s})

            if np.all(dones):
                crashes = [infos["crashed"][agent_id] for agent_id in range(self.n_defenders + self.n_attackers)]
                position = [infos["position"][agent_id] for agent_id in range(self.n_defenders + self.n_attackers)]
                
                for i, c in enumerate(crashes):
                    if i < self.n_defenders:
                        infos.update({"defender_{}_crash".format(i): float(c)})
                    else:
                        infos.update({"attacker_{}_crash".format(i): float(c)})

                for i, pos in enumerate(position):
                    dis = np.linalg.norm(pos - self.init_position[i])
                    if i < self.n_defenders:
                        infos.update({"defender_{}_distance".format(i): dis})
                    else:
                        infos.update({"attacker_{}_distance".format(i): dis})

                infos.update({"episode_length": self.current_step})

            self.current_step += 1

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

            self.other_obs = np.array([np.concatenate(all_obs[self.load_start_idx + agent_id]) \
                                    for agent_id in range(self.n_other_agents)])
            self.masks = np.ones((self.n_other_agents, 1), dtype=np.float32)

            # deal with agents that need to train
            obs = np.array([np.concatenate(all_obs[self.train_start_idx + agent_id]) for agent_id in range(self.n_agents)])

            self.current_step += 1

            if self.use_render_vulnerability:
                self.cache_frames.append(self.render('rgb_array'))

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
