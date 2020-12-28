import gym
import numpy as np
from functools import reduce
import torch
from onpolicy.envs.highway.common.factory import load_environment
from copy import deepcopy


class HighwayEnv(gym.core.Wrapper):
    def __init__(self, all_args, device):
        self.all_args = all_args
        self.use_centralized_V = all_args.use_centralized_V
        self.device = device
        self.task_type = all_args.task_type

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
                    "id": all_args.scenario_name,
                    "import_module": "onpolicy.envs.highway.highway_env",
                    # u must keep this order!!! can not change that!!!
                    "controlled_vehicles": self.n_defenders + self.n_attackers + self.n_dummies,

                    "duration": self.all_args.horizon,
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
                    "vehicles_count": 50,
        }

        self.env_init = load_environment(self.env_dict)

        super().__init__(self.env_init)

        # here we load other agents and dummies, can not change the order of the following code!!
        self.load_other_agents()
        self.load_dummies()

        # get new obs and action space
        self.new_observation_space = []
        self.new_action_space = []
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

    def load_dummies(self):
        from .agents.tree_search.mcts import MCTSAgent as DummyAgent
        self.dummies = []
        for dummy_id in range(self.n_dummies):
            self.dummies.append(DummyAgent(self.env_init, 
                                            id = dummy_id + self.n_attackers + self.n_defenders,
                                            config=dict(budget=200, temperature=200, max_depth=1)))

    def load_other_agents(self):
        from .agents.policy_pool.policy import Policy
        # TODO: need to support different models in the future.
        policy_path = self.all_args.policy_path
        self.other_agents = []
         
        for agent_id in range(self.n_other_agents):
            policy = Policy(self.observation_space[self.load_start_idx + agent_id],
                            self.action_space[self.load_start_idx + agent_id],
                            hidden_size = self.all_args.hidden_size, # re-structure this!
                            use_recurrent_policy = self.all_args.use_recurrent_policy, #  re-structure this!
                            device=self.device) # cpu is fine actually, keep it for now.
            policy_state_dict = torch.load(policy_path, map_location=self.device)
            policy.load_state_dict(policy_state_dict)
            self.other_agents.append(policy)


    def step(self, action):
        # we need to get actions of other agents
        if self.n_other_agents > 0:
            other_actions = []
            for agent_id in range(self.n_other_agents):
                other_action, rnn_state = self.other_agents[agent_id](self.other_obs[agent_id], 
                                                                        self.rnn_states[agent_id], 
                                                                        self.masks[agent_id], 
                                                                        deterministic=True)
                other_actions.append(other_action.detach().cpu().numpy())
                self.rnn_states[agent_id] = rnn_state.detach().cpu().numpy()
            
            if self.train_start_idx == 0:
                action = np.concatenate([action, other_actions])
            else:
                action = np.concatenate([other_actions, action])

        # then we need to get actions of dummies
        if self.n_dummies > 0:
            dummy_actions = []
            for dummy in self.dummies:
                dummy_action = dummy.act(self.dummy_obs)
                dummy_actions.append([dummy_action])
            action = np.concatenate([action, dummy_actions])
        
        # for discrete action, drop the unneeded axis
        action = np.squeeze(action, axis=-1)

        all_obs, all_rewards, all_dones, infos = self.env.step(tuple(action))

        # obs
        # 1. train obs
        obs = [np.concatenate(all_obs[self.train_start_idx + agent_id]) for agent_id in range(self.n_agents)]
        # 2. other obs
        self.other_obs = [np.concatenate(all_obs[self.load_start_idx: self.load_start_idx + agent_id]) \
                                for agent_id in range(self.n_other_agents)]
        # 3. dummy obs
        self.dummy_obs = all_obs
       
        # rewards
        # ! @zhuo if agent is dead, rewards need to be set zero!
        # 1. train rewards
        rewards = [[all_rewards[self.train_start_idx + agent_id]] for agent_id in range(self.n_agents)]
        self.episode_rewards.append(rewards)
        # 2. other rewards
        other_rewards = [[all_rewards[self.load_start_idx: self.load_start_idx + agent_id]] \
                                for agent_id in range(self.n_other_agents)]
        self.episode_other_rewards.append(other_rewards)
        # 3. dummy rewards
        dummy_rewards = [[all_rewards[self.n_attackers + self.n_defenders: self.n_attackers + self.n_defenders + dummy_id]] \
                                for dummy_id in range(self.n_dummies)]
        self.episode_dummy_rewards.append(dummy_rewards)

        # ! @zhuo u need to use this one!
        # 1. train dones
        dones = [all_dones[self.train_start_idx + agent_id] for agent_id in range(self.n_agents)]
        # 2. other dones
        other_dones = [all_dones[self.load_start_idx: self.load_start_idx + agent_id] for agent_id in range(self.n_other_agents)]
        # 3. dummy dones
        dummy_dones = [all_dones[self.n_attackers + self.n_defenders: self.n_attackers + self.n_defenders + dummy_id] for dummy_id in range(self.n_dummies)]
        
        # wrong way!
        # dones = [all_dones for agent_id in range(self.n_agents)]

        # if np.all(dones):
        #     self.episodes_rew += rewards[0][0]
        #     self.episodes_rews.append(self.episodes_rew)
        #     self.episodes_rew = 0
        # else:
        #     self.episodes_rew += rewards[0][0]

        #     self.epi_average_rew = np.mean(np.array(self.episodes_rews))

        infos.update({"episode_rewards": np.sum(self.episode_rewards, axis=0)}, 
                    {"episode_other_rewards": np.sum(self.episode_other_rewards, axis=0)},
                    {"episode_dummy_rewards": np.sum(self.episode_dummy_rewards, axis=0)}
                    )

        return obs, rewards, dones, infos

    def reset(self):

        self.episode_rewards = []
        self.episode_dummy_rewards = []
        self.episode_other_rewards = []

        all_obs = self.env.reset()

        # dummy needs to take all obs
        self.dummy_obs = all_obs 

        # deal with other agents
        self.rnn_states = np.zeros((self.n_other_agents, self.all_args.hidden_size), dtype=np.float32)
        # o = [np.concatenate(all_obs[i]) for i in range(len(all_obs))]
        self.other_obs = [np.concatenate(all_obs[self.load_start_idx: self.load_start_idx + agent_id]) \
                                for agent_id in range(self.n_other_agents)]
        self.masks = np.ones((self.n_other_agents, 1), dtype=np.float32)

        # deal with agents that need to train
        obs = [np.concatenate(all_obs[self.train_start_idx + agent_id]) for agent_id in range(self.n_agents)]

        return obs
