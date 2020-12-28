import gym
import numpy as np
from functools import reduce
import torch
from onpolicy.envs.highway.common.factory import load_environment
from copy import deepcopy
class HighwayEnv(gym.core.Wrapper):
    def __init__(self, all_args,device):
        self.all_args=all_args
        self.device=device
        self.n_agents = all_args.num_agents
        self.n_attacker=all_args.n_attacker
        self.use_centralized_V = all_args.use_centralized_V
        self.env_dict={
                    "id": all_args.scenario_name,
                    "import_module": "onpolicy.envs.highway.highway_env",
                    "controlled_vehicles": self.n_agents+all_args.n_attacker,

                    "duration": all_args.episode_length,

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
                    }
                }
        env = load_environment(self.env_dict)

        super().__init__(env)

        self.new_observation_space = []
        self.share_observation_space = []
        for agent_id in range(self.n_agents):
            obs_shape = list(self.observation_space[agent_id].shape)
            self.obs_dim = reduce(lambda x, y: x*y, obs_shape)
            self.share_obs_dim = self.obs_dim * self.n_agents if self.use_centralized_V else self.obs_dim
            self.new_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,)))
            self.share_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.share_obs_dim,)))
        
        self.observation_space = self.new_observation_space
        self.action_space=list(self.action_space)


        #####rl_agent!
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        share_observation_space = self.share_observation_space[0] if all_args.use_centralized_V else self.observation_space[0]
        all_arg=deepcopy(all_args)
        all_arg.num_agents=1
        self.rl_agent = Policy(all_arg,
                             self.observation_space[0],
                             share_observation_space,
                             self.action_space[0],
                             device=device,
                             cat_self=False)
        policy_actor_state_dict = torch.load(all_args.rl_agent_path,map_location=self.device)
        self.rl_agent.actor.load_state_dict(policy_actor_state_dict)

        self.episodes_rew = 0
        self.episodes_rews=[]
        self.time_step = 0

    def step(self, action):

        attack_actions,_, self.rnn_states = self.rl_agent.actor(self.obs, self.rnn_state, self.masks, deterministic=False)
        action=np.concatenate([action,attack_actions])

        ####for discrete action!!!
        action = np.squeeze(action, axis=-1)
        o, r, d, infos = self.env.step(tuple(action))

        obs = [np.concatenate(o[i]) for i in range(self.n_agents)]
        rewards = [[r[i]] for i in range(self.n_agents)]
        dones = [d for i in range(self.n_agents)]
        self.render()

        o = [np.concatenate(o[i]) for i in range(len(o))]
        self.obs = torch.tensor(o[self.n_agents:]).to(self.device)

        if np.all(dones):
            self.episodes_rew += rewards[0][0]
            self.episodes_rews.append(self.episodes_rew)
            self.episodes_rew = 0
        else:
            self.episodes_rew += rewards[0][0]

        self.time_step+=1
        if self.time_step ==self.all_args.episode_length:
            self.time_step=0
            self.epi_average_rew = np.mean(np.array(self.episodes_rews))
            infos.update({"mean_rew":self.epi_average_rew})

        return obs, rewards, dones, infos

    def reset(self):


        o = self.env.reset()
        obs = [np.concatenate(o[i]) for i in range(self.n_agents)]

        self.rnn_state = torch.zeros((self.n_attacker, self.all_args.hidden_size)).to(self.device)
        o = [np.concatenate(o[i]) for i in range(len(o))]
        self.obs = torch.tensor(o[self.n_agents:]).to(self.device)
        self.masks=torch.ones((self.n_attacker,1)).to(self.device)

        return obs
