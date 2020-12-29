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
        self.n_npc=all_args.n_npc
        self.use_centralized_V = all_args.use_centralized_V
        self.env_dict={
                    "id": all_args.scenario_name,
                    "import_module": "onpolicy.envs.highway.highway_env",
                    "controlled_vehicles": self.n_agents+ self.n_attacker+self.n_npc,

                    "duration": self.all_args.episode_length,
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

        self.new_observation_space = []
        self.share_observation_space = []
        for agent_id in range(self.n_agents):
            obs_shape = list(self.observation_space[agent_id].shape)
            self.obs_dim = reduce(lambda x, y: x*y, obs_shape)
            self.share_obs_dim = self.obs_dim * self.n_agents if self.use_centralized_V else self.obs_dim
            self.new_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,)))
            self.share_observation_space.append(gym.spaces.Box(low=-1e10, high=1e10, shape=(self.share_obs_dim,)))

        self.observation_space = self.new_observation_space
        self.action_space=list(self.action_space)[:self.n_agents]

        self.load_rl_agents()
        if self.n_npc>0:
            self.load_npc_agents()

        self.episodes_rew = 0
        self.episodes_rews=[]
        self.time_step = 0


    def load_npc_agents(self):
        from onpolicy.envs.highway.agents.tree_search.mcts import MCTSAgent
        self.npcs=[]
        for i in range(self.n_npc):
            self.npcs.append( MCTSAgent(self.env_init,id=i+self.n_attacker+self.n_agents,
                                        config=dict(budget=200, temperature=200, max_depth=1)))

    def load_rl_agents(self):
        all_arg = deepcopy(self.all_args)

        ###pay attention:all_arg is for attack_agents initialization
        ###to do: add if-else to redefine Policy,use_centralized_V,
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        share_observation_space = self.share_observation_space[0] if all_arg.use_centralized_V else \
        self.observation_space[0]

        all_arg.num_agents = self.n_attacker
        self.rl_agent = Policy(all_arg,
                               self.observation_space[0],
                               share_observation_space,
                               self.action_space[0],
                               device=self.device,
                               cat_self=False)
        policy_actor_state_dict = torch.load(all_arg.rl_agent_path, map_location=self.device)
        self.rl_agent.actor.load_state_dict(policy_actor_state_dict)


    def step(self, action):
        #### attack_agents action
        attack_actions,_, self.rnn_states = self.rl_agent.actor(self.obs_attack, self.rnn_state, self.masks, deterministic=False)
        action=np.concatenate([action,attack_actions])
        self.render()
        #####npc action
        if self.n_npc>0:
            npc_action=[]
            for npc in self.npcs:
                a_npc = npc.act(self.o_npc)
                npc_action.append([a_npc])
            action = np.concatenate([action, npc_action])
        ####for discrete action!
        action = np.squeeze(action, axis=-1)

        o, r, d, infos = self.env.step(tuple(action))
        print(r)
        self.render()
        self.o_npc=o
        obs = [np.concatenate(o[i]) for i in range(self.n_agents)]
        rewards = [[r[i]] for i in range(self.n_agents)]
        dones = [d for i in range(self.n_agents)]

        ### attack_agents obs
        o = [np.concatenate(o[i]) for i in range(len(o))]
        self.obs_attack = torch.tensor(o[self.n_agents:self.n_agents+self.n_attacker]).to(self.device)


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
        self.o_npc = o
        obs = [np.concatenate(o[i]) for i in range(self.n_agents)]

        ### init attack_agent obs
        self.rnn_state = torch.zeros((self.n_attacker, self.all_args.hidden_size)).to(self.device)
        o = [np.concatenate(o[i]) for i in range(len(o))]
        self.obs_attack = torch.tensor(o[self.n_agents:self.n_agents+self.n_attacker]).to(self.device)
        self.masks=torch.ones((self.n_attacker,1)).to(self.device)

        return obs
