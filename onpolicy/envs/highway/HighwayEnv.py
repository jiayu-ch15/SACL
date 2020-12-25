env_dict={
    "id": "highway-v0",
    "import_module": "highway_env",
    "controlled_vehicles": 3,
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

import gym
from rl_agents.agents.common.factory import load_environment
import numpy as np
class highway_(gym.core.Wrapper):
    def __init__(self,env,all_args):
        super().__init__(env)
        self.n_agents = all_args.num_agents

        self.obs_dim=list(self.observation_space)[0].shape[0]*list(self.observation_space)[0].shape[1]
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10,
                                                 shape=(self.obs_dim,))] * self.n_agents
        self.action_space=list(self.action_space)
        self.share_observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(self.obs_dim,))] * self.n_agents

    def step(self, action):
        obs,rew,done,info=self.env.step(tuple(action))
        ob = []
        for i in range(self.n_agents):
            ob.append(np.concatenate(obs[i]))

        return ob,[[rew]]*self.n_agents,[done]*self.n_agents,info

    def reset(self):
        obs=self.env.reset()
        ob=[]
        for i in range(self.n_agents):
            ob.append(np.concatenate(obs[i]))

        return ob

def highway(all_args):

    env=load_environment(env_dict)

    return highway_(env,all_args)