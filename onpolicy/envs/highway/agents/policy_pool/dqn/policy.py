from onpolicy.envs.highway.agents.common.models import model_factory
import numpy as np
import torch
from torch import nn
from onpolicy.algorithms.utils.util import check

class actor(nn.Module):
    def __init__(self, args=None, obs_space=None, action_space=None, hidden_size=None, use_recurrent_policy=None):
        super(actor, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.tpdv = dict(dtype=torch.float32)
        if not isinstance(hidden_size, list):
            hidden_size = [256, 128]
        self.config = {
            "model": {
                "type": "DuelingNetwork",
                "base_module": {
                    "layers": hidden_size
                },
                "value": {
                    "layers": [hidden_size[1]]
                },
                "advantage": {
                    "layers": [hidden_size[1]]
                },
                "in": int(np.prod(self.obs_space.shape)),
                "out": self.action_space.n
            },
        }
        self.value_net = model_factory(self.config["model"])

    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):   
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        obs = check(obs).to(self.tpdv)
        values = self.value_net(obs)
        if deterministic:
            actions = torch.argmax(values)
        else:
            print("only support greedy action while evaluating!")
            raise NotImplementedError

        return actions, rnn_states

    def act(self, obs):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        
        obs = check(obs).to(self.tpdv)
        values = self.value_net(obs).detach().numpy()
        actions = np.argmax(values)

        return actions

    def load_state_dict(self, policy_state_dict):
        self.value_net.load_state_dict(policy_state_dict['state_dict'])