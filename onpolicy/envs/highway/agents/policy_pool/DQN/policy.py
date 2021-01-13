from onpolicy.envs.highway.agents.common.models import model_factory, trainable_parameters
from onpolicy.envs.highway.agents.common.utils import choose_device
from gym import spaces
import numpy as np
import torch
from torch import nn

class dqn_actor(nn.Module):
    def __init__(self, args=None, obs_space=None, action_space=None, hidden_size=None, use_recurrent_policy=None):
        super(dqn_actor, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
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
            "gamma": 0.8
        }
        self.device = choose_device(self.config.get("device", "cuda:0"))
        self.value_net = model_factory(self.config["model"])
        self.value_net.to(self.config.get("device", "cuda:0"))

    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):   
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return torch.Tensor([[self.act(obs)]]), obs

    def act(self, state, step_exploration_time=True):
        """
            Act according to the state-action value model and an exploration policy
        :param state: current state
        :param step_exploration_time: step the exploration schedule
        :return: an action
        """
        self.previous_state = state
        # Handle multi-agent observations
        # TODO: it would be more efficient to forward a batch of states
        if isinstance(state, tuple):
            return tuple(self.act(agent_state, step_exploration_time=False) for agent_state in state)

        # Single-agent setting
        values =  self.value_net(torch.tensor([state], dtype=torch.float).to(self.device)).data.cpu().numpy()[0]
        return np.argmax(values)


    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        return filename

    def load_state_dict(self, policy_state_dict):
        self.value_net.load_state_dict(policy_state_dict['state_dict'])

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return [self.act(state)]


    def seed(self, seed):
        pass


    def reset(self):
        pass


    def eval(self):
        pass