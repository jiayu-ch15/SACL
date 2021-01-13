import torch
from gym import spaces

from onpolicy.envs.highway.agents.common.models import model_factory, size_model_config
from onpolicy.envs.highway.agents.common.utils import choose_device

import numpy as np

class DQNAgent():
    def __init__(self, env, config=None, vehicle_id = 0):
        self.env = env
        self.config = config
        size_model_config(self.env, self.config["model"])
        self.value_net = model_factory(self.config["model"])
    
        self.device = choose_device(self.config["device"])
        self.value_net.to(self.device)

        self.vehicle_id = vehicle_id

    def act(self, state):
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
            return tuple(self.act(agent_state) for agent_state in state)

        # Single-agent setting
        values =  self.value_net(torch.tensor([state], dtype=torch.float).to(self.device)).data.cpu().numpy()[0]
        return np.argmax(values)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return [self.act(state)]