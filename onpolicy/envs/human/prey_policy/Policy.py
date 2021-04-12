import numpy as np
import torch
from .Model import R_Actor

class PreyPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.device = device

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
