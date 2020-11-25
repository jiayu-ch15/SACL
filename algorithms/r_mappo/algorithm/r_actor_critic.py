import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.popart import PopArt
from utils.util import init
from utils.cnn import CNNBase
from utils.mlp import MLPBase
from utils.conv import CONVBase
from utils.rnn import RNNLayer


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._naive_recurrent_policy = args.naive_recurrent_policy
        self._recurrent_policy = args.recurrent_policy
        self.mixed_action = False
        self.multi_discrete = False
        self.device = device

        if obs_space.__class__.__name__ == "Box":
            obs_shape = obs_space.shape
        elif obs_space.__class__.__name__ == "list":
            obs_shape = obs_space
        else:
            raise NotImplementedError

        if len(obs_shape) == 3:
            self.base = CNNBase(args, obs_shape)
        else:
            self.base = MLPBase(args, obs_shape)

        if self._naive_recurrent_policy or self._recurrent_policy:
            self.rnn = RNNLayer(args, self.hidden_size, self.hidden_size)

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(
                self.hidden_size, action_dim, self._use_orthogonal, self._gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                self.hidden_size, action_dim, self._use_orthogonal, self._gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(
                self.hidden_size, action_dim, self._use_orthogonal, self._gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            self.discrete_N = action_space.shape
            action_size = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_size:
                self.action_outs.append(Categorical(
                    self.hidden_size, action_dim, self._use_orthogonal, self._gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(self.hidden_size, continous_dim, self._use_orthogonal, self._gain), Categorical(
                self.hidden_size, discrete_dim, self._use_orthogonal, self._gain)])

    def forward(self, obs, rnn_hidden_states, masks, available_actions=None, deterministic=False):
        obs = obs.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)
        masks = masks.to(self.device)
        if available_actions is not None:
            available_actions = available_actions.to(self.device)

        actor_features = self.base(obs)

        if self._naive_recurrent_policy or self._recurrent_policy:
            actor_features, rnn_hidden_states = self.rnn(
                actor_features, rnn_hidden_states, masks)

        if self.mixed_action:
            action_outs, actions, action_log_probs = [
                None, None], [None, None], [None, None]
            for i in range(2):
                action_outs[i] = self.action_outs[i](actor_features)

                if deterministic:
                    actions[i] = action_outs[i].mode().float()
                else:
                    actions[i] = action_outs[i].sample().float()

                action_log_probs[i] = action_outs[i].log_probs(actions[i])

            action_out = torch.cat(actions, -1)
            action_log_probs_out = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for i in range(self.discrete_N):
                action_out = self.action_outs[i](actor_features)

                if deterministic:
                    action = action_out.mode()
                else:
                    action = action_out.sample()

                action_log_prob = action_out.log_probs(action)

                actions.append(action)
                action_log_probs.append(action_log_prob)

            action_out = torch.cat(actions, -1)
            action_log_probs_out = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True)

        else:
            action_out = self.action_out(actor_features, available_actions)

            if deterministic:
                action = action_out.mode()
            else:
                action = action_out.sample()

            action_log_probs = action_out.log_probs(action)

            action_out = action
            action_log_probs_out = action_log_probs

        return action_out, action_log_probs_out, rnn_hidden_states

    def evaluate_actions(self, obs, rnn_hidden_states, action, masks, active_masks=None):
        obs = obs.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)
        masks = masks.to(self.device)
        if active_masks is not None:
            active_masks = active_masks.to(self.device)
        action = action.to(self.device)

        actor_features = self.base(obs)
        if self._naive_recurrent_policy or self._recurrent_policy:
            actor_features, rnn_hidden_states = self.rnn(
                actor_features, rnn_hidden_states, masks)

        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_outs, action_log_probs, dist_entropy = [
                None, None], [None, None], [None, None]
            for i in range(2):
                action_outs[i] = self.action_outs[i](actor_features)
                action_log_probs[i] = action_outs[i].log_probs(action[i])
                if active_masks is not None:
                    if len(action_outs[i].entropy().shape) == len(active_masks.shape):
                        dist_entropy[i] = (
                            action_outs[i].entropy() * active_masks).sum()/active_masks.sum()
                    else:
                        dist_entropy[i] = (
                            action_outs[i].entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy[i] = action_outs[i].entropy().mean()
            action_log_probs_out = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy_out = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for i in range(self.discrete_N):
                action_out = self.action_outs[i](actor_features)
                action_log_probs.append(action_out.log_probs(action[i]))
                if active_masks is not None:
                    dist_entropy.append(
                        (action_out.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_out.entropy().mean())

            action_log_probs_out = torch.sum(
                torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy_out = torch.tensor(dist_entropy).mean()

        else:
            action_out = self.action_out(actor_features)
            action_log_probs = action_out.log_probs(action)
            if active_masks is not None:
                dist_entropy = (
                    action_out.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_out.entropy().mean()
            action_log_probs_out = action_log_probs
            dist_entropy_out = dist_entropy

        return action_log_probs_out, dist_entropy_out, rnn_hidden_states


class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu"), cat_self=True):
        super(R_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._naive_recurrent_policy = args.naive_recurrent_policy
        self._recurrent_policy = args.recurrent_policy
        self.device = device

        if share_obs_space.__class__.__name__ == "Box":
            share_obs_shape = share_obs_space.shape
        elif share_obs_space.__class__.__name__ == "list":
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        if len(share_obs_shape) == 3:
            self.base = CNNBase(args, share_obs_shape)
        else:
            self.base = MLPBase(args, share_obs_shape, cat_self)

        if self._naive_recurrent_policy or self._recurrent_policy:
            self.rnn = RNNLayer(args, self.hidden_size, self.hidden_size)

        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0))
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

    def forward(self, share_obs, rnn_hidden_states, masks):

        share_obs = share_obs.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)
        masks = masks.to(self.device)

        critic_features = self.base(share_obs)
        if self._naive_recurrent_policy or self._recurrent_policy:
            critic_features, rnn_hidden_states = self.rnn(
                critic_features, rnn_hidden_states, masks)
        value = self.v_out(critic_features)

        return value, rnn_hidden_states
