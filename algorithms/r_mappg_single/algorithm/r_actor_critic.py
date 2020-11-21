import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.popart import PopArt
from utils.util import init
from utils.cnn import CNNBase
from utils.mlp import MLPLayer, MLPBase
from utils.rnn import RNNLayer


class R_Model(nn.Module):
    def __init__(self, args, obs_space, share_obs_space, action_space, device=torch.device("cpu"), cat_self=True):
        super(R_Model, self).__init__()
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._use_feature_normalization = args.use_feature_normalization
        self._use_feature_popart = args.use_feature_popart
        self._use_attn = args.use_attn
        self._use_average_pool = args.use_average_pool
        self._layer_N = args.layer_N
        self._attn_size = args.attn_size
        self.hidden_size = args.hidden_size
        self.mixed_action = False
        self.multi_discrete = False
        self.device = device

        # obs space
        if obs_space.__class__.__name__ == "Box":
            obs_shape = obs_space.shape
        elif obs_space.__class__.__name__ == "list":
            obs_shape = obs_space
        else:
            raise NotImplementedError

        if len(obs_shape) == 3:
            self.obs_prep = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal)
        else:
            obs_dim = obs_shape[0]
            if self._use_feature_popart:
                self.obs_feature_norm = PopArt(obs_dim)

            if self._use_feature_normalization:
                self.obs_feature_norm = nn.LayerNorm(obs_dim)
            
            if self._use_attn:
                if self._use_average_pool:
                    inputs_dim = self._attn_size + obs_shape[-1][1]
                else:
                    split_inputs_dim = 0
                    split_shape = obs_shape[1:]
                    for i in range(len(split_shape)):
                        split_inputs_dim += split_shape[i][0]
                    inputs_dim = split_inputs_dim * self._attn_size
                self.obs_attn = Encoder(args, obs_shape)
                self.obs_attn_norm = nn.LayerNorm(inputs_dim)
            else:
                inputs_dim = obs_dim
            self.obs_prep = MLPLayer(inputs_dim, self.hidden_size, layer_N=0, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)

        # share obs space
        if share_obs_space.__class__.__name__ == "Box":
            share_obs_shape = share_obs_space.shape
        elif share_obs_space.__class__.__name__ == "list":
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        if len(share_obs_shape) == 3:
            self.share_obs_prep = CNNLayer(share_obs_shape, self.hidden_size, self._use_orthogonal)
        else:
            share_obs_dim = share_obs_shape[0]
            if self._use_feature_popart:
                self.share_obs_feature_norm = PopArt(share_obs_dim)

            if self._use_feature_normalization:
                self.share_obs_feature_norm = nn.LayerNorm(share_obs_dim)
            
            if self._use_attn:
                if self._use_average_pool:
                    if cat_self:
                        inputs_dim = self._attn_size + share_obs_shape[-1][1]
                    else:
                        inputs_dim = self._attn_size
                else:
                    split_inputs_dim = 0
                    split_shape = obs_shape[1:]
                    for i in range(len(split_shape)):
                        split_inputs_dim += split_shape[i][0]
                    inputs_dim = split_inputs_dim * self._attn_size
                self.share_obs_attn = Encoder(args, share_obs_shape, cat_self)
                self.share_obs_attn_norm = nn.LayerNorm(inputs_dim)
            else:
                inputs_dim = share_obs_dim
            self.share_obs_prep = MLPLayer(inputs_dim, self.hidden_size, layer_N=0, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)

        # common layer
        self.common = MLPLayer(self.hidden_size, self.hidden_size, layer_N=0, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)
        self.rnn = RNNLayer(args, self.hidden_size, self.hidden_size)

        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0))
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0))
        # value 
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        
        # action
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

    def get_actions(self, obs, rnn_hidden_states, masks, available_actions=None, deterministic=False):
        obs = obs.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)
        masks = masks.to(self.device)
        if available_actions is not None:
            available_actions = available_actions.to(self.device)

        x = obs
        
        if self._use_feature_popart or self._use_feature_normalization:
            x = self.obs_feature_norm(x)
            
        if self._use_attn:
            x = self.obs_attn(x, self_idx=-1)
            x = self.obs_attn_norm(x)

        x = self.obs_prep(x)
        # common
        rnn_inp = self.common(x)
        actor_features, rnn_hidden_states = self.rnn(rnn_inp, rnn_hidden_states, masks)

        if self.mixed_action:
            action_outs, actions, action_log_probs = [None, None], [None, None], [None, None]
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

        x = obs

        if self._use_feature_popart or self._use_feature_normalization:
            x = self.obs_feature_norm(x)

        if self._use_attn:
            x = self.obs_attn(x, self_idx=-1)
            x = self.obs_attn_norm(x)

        x = self.obs_prep(x)
        rnn_inp = self.common(x)
        actor_features, rnn_hidden_states = self.rnn(
            rnn_inp, rnn_hidden_states, masks)

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

    def get_value(self, share_obs, rnn_hidden_states, masks):
        share_obs = share_obs.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)
        masks = masks.to(self.device)

        share_x = share_obs

        if self._use_feature_popart or self._use_feature_normalization:
            share_x = self.share_obs_feature_norm(share_x)

        if self._use_attn:
            share_x = self.share_obs_attn(share_x, self_idx=-1)
            share_x = self.share_obs_attn_norm(share_x)

        share_x = self.share_obs_prep(share_x)
        rnn_inp = self.common(share_x)
        critic_features, rnn_hidden_states = self.rnn(
            rnn_inp, rnn_hidden_states, masks)
        value = self.v_out(critic_features)

        return value, rnn_hidden_states
