import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.cnn import CNNLayer
from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.util import init, check

from onpolicy.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

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
        self._recurrent_N = args.recurrent_N
        self._attn_size = args.attn_size
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_centralized_V = args.use_centralized_V
        self._use_conv1d = args.use_conv1d
        self._stacked_frames = args.stacked_frames
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # obs space
        obs_shape = get_shape_from_obs_space(obs_space)

        if len(obs_shape) == 3:
            self.obs_prep = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal)
        else:
            obs_dim = obs_shape[0]

            # feature norm
            if self._use_feature_popart:
                self.obs_feature_norm = PopArt(obs_dim)

            if self._use_feature_normalization:
                self.obs_feature_norm = nn.LayerNorm(obs_dim)
            
            # attn model
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

            # conv1d model
            if self._use_conv1d:
                self.obs_conv = CONVLayer(
                        self._stacked_frames, self.hidden_size, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)
                random_x = torch.FloatTensor(1,self._stacked_frames,inputs_dim)
                random_out = self.obs_conv(random_x)
                assert len(random_out.shape)==3
                inputs_dim = random_out.size(-1) * random_out.size(-2)
            
            # fc model
            self.obs_prep = MLPLayer(inputs_dim, self.hidden_size, layer_N=0,
                                    use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)
                
        # share obs space
        if self._use_centralized_V:
            share_obs_shape = get_shape_from_obs_space(share_obs_space)

            if len(share_obs_shape) == 3:
                self.share_obs_prep = CNNLayer(share_obs_shape, self.hidden_size, self._use_orthogonal)
            else:
                share_obs_dim = share_obs_shape[0]
                # feature norm
                if self._use_feature_popart:
                    self.share_obs_feature_norm = PopArt(share_obs_dim)

                if self._use_feature_normalization:
                    self.share_obs_feature_norm = nn.LayerNorm(share_obs_dim)

                # attn model
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

                # conv1d model
                if self._use_conv1d:
                    self.share_obs_conv = CONVLayer(self._stacked_frames, self.hidden_size, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)
                
                    random_x = torch.FloatTensor(1,self.stacked_frames,inputs_dim)
                    random_out = self.share_obs_conv(random_x)
                    assert len(random_out.shape)==3
                    inputs_dim = random_out.size(-1) * random_out.size(-2)
                
                # fc model
                self.share_obs_prep = MLPLayer(inputs_dim, self.hidden_size, layer_N=0, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)
        else:
            if self._use_feature_popart:
                self.share_obs_feature_norm = self.obs_feature_norm

            if self._use_feature_normalization:
                self.share_obs_feature_norm = self.obs_feature_norm

            if self._use_attn:
                self.share_obs_attn = self.obs_attn
                self.share_obs_attn_norm = self.obs_attn_norm
            
            if self._use_conv1d:
                self.share_obs_conv = self.obs_conv
            
            self.share_obs_prep = self.obs_prep

        # common layer
        self.common = MLPLayer(self.hidden_size, self.hidden_size, layer_N=0, use_orthogonal=self._use_orthogonal, use_ReLU=self._use_ReLU)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        # value
        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        # action
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(self.device)

    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        x = obs

        if self._use_feature_popart or self._use_feature_normalization:
            x = self.obs_feature_norm(x)

        if self._use_attn:
            x = self.obs_attn(x, self_idx=-1)
            x = self.obs_attn_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.obs_conv(x)
            x = x.view(batch_size, -1)
  
        x = self.obs_prep(x)
        # common
        actor_features = self.common(x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        x = obs

        if self._use_feature_popart or self._use_feature_normalization:
            x = self.obs_feature_norm(x)

        if self._use_attn:
            x = self.obs_attn(x, self_idx=-1)
            x = self.obs_attn_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.obs_conv(x)
            x = x.view(batch_size, -1)
  
        x = self.obs_prep(x)

        actor_features = self.common(x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)
       
        return action_log_probs, dist_entropy

    def get_values(self, share_obs, rnn_states, masks):
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        share_x = share_obs

        if self._use_feature_popart or self._use_feature_normalization:
            share_x = self.share_obs_feature_norm(share_x)

        if self._use_attn:
            share_x = self.share_obs_attn(share_x, self_idx=-1)
            share_x = self.share_obs_attn_norm(share_x)

        if self._use_conv1d:
            batch_size = share_x.size(0)
            share_x = share_x.view(batch_size, self._stacked_frames, -1)
            share_x = self.share_obs_conv(share_x)
            share_x = share_x.view(batch_size, -1)

        share_x = self.share_obs_prep(share_x)

        critic_features = self.common(share_x)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
