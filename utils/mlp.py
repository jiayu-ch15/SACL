import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import init, get_clones
from utils.attention import Encoder

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        if use_orthogonal:
            if use_ReLU:
                active_func = nn.ReLU()

                def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()

                def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('tanh'))
        else:
            if use_ReLU:
                active_func = nn.ReLU()

                def init_(m): return init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()
                def init_(m): return init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('tanh'))

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_feature_popart = args.use_feature_popart
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._use_attn = args.use_attn
        self._use_average_pool = args.use_average_pool
        self._layer_N = args.layer_N
        self._attn_size = args.attn_size
        self.hidden_size = args.hidden_size

        assert (self._use_feature_normalization and self._use_feature_popart) == False, (
            "--use_feature_normalization and --use_feature_popart can not be set True simultaneously.")

        obs_dim = obs_shape[0]

        if self._use_feature_popart:
            self.feature_norm = PopArt(obs_dim)

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        if self._use_attn:
            if self._use_average_pool:
                if cat_self:
                    inputs_dim = self._attn_size + obs_shape[-1][1]
                else:
                    inputs_dim = self._attn_size
            else:
                split_inputs_dim = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    split_inputs_dim += split_shape[i][0]
                inputs_dim = split_inputs_dim * self._attn_size
            self.attn = Encoder(args, obs_shape, cat_self)
            self.attn_norm = nn.LayerNorm(inputs_dim)
        else:
            inputs_dim = obs_dim

        self.mlp = MLPLayer(inputs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_popart or self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_attn:
            x = self.attn(x, self_idx=-1)
            x = self.attn_norm(x)

        x = self.mlp(x)

        return x