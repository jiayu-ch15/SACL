import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MIXBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None):
        super(MIXBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size
        self.cnn_keys = []
        self.mlp_keys = []
        self.n_cnn_input = 0
        self.n_mlp_input = 0

        for sensor in obs_shape:
            sensor_obs_shape = obs_shape[sensor]
            if sensor_obs_shape.__class__.__name__ == 'Box':
                if len(sensor_obs_shape) == 3:
                    self.cnn_keys.append(sensor)
                else:
                    self.mlp_keys.append(sensor)
            else:
                raise NotImplementedError

        if len(self.cnn_keys) > 0:
            self.cnn = self._build_cnn_model(obs_shape, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._use_ReLU)
        if len(self.mlp_keys) > 0:
            self.mlp = self._build_mlp_model(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        out_x = x
        if len(self.cnn_keys) > 0:
            cnn_input = self._build_cnn_input(x)
            cnn_x = self.cnn(cnn_input)
            out_x = cnn_x

        if len(self.mlp_keys) > 0:
            mlp_input = self._build_mlp_input(x)
            mlp_x = self.mlp(mlp_input)
            out_x = torch.cat([out_x, mlp_x], dim=1)

        return out_x

    def _build_cnn_model(self, obs_shape, cnn_layers_params, hidden_size, use_orthogonal, use_ReLU):
        if cnn_layers_params is None:
            cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        else:
            cnn_layers_params = cnn_layers_params

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for key in self.cnn_keys:
            if key in ['rgb','depth']:
                self.n_cnn_input += obs_shape[key].shape[2] 
                cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
            elif key in ['global_map','local_map']:
                self.n_cnn_input += obs_shape[key].shape[0] 
                cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
            else:
                raise NotImplementedError

        cnn_layers = []
        prev_out_channels = None
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if i == 0:
                in_channels = self.n_cnn_input
            else:
                in_channels = prev_out_channels
            cnn_layers.append(init_(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
            if i != len(cnn_layers_params) - 1:
                cnn_layers.append(active_func)
            prev_out_channels = out_channels
        
        for _, kernel_size, stride, padding in cnn_layers_params:
            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )
        
        cnn_layers += [
            Flatten(),
            init_(nn.Linear(cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                        hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        ]
        return nn.Sequential(*cnn_layers)

    def _build_mlp_model(self, obs_shape, hidden_size, use_orthogonal, use_ReLU):

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for key in self.mlp_keys:
            self.n_mlp_input += obs_shape[key].shape[0]

        return nn.Sequential(
            init_(nn.Linear(self.n_mlp_input, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        
    def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    def _build_cnn_input(self, obs):
        cnn_input = []

        for key in self.cnn_keys:
            if key in ['rgb','depth']:
                cnn_input.append(obs[key].permute(0, 3, 1, 2) / 255.0)
            elif key in ['global_map','local_map']:
                cnn_input.append(obs[key])
            else:
                raise NotImplementedError

        cnn_input = torch.cat(cnn_input, dim=1)
        return cnn_input

    def _build_mlp_input(self, obs):
        mlp_input = []
        for key in self.mlp_keys:
            mlp_input.append(obs[key])

        mlp_input = torch.cat(mlp_input, dim=1)
        return mlp_input