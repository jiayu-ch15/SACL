from dgl.geometry import farthest_point_sampler
import copy
import numpy as np
import time
import torch


class CurriculumBuffer(object):

    def __init__(self, buffer_size, update_method="fps"):
        self.eps = 1e-10
        self.buffer_size = buffer_size
        self.update_method = update_method

        self._state_buffer = np.zeros((0, 1), dtype=np.float32)
        self._weight_buffer = np.zeros((0, 1), dtype=np.float32)
        self._share_obs_buffer = np.zeros((0, 1), dtype=np.float32)
        self._temp_state_buffer = []
        self._temp_share_obs_buffer = []

    def insert(self, states, share_obs):
        """
        input:
            states: list of np.array(size=(state_dim, ))
            weight: list of np.array(size=(1, ))
        """
        self._temp_state_buffer.extend(copy.deepcopy(states))
        self._temp_share_obs_buffer.extend(copy.deepcopy(share_obs))

    def update_states(self):
        start_time = time.time()

        # concatenate to get all states
        all_states = np.array(self._temp_state_buffer)
        all_share_obs = np.array(self._temp_share_obs_buffer)
        if self._state_buffer.shape[0] != 0:  # state buffer is not empty
            all_states = np.concatenate([self._state_buffer, all_states], axis=0)
            all_share_obs = np.concatenate([self._share_obs_buffer, all_share_obs], axis=0)

        # update 
        if all_states.shape[0] <= self.buffer_size:
            self._state_buffer = all_states
            self._share_obs_buffer = all_share_obs
        else:
            if self.update_method == "random":
                random_idx = np.random.choice(all_states.shape[0], self.buffer_size, replace=False)
                self._state_buffer = all_states[random_idx]
                self._share_obs_buffer = all_share_obs[random_idx]                
            elif self.update_method == "fps":
                min_states = np.min(all_states, axis=0)
                max_states = np.max(all_states, axis=0)
                all_states_normalized = (all_states - min_states) / (max_states - min_states + self.eps)
                # mask unnecessary dim
                # consider_dim = np.array([True for _ in range(21)])  # consider everything.
                consider_dim = np.array([True for _ in range(20)] + [False])  # w.o. time.
                # consider_dim = np.array([True for _ in range(16)] + [False for _ in range(4)] + [True])  # do not consider landmarks
                # consider_dim = np.array([True for _ in range(16)] + [False for _ in range(5)])  # do not consider landmarks and time.
                all_states_tensor = torch.tensor(all_states_normalized[np.newaxis, :, consider_dim])
                # farthest point sampling
                fps_idx = farthest_point_sampler(all_states_tensor, self.buffer_size)[0].numpy()
                self._state_buffer = all_states[fps_idx]
                self._share_obs_buffer = all_share_obs[fps_idx]
            else:
                raise NotImplementedError(f"Update method {self.update_method} is not supported.")
        
        # reset temp state and weight buffer
        self._temp_state_buffer = []
        self._temp_share_obs_buffer = []

        # print update time
        end_time = time.time()
        print(f"curriculum buffer update states time: {end_time - start_time}s")

        return self._share_obs_buffer.copy()

    def update_weights(self, weights):
        self._weight_buffer = weights.copy()

    def sample(self, num_samples):
        """
        return list of np.array
        """
        if self._state_buffer.shape[0] == 0:  # state buffer is empty
            initial_states = [None for _ in range(num_samples)]
        else:
            weights = self._weight_buffer / np.mean(self._weight_buffer)
            probs = weights ** 5 / np.sum(weights ** 5)
            sample_idx = np.random.choice(self._state_buffer.shape[0], num_samples, replace=True, p=probs)
            initial_states = [self._state_buffer[idx] for idx in sample_idx]
        return initial_states
