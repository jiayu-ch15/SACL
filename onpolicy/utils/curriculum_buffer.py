from dgl.geometry import farthest_point_sampler
import copy
import numpy as np
import time
import torch


class CurriculumBuffer(object):

    def __init__(self, buffer_size, update_method="fps", sample_method="random"):
        self.eps = 1e-10
        self.buffer_size = buffer_size
        self.update_method = update_method
        self.sample_method = sample_method

        self._state_buffer = np.zeros((0, 1), dtype=np.float32)
        self._weight_buffer = np.zeros((0, 1), dtype=np.float32)
        self._temp_state_buffer = []
        self._temp_weight_buffer = []

    def insert(self, states, weights=None):
        """
        input:
            states: list of np.array(size=(state_dim, ))
            weight: list of np.array(size=(1, ))
        """
        self._temp_state_buffer.extend(states)
        if weights is None:
            self._temp_weight_buffer.extend([np.zeros(1) for _ in range(len(states))])
        else:
            self._temp_weight_buffer.extend(weights)

    def update(self):
        start_time = time.time()

        # concatenate to get all states and weight
        all_states = np.array(self._temp_state_buffer)
        all_weights = np.array(self._temp_weight_buffer)
        if self._state_buffer.shape[0] != 0:  # state buffer is not empty
            all_states = np.concatenate([self._state_buffer, all_states], axis=0)
            all_weights = np.concatenate([self._weight_buffer, all_weights], axis=0)

        # update 
        if all_states.shape[0] <= self.buffer_size:
            self._state_buffer = copy.deepcopy(all_states)
            self._weight_buffer = copy.deepcopy(all_weights)
        else:
            if self.update_method == "random":
                random_idx = np.random.choice(all_states.shape[0], self.buffer_size, replace=False)
                self._state_buffer = copy.deepcopy(all_states[random_idx])
                self._weight_buffer = copy.deepcopy(all_weights[random_idx])
            elif self.update_method == "fps":
                # farthest point sampling
                min_states = np.min(all_states, axis=0)
                max_states = np.max(all_states, axis=0)
                all_states_normalized = (all_states - min_states) / (max_states - min_states + self.eps)
                # TODO: mask unnecessary dim
                # consider_dim = np.array([True for _ in range(20)] + [False])  # Don't consider time.
                consider_dim = np.array([True for _ in range(16)] + [False for _ in range(5)])  # Don't consider landmarks and time.
                all_states_tensor = torch.tensor(all_states_normalized[np.newaxis, :, consider_dim])
                fps_idx = farthest_point_sampler(all_states_tensor, self.buffer_size)[0].numpy()
                # update state and weight buffer
                self._state_buffer = copy.deepcopy(all_states[fps_idx])
                self._weight_buffer = copy.deepcopy(all_weights[fps_idx])
            else:
                raise NotImplementedError(f"Update method {self.update_method} is not supported.")
        
        # reset temp state and weight buffer
        self._temp_state_buffer = []
        self._temp_weight_buffer = []

        # print update time
        end_time = time.time()
        print(f"curriculum update time: {end_time - start_time}s")

    def sample(self, num_samples):
        """
        return list of np.array
        """
        if self.sample_method == "random":
            random_idx = np.random.choice(self._state_buffer.shape[0], num_samples, replace=True)
            initial_states = [self._state_buffer[idx] for idx in random_idx]
        else:
            raise NotImplementedError(f"Sample method {self.update_method} is not supported.")
        
        return initial_states
