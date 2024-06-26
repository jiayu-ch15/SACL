from dgl.geometry import farthest_point_sampler
import copy
import numpy as np
import time
import torch


class CurriculumBuffer(object):

    def __init__(self, buffer_size, update_method="fps", scenario='mpe', sample_metric='variance_add_bias'):
        self.eps = 1e-10
        self.buffer_size = buffer_size
        self.update_method = update_method
        self.scenario = scenario
        self.sample_metric = sample_metric

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
        if all_states.shape[0] <= self.buffer_size or self.update_method == "greedy":
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
                if self.scenario == 'mpe':
                    consider_dim = np.array([True for _ in range(20)] + [False])  # w.o. time.
                    # consider_dim = np.array([True for _ in range(16)] + [False for _ in range(4)] + [True])  # w.o. landmarks.
                    # consider_dim = np.array([True for _ in range(16)] + [False for _ in range(5)])  # w.o. time and landmarks.
                elif self.scenario == 'football':
                    consider_dim = np.ones(all_states_normalized.shape[-1],dtype=bool)
                else:
                    consider_dim = np.ones(all_states_normalized.shape[-1],dtype=bool)
                all_states_tensor = torch.tensor(all_states_normalized[np.newaxis, :, consider_dim])
                # farthest point sampling
                fps_idx = farthest_point_sampler(all_states_tensor, self.buffer_size)[0].numpy()
                self._state_buffer = all_states[fps_idx]
                self._share_obs_buffer = all_share_obs[fps_idx]
                if 'least_visited' in self.sample_metric:
                    _state_density = np.matmul(self._state_buffer,all_states.T).mean(axis=1)
            else:
                raise NotImplementedError(f"Update method {self.update_method} is not supported.")
        
        # reset temp state and weight buffer
        self._temp_state_buffer = []
        self._temp_share_obs_buffer = []

        # print update time
        end_time = time.time()
        print(f"curriculum buffer update states time: {end_time - start_time}s")

        if 'least_visited' in self.sample_metric:
            return self._share_obs_buffer.copy(), _state_density.copy()
        else:
            return self._share_obs_buffer.copy()

    def update_weights(self, weights):
        self._weight_buffer = weights.copy()
        if self.update_method == 'greedy':
            if len(self._weight_buffer) > self.buffer_size:
                self._weight_buffer, self._state_buffer, self._share_obs_buffer = self._buffer_sort(self._weight_buffer, self._state_buffer, self._share_obs_buffer)
                self._weight_buffer = np.array(self._weight_buffer[len(self._weight_buffer) - self.buffer_size:])
                self._state_buffer = np.array(self._state_buffer[len(self._state_buffer) - self.buffer_size:])
                self._share_obs_buffer = np.array(self._share_obs_buffer[len(self._share_obs_buffer) - self.buffer_size:])

    def sample(self, num_samples):
        """
        return list of np.array
        """
        if self._state_buffer.shape[0] == 0:  # state buffer is empty
            initial_states = [None for _ in range(num_samples)]
        else:
            weights = self._weight_buffer / np.mean(self._weight_buffer)
            probs = weights / np.sum(weights)
            sample_idx = np.random.choice(self._state_buffer.shape[0], num_samples, replace=True, p=probs)
            initial_states = [self._state_buffer[idx] for idx in sample_idx]
        return initial_states
    
    def save_task(self, model_dir, episode):
        np.save('{}/tasks_{}.npy'.format(model_dir,episode), self._state_buffer)
        np.save('{}/scores_{}.npy'.format(model_dir,episode), self._weight_buffer)

    # for update_metric = greedy
    def _buffer_sort(self, list1, *args): # sort by list1, ascending order
        zipped = zip(list1,*args)
        sort_zipped = sorted(zipped,key=lambda x:(x[0],np.mean(x[1])))
        result = zip(*sort_zipped)
        return [list(x) for x in result]