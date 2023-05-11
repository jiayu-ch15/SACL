import imageio
import numpy as np
import time
import torch

from onpolicy.runner.competitive.base_ensemble_runner import Runner
import pdb
import copy
import wandb
from sklearn.mixture import GaussianMixture as GMM
from gym.spaces import Box

def _t2n(x):
    return x.detach().cpu().numpy()

DATA_X = 0
DATA_Y = 1
class Databag(object):
    """Hold a set of vectors and provides nearest neighbors capabilities"""

    def __init__(self, dim):
        """
            Args:
                dim:  the dimension of the data vectors
        """
        self.dim = dim
        self.reset()

    def __repr__(self):
        return 'Databag(dim={0}, data=[{1}])'.format(self.dim, ', '.join(str(x) for x in self.data))

    def add(self, x):
        assert len(x) == self.dim
        self.data.append(np.array(x))
        self.size += 1
        self.nn_ready = False

    def reset(self):
        """Reset the dataset to zero elements."""
        self.data     = []
        self.size     = 0
        self.kdtree   = None  # KDTree
        self.nn_ready = False # if True, the tree is up-to-date.

    def nn(self, x, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the k nearest neighbors of x in the observed input data
        Args:
            x:      center
            k:      the number of nearest neighbors to return (default: 1)
            eps:    approximate nearest neighbors.
                     the k-th returned value is guaranteed to be no further than
                     (1 + eps) times the distance to the real k-th nearest neighbor.
            p:      Which Minkowski p-norm to use. (default: 2, euclidean)
            radius: the maximum radius (default: +inf)
        Returns:
            distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim, 'dimension of input {} does not match expected dimension {}.'.format(len(x), self.dim)
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(np.array(x), k_x, radius = radius, eps = eps, p = p)

    def get(self, index):
        return self.data[index]

    def iter(self):
        return iter(self.data)

    def _nn(self, v, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Compute the k nearest neighbors of v in the observed data,
        :see: nn() for arguments descriptions.
        """
        self._build_tree()
        dists, idxes = self.kdtree.query(v, k = k, distance_upper_bound = radius,
                                         eps = eps, p = p)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self):
        """Build the KDTree for the observed data
        """
        if not self.nn_ready:
            self.kdtree   = scipy.spatial.cKDTree(self.data)
            self.nn_ready = True

    def __len__(self):
        return self.size

class Dataset(object):
    """Hold observations an provide nearest neighbors facilities"""

    @classmethod
    def from_data(cls, data):
        """ Create a dataset from an array of data, infering the dimension from the datapoint """
        if len(data) == 0:
            raise ValueError("data array is empty.")
        dim_x, dim_y = len(data[0][0]), len(data[0][1])
        dataset = cls(dim_x, dim_y)
        for x, y in data:
            assert len(x) == dim_x and len(y) == dim_y
            dataset.add_xy(x, y)
        return dataset

    @classmethod
    def from_xy(cls, x_array, y_array):
        """ Create a dataset from two arrays of data.
            :note: infering the dimensions for the first elements of each array.
        """
        if len(x_array) == 0:
            raise ValueError("data array is empty.")
        dim_x, dim_y = len(x_array[0]), len(y_array[0])
        dataset = cls(dim_x, dim_y)
        for x, y in zip(x_array, y_array):
            assert len(x) == dim_x and len(y) == dim_y
            dataset.add_xy(x, y)
        return dataset

    def __init__(self, dim_x, dim_y, lateness=0, max_size=None):
        """
            Args:
                dim_x:  the dimension of the input vectors
                dim_y:  the dimension of the output vectors
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lateness = lateness
        self.max_size = max_size

        self.reset()

# The two next methods are used for plicling/unpickling the object (because cKDTree cannot be pickled).
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['kdtree']
        return odict

    def __setstate__(self,dict):
        self.__dict__.update(dict)
        self.nn_ready = [False, False]
        self.kdtree   = [None, None]


    def reset(self):
        """Reset the dataset to zero elements."""
        self.data     = [[], []]
        self.size     = 0
        self.kdtree   = [None, None]   # KDTreeX, KDTreeY
        self.nn_ready = [False, False] # if True, the tree is up-to-date.
        self.kdtree_y_sub = None
        self.late_points = 0
        
    def add_xy(self, x, y=None):
        #assert len(x) == self.dim_x, (len(x), self.dim_x)
        #assert self.dim_y == 0 or len(y) == self.dim_y, (len(y), self.dim_y)
        self.data[0].append(x)
        if self.dim_y > 0:
            self.data[1].append(y)
        self.size += 1
        if self.late_points == self.lateness:
            self.nn_ready = [False, False]
            self.late_points = 0
        else:
            self.late_points += 1
        # Reduce data size
        if self.max_size and self.size > self.max_size:
            n = self.size - self.max_size
            del self.data[0][:n]
            del self.data[1][:n]
            self.size = self.max_size

    def add_xy_batch(self, x_list, y_list):
        assert len(x_list) == len(y_list)
        self.data[0] = self.data[0] + x_list
        self.data[1] = self.data[1] + y_list
        self.size += len(x_list)
        # Reduce data size
        if self.max_size and self.size > self.max_size:
            n = self.size - self.max_size
            del self.data[0][:n]
            del self.data[1][:n]
            self.size = self.max_size
        
    def get_x(self, index):
        return self.data[0][index]
    
    def set_x(self, x, index):
        self.data[0][index] = x

    def get_x_padded(self, index):
        return np.append(1.0,self.data[0][index])

    def get_y(self, index):
        return self.data[1][index]

    def set_y(self, y, index):
        self.data[1][index] = y

    def get_xy(self, index):
        return self.get_x(index), self.get_y(index)
    
    def set_xy(self, x, y, index):
        self.set_x(x, index)
        self.set_y(y, index)
        
    def get_dims(self, index, dims_x=None, dims_y=None, dims=None):
        if dims is None:
            return np.hstack((np.array(self.data[0][index])[dims_x], np.array(self.data[1][index])[np.array(dims_y) - self.dim_x]))
        else:
            if max(dims) < self.dim_x:
                return np.array(self.data[0][index])[dims]
            elif min(dims) > self.dim_x:
                return np.array(self.data[1][index])[np.array(dims) - self.dim_x]                
            else:
                raise NotImplementedError

    def iter_x(self):
        return iter(d for d in self.data[0])

    def iter_y(self):
        return iter(self.data[1])

    def iter_xy(self):
        return zip(self.iter_x(), self.data[1])

    def __len__(self):
        return self.size

    def nn_x(self, x, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of x in the observed input data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim_x
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(DATA_X, x, k=k_x, radius=radius, eps=eps, p=p)

    def nn_y(self, y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of y in the observed output data
        @see Databag.nn() for argument description
        Returns:
            distance and indexes of found nearest neighbors.
        """
        assert len(y) == self.dim_y
        k_y = min(k, self.size)
        return self._nn(DATA_Y, y, k=k_y, radius=radius, eps=eps, p=p)

    def nn_dims(self, x, y, dims_x, dims_y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of a subset of dims of x and y in the observed output data
        @see Databag.nn() for argument description
        Returns:
            distance and indexes of found nearest neighbors.
        """
        assert len(x) == len(dims_x)
        assert len(y) == len(dims_y)
        if len(dims_x) == 0:
            kdtree = scipy.spatial.cKDTree([np.array(data_y)[np.array(dims_y) - self.dim_x] for data_y in self.data[DATA_Y]])
        elif len(dims_y) == 0:
            kdtree = scipy.spatial.cKDTree([np.array(data_x)[dims_x] for data_x in self.data[DATA_X]])
        else:
            kdtree = scipy.spatial.cKDTree([np.hstack((np.array(data_x)[dims_x], np.array(data_y)[np.array(dims_y) - self.dim_x])) for data_x,data_y in zip(self.data[DATA_X], self.data[DATA_Y])])
        dists, idxes =  kdtree.query(np.hstack((x, y)), 
                               k = k, 
                               distance_upper_bound = radius,
                               eps = eps, 
                               p = p)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _nn(self, side, v, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Compute the k nearest neighbors of v in the observed data,
        Aers:
            side  if equal to DATA_X, search among input data.
                     if equal to DATA_Y, search among output data.
        Returns:
            distance and indexes of found nearest neighbors.
        """
        self._build_tree(side)
        dists, idxes = self.kdtree[side].query(v, k = k, distance_upper_bound = radius,
                                               eps = eps, p = p)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self, side):
        """Build the KDTree for the observed data
        Args:
            side  if equal to DATA_X, build input data tree.
                     if equal to DATA_Y, build output data tree.
        """
        if not self.nn_ready[side]:
            self.kdtree[side]   = scipy.spatial.cKDTree(self.data[side], compact_nodes=False, balanced_tree=False) # Those options are required with scipy >= 0.16
            self.nn_ready[side] = True

class BufferedDataset(Dataset):
    """Add a buffer of a few points to avoid recomputing the kdtree at each addition"""

    def __init__(self, dim_x, dim_y, buffer_size=200, lateness=5, max_size=None):
        """
            Args:
                dim_x:  the dimension of the input vectors
                dim_y:  the dimension of the output vectors
        """
        
        self.buffer_size = buffer_size
        self.lateness = lateness
        self.buffer = Dataset(dim_x, dim_y, lateness=self.lateness)
        
        Dataset.__init__(self, dim_x, dim_y, lateness=0)
        self.max_size = max_size
        
    def reset(self):
        self.buffer.reset()
        Dataset.reset(self)
        
    def add_xy(self, x, y=None):
        if self.buffer.size < self.buffer_size:
            self.buffer.add_xy(x, y)
        else:
            self.data[0] = self.data[0] + self.buffer.data[0]
            if self.dim_y > 0:
                self.data[1] = self.data[1] + self.buffer.data[1]
            self.size += self.buffer.size
            self.buffer = Dataset(self.dim_x, self.dim_y, lateness=self.lateness)
            self.nn_ready = [False, False]
            self.buffer.add_xy(x, y)

            # Reduce data size
            if self.max_size and self.size > self.max_size:
                n = self.size - self.max_size
                del self.data[0][:n]
                del self.data[1][:n]
                self.size = self.max_size
        
    def add_xy_batch(self, x_list, y_list):
        assert len(x_list) == len(y_list)
        Dataset.add_xy_batch(self, self.buffer.data[0], self.buffer.data[1])
        self.buffer = Dataset(self.dim_x, self.dim_y, lateness=self.lateness)
        Dataset.add_xy_batch(self, x_list, y_list)
        self.nn_ready = [False, False]
            
    def get_x(self, index):
        if index >= self.size:
            return self.buffer.data[0][index-self.size]
        else:
            return self.data[0][index]
        
    def set_x(self, x, index):
        if index >= self.size:
            self.buffer.set_x(x, index-self.size)
        else:
            self.data[0][index] = x

    def get_x_padded(self, index):
        if index >= self.size:
            return np.append(1.0, self.buffer.data[0][index-self.size])
        else:
            return np.append(1.0, self.data[0][index])

    def get_y(self, index):
        if index >= self.size:
            return self.buffer.data[1][index-self.size]
        else:
            return self.data[1][index]
        
    def set_y(self, y, index):
        if index >= self.size:
            self.buffer.set_y(y, index-self.size)
        else:
            self.data[1][index] = y
        
    def get_dims(self, index, dims_x=None, dims_y=None, dims=None):
        if index >= self.size:
            return self.buffer.get_dims(index-self.size, dims_x, dims_y, dims)
        else:
            return Dataset.get_dims(self, index, dims_x, dims_y, dims)

    def iter_x(self):
        return iter(d for d in self.data[0] + self.buffer.data[0])

    def iter_y(self):
        return iter(self.data[1] + self.buffer.data[1])

    def iter_xy(self):
        return zip(self.iter_x(), self.data[1] + self.buffer.data[1])

    def __len__(self):
        return self.size + self.buffer.size

    def nn_x(self, x, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of x in the observed input data
        @see Databag.nn() for argument description
        Returns:
            distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim_x
        k_x = min(k, self.__len__())
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(DATA_X, x, k=k_x, radius=radius, eps=eps, p=p)

    def nn_y(self, y, dims=None, k = 1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of y in the observed output data
        @see Databag.nn() for argument description
        Returns:
            distance and indexes of found nearest neighbors.
        """
        if dims is None:
            assert len(y) == self.dim_y
            k_y = min(k, self.__len__())
            return self._nn(DATA_Y, y, k=k_y, radius=radius, eps=eps, p=p)
        else:
            return self.nn_y_sub(y, dims, k, radius, eps, p)
        
    def nn_dims(self, x, y, dims_x, dims_y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of a subset of dims of x and y in the observed output data
        @see Databag.nn() for argument description
        Returns:
            distance and indexes of found nearest neighbors.
        """
        if self.size > 0:
            dists, idxes = Dataset.nn_dims(self, x, y, dims_x, dims_y, k, radius, eps, p)
        else:
            return self.buffer.nn_dims(x, y, dims_x, dims_y, k, radius, eps, p)
        if self.buffer.size > 0:
            buffer_dists, buffer_idxes = self.buffer.nn_dims(x, y, dims_x, dims_y, k, radius, eps, p)
            buffer_idxes = [i + self.size for i in buffer_idxes]
            ziped = zip(dists, idxes)
            buffer_ziped = zip(buffer_dists, buffer_idxes)
            sorted_dists_idxes = sorted(ziped + buffer_ziped, key=lambda di:di[0])
            knns = sorted_dists_idxes[:k]
            return [knn[0] for knn in knns], [knn[1] for knn in knns] 
        else:
            return dists, idxes
        
    def _nn(self, side, v, k=1, radius=np.inf, eps=0.0, p=2):
        """Compute the k nearest neighbors of v in the observed data,
        Args:
            side  if equal to DATA_X, search among input data.
                     if equal to DATA_Y, search among output data.
        Returns:
            distance and indexes of found nearest neighbors.
        """
        if self.size > 0:
            dists, idxes = Dataset._nn(self, side, v, k, radius, eps, p)
        else:
            return self.buffer._nn(side, v, k, radius, eps, p)
        if self.buffer.size > 0:
            buffer_dists, buffer_idxes = self.buffer._nn(side, v, k, radius, eps, p)
            buffer_idxes = [i + self.size for i in buffer_idxes]
            if dists[0] <= buffer_dists:
                return dists, idxes
            else:
                return buffer_dists, buffer_idxes
            ziped = zip(dists, idxes)
            buffer_ziped = zip(buffer_dists, buffer_idxes)
            sorted_dists_idxes = sorted(ziped + buffer_ziped, key=lambda di:di[0])
            knns = sorted_dists_idxes[:k]
            return [knn[0] for knn in knns], [knn[1] for knn in knns] 
        else:
            return dists, idxes

class AbstractTeacher(object):
    '''
        Base class for ACL methods.
        This will be used to sample tasks for the DeepRL student given a task space provided at the beginning of training.
    '''
    def __init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed=None, **args):
        '''
            Creates an ACL teacher.
            Args:
                mins: Lower bounds of task space
                max: Upper bounds of task space
                env_reward_lb: Minimum return possible of the environment (used only if `scale_reward` is activated on the `TeacherController`)
                env_reward_ub: maximum return possible of the environment (used only if `scale_reward` is activated on the `TeacherController`)
                seed: Seed
                **args: Additional kwargs specific to the ACL method
        '''
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        self.random_state = np.random.RandomState(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        # Student's value estimator
        self.value_estimator = lambda state: None

        # If reward should be normalized
        self.env_reward_lb = env_reward_lb
        self.env_reward_ub = env_reward_ub

        # Book-keeping logs
        self.bk = {}

    def create_dist_from_bounds(self, mins, maxs, subspace):
        '''
            Create a gaussian distribution from bounds (either over the whole space or only a subspace if `subspace == True`)
            Args:
                mins: Lower bounds of task space
                max: Upper bounds of task space
                subspace (bool): Whether the gaussian distribution should be over a subspace
                                (with mean randomly sampled and std equal to 10% of each dimension) or spread over the whole
                                task space
        '''
        if subspace:
            mean = np.array([self.random_state.uniform(min, max) for min, max in zip(mins, maxs)])
            variance = [(abs(max - min) * 0.1) ** 2 for min, max in zip(mins, maxs)] # std = 10 % of each dimension
        else:
            mean = np.array([np.mean([min, max]) for min, max in zip(mins, maxs)])
            variance = [(abs(max - min) / 4)**2 for min, max in zip(mins, maxs)] # std = 0.25 * range => ~95.5% of samples are between the bounds
        variance = [1e-6 if v == 0 else v for v in variance]  # avoid errors with null variance
        covariance = np.diag(variance)

        return mean, covariance

    def get_or_create_dist(self, dist_dict, mins, maxs, subspace=False):
        '''
            Get distribution if `dist_dict` is not None else create a new one (Gaussian).
            Args:
                dist_dict: Dictionary containing a gaussian distribution
                mins: Lower bounds of task space
                max: Upper bounds of task space
                subspace (bool): Whether the gaussian distribution should be over a subspace
                          (with mean randomly sampled and std equal to 10% of each dimension) or spread over the whole
                          task space
        '''
        if dist_dict is not None:
            dist_mean = dist_dict["mean"]
            dist_variance = dist_dict["variance"]
        else:
            dist_mean, dist_variance = self.create_dist_from_bounds(mins, maxs, subspace)
        return dist_mean, dist_variance

    def rescale_task(self, task, original_space=(0, 1)):
        '''
            Maps a task from the n-dimensional task space towards a n-dimensional [0, 1] space.
            Args:
                task: Task that has to be mapped
                original_space: Target space bounds
        '''
        return np.array([np.interp(task[i], original_space, (self.mins[i], self.maxs[i]))
                         for i in range(len(self.mins))])

    def inverse_rescale_task(self, task, original_space=(0, 1)):
        '''
            Maps a task from a n-dimensional [0, 1] space towards the n-dimensional task space.
            Args:
                task: Task that has to be mapped
                original_space: Source space bounds
        '''
        return np.array([np.interp(task[i], (self.mins[i], self.maxs[i]), original_space)
                         for i in range(len(self.mins))])

    def record_initial_state(self, task, state):
        '''
            Record initial state of the environment given a task.
        '''
        pass

    def episodic_update(self, task, reward, is_success):
        '''
            Get the episodic reward and binary success reward of a task.
        '''
        pass

    def step_update(self, state, action, reward, next_state, done):
        '''
            Get step-related information.
        '''
        pass

    def sample_task(self):
        '''
            Sample a new task.
        '''
        pass

    def non_exploratory_task_sampling(self):
        '''
            Sample a task without exploration (used to visualize the curriculum)
        '''
        return {"task": self.sample_task(), "infos": None}

    def is_non_exploratory_task_sampling_available(self):
        '''
            Whether the method above can be called.
        '''
        return True

    def dump(self, dump_dict):
        '''
            Save the teacher.
            Args:
                dump_dict: Dictionary storing what must be saved.
        '''
        dump_dict.update(self.bk)
        return 

def proportional_choice(v, random_state, eps=0.):
    '''
        Return an index of `v` chosen proportionally to values contained in `v`.
        Args:
            v: List of values
            random_state: Random generator
            eps: Epsilon used for an Epsilon-greedy strategy
    '''
    if np.sum(v) == 0 or random_state.rand() < eps:
        return random_state.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(random_state.multinomial(1, probas) == 1)[0][0]

# Absolute Learning Progress (ALP) computer object
# It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm
class EmpiricalALPComputer():
    '''
        Absolute Learning Progress (ALP) computer object.
        It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm.
    '''
    def __init__(self, task_size, max_size=None, buffer_size=500):
        self.alp_knn = BufferedDataset(1, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)

    # TODO : only use var
    def compute_alp(self, task, reward):
        alp = 0
        lp = 0
        if len(self.alp_knn) > 5:
            # Compute absolute learning progress for new task
            
            # 1 - Retrieve closest previous task
            dist, idx = self.alp_knn.nn_y(task)
            
            # 2 - Retrieve corresponding reward
            closest_previous_task_reward = self.alp_knn.get_x(idx[0])

            # 3 - Compute alp as absolute difference in reward
            lp = reward - closest_previous_task_reward
            alp = np.abs(lp)

        # Add to database
        self.alp_knn.add_xy(reward, task)
        return alp, lp

# Absolute Learning Progress - Gaussian Mixture Model
class ALPGMM(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, gmm_fitness_func="aic", warm_start=False, nb_em_init=1, fit_rate=250,
                 alp_max_size=None, alp_buffer_size=500, potential_ks=np.arange(2, 11, 1), random_task_ratio=0.3, nb_bootstrap=None, initial_dist=None):
        '''
            Absolute Learning Progress - Gaussian Mixture Model (https://arxiv.org/abs/1910.07224).
            Args:
                gmm_fitness_func: Fitness criterion when selecting best GMM among range of GMMs varying in number of Gaussians.
                warm_start: Restart new fit by initializing with last fit
                nb_em_init: Number of Expectation-Maximization trials when fitting
                fit_rate: Number of episodes between two fit of the GMM
                alp_max_size: Maximum number of episodes stored
                alp_buffer_size: Maximal number of episodes to account for when computing ALP
                potential_ks: Range of number of Gaussians to try when fitting the GMM
                random_task_ratio: Ratio of randomly sampled tasks VS tasks sampling using GMM
                nb_bootstrap: Number of bootstrapping episodes, must be >= to fit_rate
                initial_dist: Initial Gaussian distribution. If None, bootstrap with random tasks
        '''
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        # Range of number of Gaussians to try when fitting the GMM
        self.potential_ks = potential_ks
        # Restart new fit by initializing with last fit
        self.warm_start = warm_start
        # Fitness criterion when selecting best GMM among range of GMMs varying in number of Gaussians.
        self.gmm_fitness_func = gmm_fitness_func
        # Number of Expectation-Maximization trials when fitting
        self.nb_em_init = nb_em_init
        # Number of episodes between two fit of the GMM
        self.fit_rate = fit_rate
        self.nb_bootstrap = nb_bootstrap if nb_bootstrap is not None else fit_rate  # Number of bootstrapping episodes, must be >= to fit_rate
        self.initial_dist = initial_dist  # Initial Gaussian distribution. If None, bootstrap with random tasks

        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = random_task_ratio
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)
        self.random_task_generator.seed(self.seed)

        # Maximal number of episodes to account for when computing ALP
        alp_max_size = alp_max_size
        alp_buffer_size = alp_buffer_size

        # Init ALP computer
        self.alp_computer = EmpiricalALPComputer(len(mins), max_size=alp_max_size, buffer_size=alp_buffer_size)

        self.tasks = []
        self.alps = []
        self.tasks_alps = []

        # Init GMMs
        self.potential_gmms = [self.init_gmm(k) for k in self.potential_ks]
        self.gmm = None

        # Boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_alps': [],
                    'episodes': [], 'tasks_origin': []}

    def init_gmm(self, nb_gaussians):
        '''
            Init the GMM given the number of gaussians.
        '''
        return GMM(n_components=nb_gaussians, covariance_type='full', random_state=self.seed,
                                            warm_start=self.warm_start, n_init=self.nb_em_init)

    def get_nb_gmm_params(self, gmm):
        '''
            Assumes full covariance.
            See https://stats.stackexchange.com/questions/229293/the-number-of-parameters-in-gaussian-mixture-model
        '''
        nb_gmms = gmm.get_params()['n_components']
        d = len(self.mins)
        params_per_gmm = (d*d - d)/2 + 2*d + 1
        return nb_gmms * params_per_gmm - 1

    def episodic_update(self, task, reward, is_success):
        self.tasks.append(task)

        is_update_time = False

        # Compute corresponding ALP
        # alp, lp = self.alp_computer.compute_alp(task, reward)
        self.alps.append(reward)

        # Concatenate task vector with ALP dimension
        self.tasks_alps.append(np.array(task.tolist() + [self.alps[-1]]))

        if len(self.tasks) >= self.nb_bootstrap:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                is_update_time = True
                # 1 - Retrieve last <fit_rate> (task, reward) pairs
                cur_tasks_alps = np.array(self.tasks_alps[-self.fit_rate:])

                # 2 - Fit batch of GMMs with varying number of Gaussians
                self.potential_gmms = [g.fit(cur_tasks_alps) for g in self.potential_gmms]

                # 3 - Compute fitness and keep best GMM
                fitnesses = []
                if self.gmm_fitness_func == 'bic':  # Bayesian Information Criterion
                    fitnesses = [m.bic(cur_tasks_alps) for m in self.potential_gmms]
                elif self.gmm_fitness_func == 'aic':  # Akaike Information Criterion
                    fitnesses = [m.aic(cur_tasks_alps) for m in self.potential_gmms]
                elif self.gmm_fitness_func == 'aicc':  # Modified AIC
                    n = self.fit_rate
                    fitnesses = []
                    for l, m in enumerate(self.potential_gmms):
                        k = self.get_nb_gmm_params(m)
                        penalty = (2*k*(k+1)) / (n-k-1)
                        fitnesses.append(m.aic(cur_tasks_alps) + penalty)
                else:
                    raise NotImplementedError
                    exit(1)
                self.gmm = self.potential_gmms[np.argmin(fitnesses)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['tasks_alps'] = self.tasks_alps
                self.bk['episodes'].append(len(self.tasks))
        return is_update_time

    def sample_task(self):
        task_origin = None
        if len(self.tasks) < self.nb_bootstrap or self.random_state.random() < self.random_task_ratio or self.gmm is None:
            if self.initial_dist and len(self.tasks) < self.nb_bootstrap:  # bootstrap in initial dist
                # Expert bootstrap Gaussian task sampling
                new_task = self.random_state.multivariate_normal(self.initial_dist['mean'],
                                                                 self.initial_dist['variance'])
                new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
                task_origin = -2  # -2 = task originates from initial bootstrap gaussian sampling
            else:
                # Random task sampling
                new_task = self.random_task_generator.sample()
                task_origin = -1  # -1 = task originates from random sampling
        else:
            # ALP-based task sampling
            # 1 - Retrieve the mean ALP value of each Gaussian in the GMM
            self.alp_means = []
            for pos, _, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                self.alp_means.append(pos[-1])

            # 2 - Sample Gaussian proportionally to its mean ALP
            idx = proportional_choice(self.alp_means, self.random_state, eps=0.0)
            task_origin = idx

            # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
            new_task = self.random_state.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)

        # boring book-keeping
        self.bk['tasks_origin'].append(task_origin)
        return new_task

    def is_non_exploratory_task_sampling_available(self):
        return self.gmm is not None

    def non_exploratory_task_sampling(self):
        # 1 - Retrieve the mean ALP value of each Gaussian in the GMM
        alp_means = []
        for pos, _, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
            alp_means.append(pos[-1])

        # 2 - Sample Gaussian proportionally to its mean ALP
        idx = proportional_choice(alp_means, self.random_state, eps=0.0)

        # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
        new_task = self.random_state.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
        new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
        return {"task": new_task,
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": idx}
                }

class GMMBuffer(object):

    def __init__(self):
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
        self._state_buffer = np.array(self._temp_state_buffer)
        self._share_obs_buffer = np.array(self._temp_share_obs_buffer)

        # reset temp state and weight buffer
        self._temp_state_buffer = []
        self._temp_share_obs_buffer = []

        return self._share_obs_buffer.copy()

    # normalize
    def update_weights(self, weights):
        self._weight_buffer = weights.copy()

class MPECurriculumRunner(Runner):
    def __init__(self, config):
        super(MPECurriculumRunner, self).__init__(config)

        self.prob_curriculum = self.all_args.prob_curriculum
        self.sample_metric = self.all_args.sample_metric
        self.alpha = self.all_args.alpha
        self.beta = self.all_args.beta

        self.no_info = np.ones(self.n_rollout_threads, dtype=bool)
        self.env_infos = dict(
            initial_dist=np.zeros(self.n_rollout_threads, dtype=int), 
            start_step=np.zeros(self.n_rollout_threads, dtype=int), 
            end_step=np.zeros(self.n_rollout_threads, dtype=int),
            episode_length=np.zeros(self.n_rollout_threads, dtype=int),
            outside=np.zeros(self.n_rollout_threads, dtype=bool), 
            collision=np.zeros(self.n_rollout_threads, dtype=bool),
            escape=np.zeros(self.n_rollout_threads, dtype=bool),
            outside_per_step=np.zeros(self.n_rollout_threads, dtype=float), 
            collision_per_step=np.zeros(self.n_rollout_threads, dtype=float),
        )
        # sample_metric == "variance_add_bias"
        self.curriculum_infos = dict(V_variance=0.0,V_bias=0.0)
        self.old_red_policy = copy.deepcopy(self.red_policy)
        self.old_blue_policy = copy.deepcopy(self.blue_policy)
        self.old_red_value_normalizer = copy.deepcopy(self.red_trainer.value_normalizer)
        self.old_blue_value_normalizer = copy.deepcopy(self.blue_trainer.value_normalizer)

        # init the gmm
        self.corner_min = self.all_args.corner_min
        self.corner_max = self.all_args.corner_max
        self.max_speed = 1.3
        self.num_landmarks = self.all_args.num_landmarks
        self.gmm_buffer = GMMBuffer()
        self.goals_gmm = ALPGMM(mins = [-self.max_speed] * (self.num_agents * 2) + [-self.corner_max] * (self.num_agents * 2 + self.num_landmarks * 2) + [0], maxs= [self.max_speed] * (self.num_agents * 2) + [self.corner_max] * (self.num_agents * 2 + self.num_landmarks * 2) + [self.all_args.episode_length - 1], seed=self.all_args.seed, env_reward_lb=None, env_reward_ub=None)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # sample actions from policy
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # env step
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # manually reset done envs
                obs = self.reset_subgames(obs, dones)
                # insert data into buffer
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
            # compute return
            self.compute()    
        
            # train network
            red_train_infos, blue_train_infos = self.train()

            self.update_curriculum()
            # train gan
            for idx in range(len(self.gmm_buffer._state_buffer)):
                self.goals_gmm.episodic_update(self.gmm_buffer._state_buffer[idx],self.gmm_buffer._weight_buffer[idx], is_success=None)  

            # hard-copy to get old_policy parameters
            self.old_red_policy = copy.deepcopy(self.red_policy)
            self.old_blue_policy = copy.deepcopy(self.blue_policy)
            self.old_red_value_normalizer = copy.deepcopy(self.red_trainer.value_normalizer)
            self.old_blue_value_normalizer = copy.deepcopy(self.blue_trainer.value_normalizer)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # save checkpoint
            if (episode + 1) % self.save_ckpt_interval == 0:
                self.save_ckpt(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                # basic info
                end = time.time()
                print(
                    f"Env: MPE, Scenario: {self.all_args.scenario_name}, Exp: {self.experiment_name}, "
                    f"Updates: {episode}/{episodes}, Env steps: {total_num_steps}/{self.num_env_steps}, "
                    f"FSP: {int(total_num_steps / (end - start))}."
                )
                # training info
                red_value_mean, red_value_var = self.red_trainer.value_normalizer.running_mean_var()
                blue_value_mean, blue_value_var = self.blue_trainer.value_normalizer.running_mean_var()
                red_train_infos["step_reward"] = np.mean(self.red_buffer.rewards)
                red_train_infos["value_normalizer_mean"] = np.mean(_t2n(red_value_mean))
                red_train_infos["value_normalizer_var"] = np.mean(_t2n(red_value_var))
                blue_train_infos["step_reward"] = np.mean(self.blue_buffer.rewards)
                blue_train_infos["value_normalizer_mean"] = np.mean(_t2n(blue_value_mean))
                blue_train_infos["value_normalizer_var"] = np.mean(_t2n(blue_value_var))
                self.log_train(red_train_infos, blue_train_infos, total_num_steps)
                print(
                    f"adv step reward: {red_train_infos['step_reward']:.2f}, "
                    f"good step reward: {blue_train_infos['step_reward']:.2f}."
                )
                # env info
                self.log_env(self.env_infos, total_num_steps)
                print(
                    f"initial distance: {np.mean(self.env_infos['initial_dist']):.2f}, "
                    f"start step: {np.mean(self.env_infos['start_step']):.2f}, "
                    f"end step: {np.mean(self.env_infos['end_step']):.2f}, "
                    f"episode length: {np.mean(self.env_infos['episode_length']):.2f}.\n"
                    f"outside: {np.mean(self.env_infos['outside']):.2f}, "
                    f"collision: {np.mean(self.env_infos['collision']):.2f}, "
                    f"escape: {np.mean(self.env_infos['escape']):.2f}.\n"
                    f"outside per step: {np.mean(self.env_infos['outside_per_step']):.2f}, "
                    f"collision per step: {np.mean(self.env_infos['collision_per_step']):.2f}.\n"
                )
                self.no_info = np.ones(self.n_rollout_threads, dtype=bool)
                self.env_infos = dict(
                    initial_dist=np.zeros(self.n_rollout_threads, dtype=int), 
                    start_step=np.zeros(self.n_rollout_threads, dtype=int), 
                    end_step=np.zeros(self.n_rollout_threads, dtype=int),
                    episode_length=np.zeros(self.n_rollout_threads, dtype=int),
                    outside=np.zeros(self.n_rollout_threads, dtype=bool), 
                    collision=np.zeros(self.n_rollout_threads, dtype=bool),
                    escape=np.zeros(self.n_rollout_threads, dtype=bool),
                    outside_per_step=np.zeros(self.n_rollout_threads, dtype=float), 
                    collision_per_step=np.zeros(self.n_rollout_threads, dtype=float),
                )
                # curriculum info
                self.log_curriculum(self.curriculum_infos, total_num_steps)
                print(
                    f"V variance: {self.curriculum_infos['V_variance']:.2f}."
                )
                self.curriculum_infos = dict(V_variance=0.0,V_bias=0.0)

            # eval
            if self.use_eval and episode % self.eval_interval == 0:
                self.eval(total_num_steps)

    def log_curriculum(self, curriculum_infos, total_num_steps):
        for k, v in curriculum_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # red buffer
        self.red_buffer.obs[0] = obs[:, :self.num_red].copy()
        self.red_buffer.share_obs[0] = obs[:, :self.num_red].copy()
        # blue buffer
        self.blue_buffer.obs[0] = obs[:, -self.num_blue:].copy()
        self.blue_buffer.share_obs[0] = obs[:, -self.num_blue:].copy()

    def reset_subgames(self, obs, dones):
        # reset subgame: env is done and p~U[0, 1] < prob_curriculum
        env_dones = np.all(dones, axis=1)
        use_curriculum = (np.random.uniform(size=self.n_rollout_threads) < self.prob_curriculum)
        env_idx = np.arange(self.n_rollout_threads)[env_dones * use_curriculum]
        # sample initial states and reset
        initial_states = []
        if len(env_idx) != 0:
            for _ in range(len(env_idx)):
                initial_states.append(self.goals_gmm.sample_task())
            obs[env_idx] = self.envs.partial_reset(env_idx.tolist(), initial_states)
        return obs
    
    def update_curriculum(self):
        # update and get share obs
        share_obs = self.gmm_buffer.update_states()
        # get weights according to metric
        num_weights = share_obs.shape[0]
        if self.sample_metric == "ensemble_var_add_bias":
            # current red V_value
            red_rnn_states_critic = np.zeros((num_weights * self.num_red, self.recurrent_N, self.hidden_size), dtype=np.float32)
            red_masks = np.ones((num_weights * self.num_red, 1), dtype=np.float32)
            red_values = self.red_trainer.policy.get_values(
                np.concatenate(share_obs[:, :self.num_red]),
                red_rnn_states_critic,
                red_masks,
            )
            red_values = np.array(np.split(_t2n(red_values), num_weights))
            red_values_denorm = self.red_trainer.value_normalizer.denormalize(red_values)

            # old red V_value
            old_red_rnn_states_critic = np.zeros((num_weights * self.num_red, self.recurrent_N, self.hidden_size), dtype=np.float32)
            old_red_masks = np.ones((num_weights * self.num_red, 1), dtype=np.float32)
            old_red_values = self.old_red_policy.get_values(
                np.concatenate(share_obs[:, :self.num_red]),
                old_red_rnn_states_critic,
                old_red_masks,
            )
            old_red_values = np.array(np.split(_t2n(old_red_values), num_weights))
            old_red_values_denorm = self.old_red_value_normalizer.denormalize(old_red_values)

            # current blue V_value
            blue_rnn_states_critic = np.zeros((num_weights * self.num_blue, self.recurrent_N, self.hidden_size), dtype=np.float32)
            blue_masks = np.ones((num_weights * self.num_blue, 1), dtype=np.float32)
            blue_values = self.blue_trainer.policy.get_values(
                np.concatenate(share_obs[:, -self.num_blue:]),
                blue_rnn_states_critic,
                blue_masks,
            )
            blue_values = np.array(np.split(_t2n(blue_values), num_weights))
            blue_values_denorm = self.blue_trainer.value_normalizer.denormalize(blue_values)

            # old blue V_value
            old_blue_rnn_states_critic = np.zeros((num_weights * self.num_blue, self.recurrent_N, self.hidden_size), dtype=np.float32)
            old_blue_masks = np.ones((num_weights * self.num_blue, 1), dtype=np.float32)
            old_blue_values = self.old_blue_policy.get_values(
                np.concatenate(share_obs[:, -self.num_blue:]),
                old_blue_rnn_states_critic,
                old_blue_masks,
            )
            old_blue_values = np.array(np.split(_t2n(old_blue_values), num_weights))
            old_blue_values_denorm = self.old_blue_value_normalizer.denormalize(old_blue_values)

            # concat current V value
            values_denorm = np.concatenate([red_values_denorm, -blue_values_denorm], axis=1)
            # concat old V value
            old_values_denorm = np.concatenate([old_red_values_denorm, -old_blue_values_denorm], axis=1)
            # reshape : [batch, num_agents * num_ensemble]
            values_denorm = values_denorm.reshape(values_denorm.shape[0],-1)
            old_values_denorm = old_values_denorm.reshape(old_values_denorm.shape[0],-1)

            # get Var(V_current)
            V_variance = np.var(values_denorm, axis=1)
            # get |V_current - V_old|
            V_bias = np.mean(np.square(values_denorm - old_values_denorm),axis=1)

            weights = self.beta * V_variance + self.alpha * V_bias
            self.curriculum_infos = dict(V_variance=np.mean(V_variance),V_bias=np.mean(V_bias))
        
        self.gmm_buffer.update_weights(weights)

    @torch.no_grad()
    def collect(self, step):
        # red trainer
        self.red_trainer.prep_rollout()
        red_values, red_actions, red_action_log_probs, red_rnn_states, red_rnn_states_critic = self.red_trainer.policy.get_actions(
            np.concatenate(self.red_buffer.share_obs[step]),
            np.concatenate(self.red_buffer.obs[step]),
            np.concatenate(self.red_buffer.rnn_states[step]),
            np.concatenate(self.red_buffer.rnn_states_critic[step]),
            np.concatenate(self.red_buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        # values:[self.envs, agents, num_critic]
        red_values = np.array(np.split(_t2n(red_values), self.n_rollout_threads))
        red_actions = np.array(np.split(_t2n(red_actions), self.n_rollout_threads))
        red_action_log_probs = np.array(np.split(_t2n(red_action_log_probs), self.n_rollout_threads))
        red_rnn_states = np.array(np.split(_t2n(red_rnn_states), self.n_rollout_threads))
        red_rnn_states_critic = np.array(np.split(_t2n(red_rnn_states_critic), self.n_rollout_threads))
        # blue trainer
        self.blue_trainer.prep_rollout()
        blue_values, blue_actions, blue_action_log_probs, blue_rnn_states, blue_rnn_states_critic = self.blue_trainer.policy.get_actions(
            np.concatenate(self.blue_buffer.share_obs[step]),
            np.concatenate(self.blue_buffer.obs[step]),
            np.concatenate(self.blue_buffer.rnn_states[step]),
            np.concatenate(self.blue_buffer.rnn_states_critic[step]),
            np.concatenate(self.blue_buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        blue_values = np.array(np.split(_t2n(blue_values), self.n_rollout_threads))
        blue_actions = np.array(np.split(_t2n(blue_actions), self.n_rollout_threads))
        blue_action_log_probs = np.array(np.split(_t2n(blue_action_log_probs), self.n_rollout_threads))
        blue_rnn_states = np.array(np.split(_t2n(blue_rnn_states), self.n_rollout_threads))
        blue_rnn_states_critic = np.array(np.split(_t2n(blue_rnn_states_critic), self.n_rollout_threads))
        # concatenate
        values = np.concatenate([red_values, blue_values], axis=1)
        actions = np.concatenate([red_actions, blue_actions], axis=1)
        action_log_probs = np.concatenate([red_action_log_probs, blue_action_log_probs], axis=1)
        rnn_states = np.concatenate([red_rnn_states, blue_rnn_states], axis=1)
        rnn_states_critic = np.concatenate([red_rnn_states_critic, blue_rnn_states_critic], axis=1)
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # red buffer
        self.red_buffer.insert(
            share_obs=obs[:, :self.num_red],
            obs=obs[:, :self.num_red],
            rnn_states=rnn_states[:, :self.num_red],
            rnn_states_critic=rnn_states_critic[:, :self.num_red],
            actions=actions[:, :self.num_red],
            action_log_probs=action_log_probs[:, :self.num_red],
            value_preds=values[:, :self.num_red],
            rewards=rewards[:, :self.num_red],
            masks=masks[:, :self.num_red],
        )
        # blue buffer
        self.blue_buffer.insert(
            share_obs=obs[:, -self.num_blue:],
            obs=obs[:, -self.num_blue:],
            rnn_states=rnn_states[:, -self.num_blue:],
            rnn_states_critic=rnn_states_critic[:, -self.num_blue:],
            actions=actions[:, -self.num_blue:],
            action_log_probs=action_log_probs[:, -self.num_blue:],
            value_preds=values[:, -self.num_blue:],
            rewards=rewards[:, -self.num_blue:],
            masks=masks[:, -self.num_blue:],
        )

        # curriculum buffer
        # TODO: better way to filter bad states

        # # only red value
        # denormalized_values = self.red_trainer.value_normalizer.denormalize(values[:, :self.num_red])
        
        # # both red and blue value
        # denormalized_red_values = self.red_trainer.value_normalizer.denormalize(values[:, :self.num_red])
        # denormalized_blue_values = self.blue_trainer.value_normalizer.denormalize(values[:, -self.num_blue:])
        # denormalized_values = np.concatenate([denormalized_red_values, -denormalized_blue_values], axis=1)
        
        new_states = []
        new_share_obs = []
        for info, share_obs in zip(infos, obs):
            state = info[0]["state"]
            if np.any(np.abs(state[0:2]) > self.all_args.corner_max):
                continue
            if np.any(np.abs(state[4:6]) > self.all_args.corner_max):
                continue
            if np.any(np.abs(state[8:10]) > self.all_args.corner_max):
                continue
            if np.any(np.abs(state[12:14]) > self.all_args.corner_max):
                continue
            if np.any(state[-1] >= self.all_args.horizon):
                continue
            new_states.append(state)
            new_share_obs.append(share_obs)
        self.gmm_buffer.insert(new_states, new_share_obs)

        # info dict
        env_dones = np.all(dones, axis=1)
        for idx in np.arange(self.n_rollout_threads)[env_dones * self.no_info]:
            self.env_infos["initial_dist"][idx] = infos[idx][-1]["initial_dist"]
            self.env_infos["start_step"][idx] = infos[idx][-1]["start_step"]
            self.env_infos["end_step"][idx] = infos[idx][-1]["num_steps"]
            self.env_infos["episode_length"][idx] = infos[idx][-1]["episode_length"]
            self.env_infos["outside"][idx] = (infos[idx][-1]["outside_per_step"] > 0)
            self.env_infos["collision"][idx] = (infos[idx][-1]["collision_per_step"] > 0)
            self.env_infos["escape"][idx] = (not self.env_infos["outside"][idx]) and (not self.env_infos["collision"][idx])
            self.env_infos["outside_per_step"][idx] = infos[idx][-1]["outside_per_step"]
            self.env_infos["collision_per_step"][idx] = infos[idx][-1]["collision_per_step"]
            self.no_info[idx] = False

    @torch.no_grad()
    def render(self):
        envs = self.envs
        
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                all_frames = []
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            step_rewards = []
            
            for step in range(self.episode_length):
                # red action
                self.red_trainer.prep_rollout()
                red_actions, red_rnn_states = self.red_trainer.policy.act(
                    np.concatenate(obs[:, :self.num_red]),
                    np.concatenate(rnn_states[:, :self.num_red]),
                    np.concatenate(masks[:, :self.num_red]),
                    deterministic=False,
                )
                red_actions = np.array(np.split(_t2n(red_actions), self.n_rollout_threads))
                red_rnn_states = np.array(np.split(_t2n(red_rnn_states), self.n_rollout_threads))
                # blue action
                self.blue_trainer.prep_rollout()
                blue_actions, blue_rnn_states = self.blue_trainer.policy.act(
                    np.concatenate(obs[:, -self.num_blue:]),
                    np.concatenate(rnn_states[:, -self.num_blue:]),
                    np.concatenate(masks[:, -self.num_blue:]),
                    deterministic=False,
                )
                blue_actions = np.array(np.split(_t2n(blue_actions), self.n_rollout_threads))
                blue_rnn_states = np.array(np.split(_t2n(blue_rnn_states), self.n_rollout_threads))
                # concatenate and env action
                actions = np.concatenate([red_actions, blue_actions], axis=1)
                rnn_states = np.concatenate([red_rnn_states, blue_rnn_states], axis=1)
                actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)

                # env step
                obs, rewards, dones, infos = envs.step(actions_env)
                step_rewards.append(rewards)

                # update
                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                # append image
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            # print result
            step_rewards = np.array(step_rewards)
            adv_step_reward = np.mean(step_rewards[:, :, :self.num_red])
            good_step_reward = np.mean(step_rewards[:, :, -self.num_blue:])

            initial_dist = infos[0][-1]["initial_dist"]
            start_step = infos[0][-1]["start_step"]
            end_step = infos[0][-1]["num_steps"]
            outside_per_step = infos[0][-1]["outside_per_step"]
            collision_per_step = infos[0][-1]["collision_per_step"]
            print(
                f"episode {episode}: adv step rewards={adv_step_reward:.2f}, good step reward={good_step_reward:.2f}, "
                f"initial_dist={initial_dist}, start step={start_step}, end step={end_step}, "
                f"outside per step={outside_per_step}, collision per step={collision_per_step}."
            )

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(f"{self.gif_dir}/episode{episode}.gif", all_frames[:-1], duration=self.all_args.ifi)
