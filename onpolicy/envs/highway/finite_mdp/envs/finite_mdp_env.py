import copy
import sys
sys.path.extend("../../")
sys.path.append("../../")

import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

from finite_mdp.mdp import MDP
from finite_mdp.viewer import MDPViewer


class FiniteMDPEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    VIDEO_FRAMES_PER_STEP = 5

    def __init__(self):
        # Seeding
        self.np_random = None
        self.seed()

        self.viewer = None
        self.automatic_rendering_callback = None
        self.frames_to_render = self.VIDEO_FRAMES_PER_STEP

        self.config = FiniteMDPEnv.default_config()
        self.mdp = None
        self.steps = 0
        self.load_config()
        self.reset()

    @staticmethod
    def default_config():
        return dict(mode="deterministic",
                    transition=[[0]],
                    reward=[[0]],
                    max_steps=100)

    def configure(self, config):
        self.config.update(config)
        self.load_config()

    def copy_with_config(self, config):
        env_copy = copy.deepcopy(self)
        env_copy.config.update(config)
        env_copy.mdp.update(config)
        return env_copy

    def load_config(self):
        self.mdp = MDP.from_config(self.config, np_random=self.np_random)
        self.observation_space = spaces.Discrete(np.shape(self.mdp.transition)[0])
        self.action_space = spaces.Discrete(np.shape(self.mdp.transition)[1])

    def reset(self):
        self.steps = 0
        self.load_config()
        return self.mdp.reset()

    def step(self, action):
        self._automatic_rendering()
        state, reward, done, info = self.mdp.step(action, np_random=self.np_random)
        self.steps += 1
        done = done or self.steps >= self.config["max_steps"]
        return state, reward, done, info

    def _automatic_rendering(self):
        if self.automatic_rendering_callback and self.viewer is not None:
            for _ in range(self.frames_to_render):
                self.automatic_rendering_callback()
            self.frames_to_render = 0

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        if self.viewer is None:
            self.viewer = MDPViewer(self.mdp)

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            self.frames_to_render = self.VIDEO_FRAMES_PER_STEP
            return image

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __deepcopy__(self, memo):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
