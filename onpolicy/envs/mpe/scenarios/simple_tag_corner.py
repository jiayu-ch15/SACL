from gym import spaces
import copy
import numpy as np

from onpolicy.envs.mpe.core import World, Agent, Landmark, Wall
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        self.dim_p = 2
        self.num_adv = args.num_adv
        self.num_good = args.num_good
        self.num_agents = args.num_adv + args.num_good
        self.num_landmarks = args.num_landmarks
        self.corner_min = args.corner_min
        self.corner_max = args.corner_max
        self.horizon = args.horizon
        self.hard_boundary = args.hard_boundary

        world = World(self.corner_max, self.hard_boundary)
        # set any world properties first
        world.world_length = args.horizon
        world.dim_p = 2
        world.dim_c = 2
        world.num_adv = args.num_adv  # 3
        world.num_good = args.num_good  # 1
        world.num_agents = args.num_adv + args.num_good
        world.num_landmarks = args.num_landmarks  # 2
        world.camera_range = args.corner_max + 1
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent {i}"
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < world.num_adv else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark {i}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        # make initial conditions
        self.reset_world(world)
        return world

    @property
    def action_space(self):
        return [spaces.Discrete(self.dim_p * 2 + 1)] * self.num_agents

    @property
    def observation_space(self):
        obs_dim = [2 * self.num_landmarks + 4 * self.num_agents + 1]
        adv_split_shape = [[self.num_landmarks, 2], [self.num_adv - 1, 4], [self.num_good, 4], [1, 1], [1, 4]]
        good_split_shape = [[self.num_landmarks, 2], [self.num_adv, 4], [self.num_good - 1, 4], [1, 1], [1, 4]]
        adv_obs_space = [obs_dim + adv_split_shape] * self.num_adv
        good_obs_space = [obs_dim + good_split_shape] * self.num_good
        return adv_obs_space + good_obs_space

    @property
    def share_observation_space(self):
        return self.observation_space
    
    def get_state(self, world):
        """
        state:
            agent_states: num_agents x [pos_x, pos_y, vel_x, vel_y]
            landmark_states: num_landmarks x [pos_x, pos_y]
            world_step
        """
        # agent states
        agent_states = []
        for ag in world.agents:
            agent_states.append(ag.state.p_pos)
            agent_states.append(ag.state.p_vel)
        # landmark states
        landmark_states = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmark_states.append(landmark.state.p_pos)
        state = np.concatenate(agent_states + landmark_states + [np.array([world.world_step])])
        return state

    def reset_world(self, world, initial_state=None):
        self.num_outside = 0
        self.num_collision = 0
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()

        if initial_state is None:  # sample initial state according to initial state distribution
            # start from 0
            self.start_step = 0
            world.world_step = 0
            # agents
            for agent in world.agents:
                if agent.adversary:  # adversary agent: upper right corner
                    agent.state.p_pos = np.random.uniform(self.corner_min, self.corner_max, world.dim_p)
                else:  # good agent: lower left corner
                    agent.state.p_pos = np.random.uniform(-self.corner_max, -self.corner_min, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            # landmarks
            for landmark in world.landmarks:
                if not landmark.boundary:
                    landmark.state.p_pos = 0.8 * np.random.uniform(-self.corner_max, self.corner_max, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)

        else:  # set environment to initial_state
            # set start step
            self.start_step = int(initial_state[-1])
            world.world_step = int(initial_state[-1])

            # # start from 0
            # self.start_step = 0
            # world.world_step = 0

            # set agents
            for idx, agent in enumerate(world.agents):
                agent.state.p_pos = copy.deepcopy(initial_state[(4 * idx) : (4 * idx + 2)])
                agent.state.p_vel = copy.deepcopy(initial_state[(4 * idx + 2) : (4 * idx + 4)])
                agent.state.c = np.zeros(world.dim_c)

            # set landmarks
            for idx, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    landmark.state.p_pos = copy.deepcopy(initial_state[(4 * self.num_agents + 2 * idx) : (4 * self.num_agents + 2 * idx + 2)])
                    landmark.state.p_vel = np.zeros(world.dim_p)

            # # randomly generate landmarks
            # for landmark in world.landmarks:
            #     if not landmark.boundary:
            #         landmark.state.p_pos = 0.8 * np.random.uniform(-self.corner_max, self.corner_max, world.dim_p)
            #         landmark.state.p_vel = np.zeros(world.dim_p)
        
        # log initial dist
        initial_dist = []
        for good in self.good_agents(world):
            for adv in self.adversaries(world):
                initial_dist.append(np.linalg.norm(good.state.p_pos - adv.state.p_pos))
        self.initial_dist = np.mean(initial_dist)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward
    
    def done(self, agent, world):
        if world.world_step >= self.horizon:
            return True
        else:
            return False
    
    def info(self, world):
        state = self.get_state(world)
        outside_per_step = self.num_outside / (world.world_step - self.start_step)
        collision_per_step = self.num_collision / (world.world_step - self.start_step)
        info = dict(
            state=state, initial_dist=self.initial_dist, start_step=self.start_step,
            num_steps=world.world_step, episode_length=(world.world_step - self.start_step),
            outside_per_step=outside_per_step, collision_per_step=collision_per_step,
        )
        return info

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False  # different from openai
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
                    self.num_collision += 1

        if not self.hard_boundary:
            # agents are penalized for exiting the screen, so that they can be caught by the adversaries
            def bound(x):
                if x < self.corner_max - 0.1:
                    return 0
                if x < self.corner_max:
                    return (x - (self.corner_max - 0.1)) * 10
                return min(np.exp(2 * (x - self.corner_max)), 10)
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= bound(x)
            self.num_outside += np.any(np.abs(agent.state.p_pos) > self.corner_max)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False #different from openai
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10

        if not self.hard_boundary:
            # Make env zero-sum: adversaryies are rewarded if good agents exits the screen.
            def bound(x):
                if x < self.corner_max - 0.1:
                    return 0
                if x < self.corner_max:
                    return (x - (self.corner_max - 0.1)) * 10
                return min(np.exp(2 * (x - self.corner_max)), 10)
            for ag in agents:
                for p in range(world.dim_p):
                    x = abs(ag.state.p_pos[p])
                    rew += bound(x)

        return rew

    def observation(self, agent, world):
        """
        obs:
            num_landmarks x [rel_pos_x, rel_pos_y]
            num_adv x [rel_pos_x, rel_pos_y, vel_x, vel_y]
            num_good x [rel_pos_x, rel_pos_y, vel_x, vel_y]
            1 x [world_step]
            1 x [pos_x, pos_y, vel_x, vel_y]
        """
        # relative positions of landmarks
        landmarks_pos = []
        for landmark in world.landmarks:
            if not landmark.boundary:
                landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
        # relative positions and absolute velocity of other agents
        others_info = []
        for other in world.agents:
            if other is agent: continue
            others_info.append(other.state.p_pos - agent.state.p_pos)
            others_info.append(other.state.p_vel)
        return np.concatenate(landmarks_pos + others_info + [np.array([world.world_step]), agent.state.p_pos, agent.state.p_vel])
