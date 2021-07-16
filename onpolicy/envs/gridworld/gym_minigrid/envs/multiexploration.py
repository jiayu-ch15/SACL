#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time
import pyastar
from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
from .multiroom import *
from icecream import ic
import cv2


class MultiExplorationEnv(MultiRoomEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(
        self,
        grid_size,
        max_steps,
        agent_view_size,
        num_obstacles,
        num_agents=2,
        agent_pos=None, 
        goal_pos=None, 
        use_merge = True,
        use_local = True,
        use_same_location = True,
        use_complete_reward = True,
        use_multiroom = False,
        use_time_penalty = False
    ):
        self.grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_size = 1
        self.max_steps = max_steps
        self.use_same_location = use_same_location
        self.use_complete_reward = use_complete_reward
        self.use_multiroom = use_multiroom
        self.use_time_penalty = use_time_penalty
        self.maxNum = 5
        self.minNum = 2

        if num_obstacles <= grid_size/2 + 1:
            self.num_obstacles = int(num_obstacles)
        else:
            self.num_obstacles = int(grid_size/2)

        super().__init__(minNumRooms = 4,
                        maxNumRooms = 7,
                        maxRoomSize = 8,
                        grid_size = grid_size, 
                        max_steps = max_steps, 
                        num_agents = num_agents, 
                        agent_view_size = agent_view_size, 
                        use_merge = use_merge,
                        use_local = use_local,
                        )
        self.augment = 255 // (np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum())
        self.target_ratio = 0.98
        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((num_agents))
        self.agent_ratio_step = np.ones((num_agents)) * max_steps
        self.merge_ratio_step = max_steps


    def overall_gen_grid(self, width, height):

        if self.use_multiroom:
            self._gen_grid(width, height)
        else:
            # Create the grid
            self.grid = Grid(width, height)
            
            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)
            
            w = self._rand_int(self.minNum, self.maxNum)
            h = self._rand_int(self.minNum, self.maxNum)

            room_w = width // w
            room_h = height // h

            # For each row of rooms
            for j in range(0, h):

                # For each column
                for i in range(0, w):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h
                    #if self.scene_id = 1:
                        # Bottom wall and door
                    if i + 1 < w:

                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))

                        for s in range(self.door_size):
                            self.grid.set(*pos, None)
                            pos = (pos[0], pos[1] + 1)

                    # Bottom wall and door
                    if j + 1 < h:

                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)
        
            # Randomize the player start position and orientation
            if self._agent_default_pos is not None:
                self.agent_pos = self._agent_default_pos
                for i in range(self.num_agents):
                    self.grid.set(*self._agent_default_pos[i], None)
                self.agent_dir = [self._rand_int(0, 4) for i in range(self.num_agents)]  # assuming random start direction
            else:
                self.place_agent(use_same_location = self.use_same_location)
            
            #place object
            self.obstacles = []
            for i_obst in range(self.num_obstacles):
                self.obstacles.append(Obstacle())
                pos = self.place_obj(self.obstacles[i_obst], max_tries=100)

            self.mission = 'Reach the goal'

    def reset(self):
        self.explorable_size = 0
        obs = MiniGridEnv.reset(self, choose=True)
        
        self.num_step = 0
        self.get_ratio = 0
        self.target_ratio = 0.98
        self.gt_map = self.grid.encode()[:,:,0].T
        self.agent_local_map = np.zeros((self.num_agents, self.agent_view_size, self.agent_view_size, 3))
        self.pad_gt_map = np.pad(self.gt_map,((self.agent_view_size, self.agent_view_size), (self.agent_view_size,self.agent_view_size)) , constant_values=(0,0))
        
        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []
        
        for i in range(self.num_agents):
            self.explored_each_map.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.previous_explored_each_map.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))

        for i in range(self.num_agents):
            local_map = np.rot90(obs[i]['image'][:,:,0].T,3)
            pos = [self.agent_pos[i][1] + self.agent_view_size, self.agent_pos[i][0] + self.agent_view_size]
            direction = self.agent_dir[i]
            current_agent_pos.append(pos)
            ### adjust angle
            local_map = np.rot90(local_map, 4-direction)
            if direction == 0:# Facing right
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment
            if direction == 1:# Facing down
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
            if direction == 2:# Facing left
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment
            if direction == 3:# Facing up
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
            for j in range(3):
                mmap = np.rot90(obs[i]['image'][:,:,j].T,3)
                mmap = np.rot90(mmap, 4-direction)
                self.agent_local_map[i,:,:, j] = mmap
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        self.previous_all_map = np.zeros((self.width + 2*self.agent_view_size, self.width + 2*self.agent_view_size))
        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i]
            obstacle_all_map += self.obstacle_each_map[i]
        self.explored_map = np.array(explored_all_map).astype(int)[self.agent_view_size : self.width+self.agent_view_size, self.agent_view_size : self.width+self.agent_view_size]
        
        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        info['agent_local_map'] = self.agent_local_map

        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_explored_reward'] = self.merge_reward
        info['agent_explored_reward'] = self.agent_reward
        info['merge_ratio_step'] = self.merge_ratio_step
        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((self.num_agents))
        self.agent_ratio_step = np.ones((self.num_agents)) * self.max_steps
        self.merge_ratio_step = self.max_steps
    
        return obs, info

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        delta_reward_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        reward_explored_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map_t.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
        for i in range(self.num_agents):     
            local_map = np.rot90(obs[i]['image'][:,:,0].T,3)
            
            pos = [self.agent_pos[i][1] + self.agent_view_size, self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)
            direction = self.agent_dir[i]
            ### adjust angle          
            local_map = np.rot90(local_map, 4-direction)
            if direction == 0:## Facing right
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment
            if direction == 1:## Facing down
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
            if direction == 2:## Facing left
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment
            if direction == 3:#Facing up
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:

                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment
            for j in range(3):
                mmap = np.rot90(obs[i]['image'][:,:,j].T,3)
                mmap = np.rot90(mmap, 4 - direction)
                self.agent_local_map[i,:,:, j] = mmap
        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])
           
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1
            
            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i] - reward_obstacle_each_map[i]
            
            each_agent_rewards.append((np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i] - self.obstacle_each_map[i]
        
        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i]
            obstacle_all_map += self.obstacle_each_map[i]
        
        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        reward_obstacle_all_map = obstacle_all_map.copy()
        reward_obstacle_all_map[reward_obstacle_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map - reward_obstacle_all_map

        reward_previous_all_map = self.previous_all_map.copy()
        reward_previous_all_map[reward_previous_all_map != 0] = 1

        merge_explored_reward = (np.array(delta_reward_all_map) - np.array(reward_previous_all_map)).sum()
        self.previous_all_map = explored_all_map - obstacle_all_map
        self.explored_map = np.array(explored_all_map).astype(int)[self.agent_view_size : self.width + self.agent_view_size, self.agent_view_size : self.width + self.agent_view_size]
        
        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        info['agent_local_map'] = self.agent_local_map
        if self.use_time_penalty:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
            info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
        else:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
            info['merge_explored_reward'] = merge_explored_reward * 0.02
        
        if delta_reward_all_map.sum() / self.no_wall_size >= self.target_ratio:#(self.width * self.height)
            done = True       
            self.merge_ratio_step = self.num_step
            if self.use_complete_reward:
                info['merge_explored_reward'] += 0.1 * (delta_reward_all_map.sum() / self.no_wall_size)     
                
        for i in range(self.num_agents):
            if delta_reward_each_map[i].sum() / self.no_wall_size >= self.target_ratio:#(self.width * self.height)
                self.agent_ratio_step[i] = self.num_step
                # if self.use_complete_reward:
                #     info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))
        
        self.agent_reward = info['agent_explored_reward']
        self.merge_reward = info['merge_explored_reward']
        self.merge_ratio = delta_reward_all_map.sum() / self.no_wall_size #(self.width * self.height)
        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_ratio_step'] = self.merge_ratio_step
        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        return obs, reward, done, info

    def get_short_term_action(self, inputs):
        actions = []
        temp_map = self.gt_map.astype(np.float32)
        temp_map[temp_map == 2] = np.inf
        for i in range(self.num_agents):
            goal = [inputs[i][1], inputs[i][0]]
            agent_pos = [self.agent_pos[i][1], self.agent_pos[i][0]]
            agent_dir = self.agent_dir[i]
            path = pyastar.astar_path(temp_map, agent_pos, goal, allow_diagonal=False)
            if len(path) == 1:
                actions.append(1)
                continue
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            # first quadrant
            if relative_pos[0] < 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 1:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 2:
                    actions.append(1)  # turn right
                    continue
            # second quadrant
            if relative_pos[0] > 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 1:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 2:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 3:
                    actions.append(1)  # turn right
                    continue
            # third quadrant
            if relative_pos[0] > 0 and relative_pos[1] < 0:
                if agent_dir == 1 or agent_dir == 2:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 3:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 0:
                    actions.append(1)  # turn right
                    continue
            # fourth quadrant
            if relative_pos[0] < 0 and relative_pos[1] < 0:
                if agent_dir == 2 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 0:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 1:
                    actions.append(1)  # turn right
                    continue
            if relative_pos[0] == 0 and relative_pos[1] ==0:
                # turn around
                actions.append(1)
                continue
            if relative_pos[0] == 0 and relative_pos[1] > 0:
                if agent_dir == 0:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(0)
                    continue
                else:
                    actions.append(1)
                    continue
            if relative_pos[0] == 0 and relative_pos[1] < 0:
                if agent_dir == 2:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(1)
                    continue
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] > 0 and relative_pos[1] == 0:
                if agent_dir == 1:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(1)
                    continue
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] < 0 and relative_pos[1] == 0:
                if agent_dir == 3:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(0)
                    continue
                else:
                    actions.append(1)
                    continue
        '''
        for i in range(self.num_agents):
            goal = inputs[i]
            agent_pos = self.agent_pos[i]
            agent_dir = self.agent_dir[i]
            relative_pos = np.array(goal) - np.array(agent_pos)
            # first quadrant
            if relative_pos[0] >= 0 and relative_pos[1] <= 0:
                if agent_dir == 0 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 1:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 2:
                    actions.append(1)  # turn right
                    continue
            # second quadrant
            if relative_pos[0] >= 0 and relative_pos[1] >= 0:
                if agent_dir == 0 or agent_dir == 1:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 2:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 3:
                    actions.append(1)  # turn right
                    continue
            # third quadrant
            if relative_pos[0] <= 0 and relative_pos[1] >= 0:
                if agent_dir == 1 or agent_dir == 2:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 3:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 0:
                    actions.append(1)  # turn right
                    continue
            # fourth quadrant
            if relative_pos[0] <= 0 and relative_pos[1] <= 0:
                if agent_dir == 2 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 0:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 1:
                    actions.append(1)  # turn right
                    continue
        '''
        return actions
            
