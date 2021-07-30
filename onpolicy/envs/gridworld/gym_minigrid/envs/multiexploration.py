#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time
import pyastar2d
from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
from .multiroom import *
from icecream import ic
import cv2
import copy

from onpolicy.envs.gridworld.frontier.apf import APF
from onpolicy.envs.gridworld.frontier.utility import utility_goal
from onpolicy.envs.gridworld.frontier.rrt import rrt_goal
from onpolicy.envs.gridworld.frontier.nearest import nearest_goal


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
        use_single = True,
        use_same_location = True,
        use_complete_reward = True,
        use_multiroom = False,
        use_time_penalty = False,
        use_agent_id = False
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
                        use_single = use_single,
                        )
        self.augment = np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()
        self.use_agent_id = use_agent_id
        self.target_ratio = 0.98
        self.merge_ratio = 0
        self.merge_reward = 0
        self.merge_overlap_ratio = 0
        self.merge_repeat_area = 0
        self.agent_repeat_area = np.zeros((num_agents))
        self.agent_length = np.zeros((num_agents))
        self.agent_reward = np.zeros((num_agents))
        self.agent_partial_reward = np.zeros((num_agents))
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
        self.last_agent_pos = self.agent_pos
        self.agent_local_map = np.zeros((self.num_agents, self.agent_view_size, self.agent_view_size, 3))
        self.pad_gt_map = np.pad(self.gt_map,((self.agent_view_size, self.agent_view_size), (self.agent_view_size,self.agent_view_size)) , constant_values=(0,0))
        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []
        # APF repeat penalty
        self.ft_goals = []
        self.apf_penalty = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))


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
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = 255
            if direction == 1:# Facing down
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = 255
            if direction == 2:# Facing left
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = 255
            if direction == 3:# Facing up
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = 255
            for j in range(3):
                mmap = np.rot90(obs[i]['image'][:,:,j].T,3)
                mmap = np.rot90(mmap, 4-direction)
                self.agent_local_map[i,:,:, j] = mmap
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        self.previous_all_map = np.zeros((self.width + 2*self.agent_view_size, self.width + 2*self.agent_view_size))
        
        # APF penalty
        for i in range(self.num_agents):
            x, y = current_agent_pos[i]
            self.apf_penalty[i, x, y] = 5.0 # constant


        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i] * (i+1) / self.augment
            obstacle_all_map += self.obstacle_each_map[i] * (i+1) / self.augment
        self.explored_map = np.array(explored_all_map).astype(int)[self.agent_view_size : self.width+self.agent_view_size, self.agent_view_size : self.width+self.agent_view_size]
        
        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['agent_direction'] = np.array(self.agent_dir)
        info['agent_local_map'] = self.agent_local_map
        info['agent_length'] = self.agent_length
        info['merge_overlap_ratio'] = self.merge_overlap_ratio
        info['agent_repeat_area'] = self.agent_repeat_area
        info['merge_repeat_area'] = self.merge_repeat_area
        
        if self.use_agent_id:
            info['explored_each_map'] = np.array(self.explored_each_map * (i+1) / self.augment)
            info['obstacle_each_map'] = np.array(self.obstacle_each_map * (i+1) / self.augment)
        else:
            info['explored_each_map'] = np.array(self.explored_each_map)
            info['obstacle_each_map'] = np.array(self.obstacle_each_map)

        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_explored_reward'] = self.merge_reward
        info['agent_explored_reward'] = self.agent_reward
        info['agent_explored_partial_reward'] = self.agent_partial_reward
        info['merge_ratio_step'] = self.merge_ratio_step
        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        self.last_merge_ratio = 0
        self.merge_ratio = 0
        self.merge_reward = 0
        self.merge_overlap_ratio = 0
        self.merge_repeat_area = 0
        self.agent_repeat_area = np.zeros((self.num_agents))
        self.agent_length = np.zeros((self.num_agents))
        self.agent_reward = np.zeros((self.num_agents))
        self.agent_partial_reward = np.zeros((self.num_agents))
        self.agent_ratio_step = np.ones((self.num_agents)) * self.max_steps
        self.merge_ratio_step = self.max_steps
        self.ft_info = copy.deepcopy(info)
        return obs, info

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        each_agent_partial_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        delta_reward_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        reward_explored_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        current_agent_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        partial_reward_explored_each_map = np.zeros((self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map_t.append(np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
        for i in range(self.num_agents): 
            local_map = np.rot90(obs[i]['image'][:,:,0].T,3)
            pos = [self.agent_pos[i][1] + self.agent_view_size, self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)
            self.agent_length[i] += abs(self.agent_pos[i][1]-self.last_agent_pos[i][1]) + abs(self.agent_pos[i][0]-self.last_agent_pos[i][0])
            self.last_agent_pos = self.agent_pos
            direction = self.agent_dir[i]
            ### adjust angle          
            local_map = np.rot90(local_map, 4-direction)
            if direction == 0:## Facing right
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]] = 255
            if direction == 1:## Facing down
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]][y+pos[1]-self.agent_view_size//2] = 255
            if direction == 2:## Facing left
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = 255 #(i+1)*self.augment
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size//2][y+pos[1]-self.agent_view_size+1] = 255
            if direction == 3:#Facing up
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:

                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = 255
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size+1][y+pos[1]-self.agent_view_size//2] = 255
            for j in range(3):
                mmap = np.rot90(obs[i]['image'][:,:,j].T,3)
                mmap = np.rot90(mmap, 4 - direction)
                self.agent_local_map[i,:,:, j] = mmap
                
        agent_previous_all_map = self.previous_all_map.copy()
        agent_previous_all_map[agent_previous_all_map != 0] = 1
        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])
            current_agent_map[i] = self.explored_each_map_t[i] - self.obstacle_each_map_t[i]
            current_agent_map[i][current_agent_map[i] != 0 ] = 1
           
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1
            
            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i] - reward_obstacle_each_map[i]
            
            each_agent_rewards.append((delta_reward_each_map[i] - reward_previous_explored_each_map).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i] - self.obstacle_each_map[i]
            
            partial_reward_explored_each_map[i] = np.maximum(agent_previous_all_map, delta_reward_each_map[i])
            each_agent_partial_rewards.append((partial_reward_explored_each_map[i] - agent_previous_all_map).sum())
            
            self.agent_repeat_area[i] += current_agent_map[i][reward_previous_explored_each_map == 1].sum()
        # APF penalty
        self.apf_penalty *= 0.95
        for i in range(self.num_agents):
            x, y = current_agent_pos[i]
            self.apf_penalty[i, x, y] += 5.0 # constant
        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i] * (i+1) / self.augment
            obstacle_all_map += self.obstacle_each_map[i] * (i+1) / self.augment
        
        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        reward_obstacle_all_map = obstacle_all_map.copy()
        reward_obstacle_all_map[reward_obstacle_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map - reward_obstacle_all_map

        reward_previous_all_map = self.previous_all_map.copy()
        reward_previous_all_map[reward_previous_all_map != 0] = 1
        
        current_all_map = np.sum(current_agent_map, axis = 0)
        current_all_map[current_all_map>1] = 1
        self.merge_repeat_area += current_all_map[reward_previous_all_map == 1].sum()

        merge_explored_reward = (np.array(delta_reward_all_map) - np.array(reward_previous_all_map)).sum()
        self.previous_all_map = explored_all_map - obstacle_all_map
        self.explored_map = np.array(explored_all_map).astype(int)[self.agent_view_size : self.width + self.agent_view_size, self.agent_view_size : self.width + self.agent_view_size]
        
        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['agent_direction'] = np.array(self.agent_dir)
        info['agent_local_map'] = self.agent_local_map
        info['agent_length'] = self.agent_length
        info['agent_repeat_area'] = self.agent_repeat_area
        info['merge_repeat_area'] = self.merge_repeat_area
        if self.use_agent_id:
            info['explored_each_map'] = np.array(self.explored_each_map * (i+1) / self.augment)
            info['obstacle_each_map'] = np.array(self.obstacle_each_map * (i+1) / self.augment)
        else:
            info['explored_each_map'] = np.array(self.explored_each_map)
            info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        if self.use_time_penalty:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
            info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
            info['agent_explored_partial_reward'] = np.array(each_agent_partial_rewards) * 0.02 - 0.01
        else:
            info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
            info['merge_explored_reward'] = merge_explored_reward * 0.02
            info['agent_explored_partial_reward'] = np.array(each_agent_partial_rewards) * 0.02
        
        if delta_reward_all_map.sum() / self.no_wall_size >= self.target_ratio:#(self.width * self.height)
            done = True       
            self.merge_ratio_step = self.num_step            
            overlap_delta_map = np.sum(delta_reward_each_map, axis = 0)
            info['merge_overlap_ratio'] = (overlap_delta_map > 1).sum() / delta_reward_all_map.sum()  
            self.merge_overlap_ratio = info['merge_overlap_ratio']
            
            if self.use_complete_reward:
                info['merge_explored_reward'] += 0.1 * (delta_reward_all_map.sum() / self.no_wall_size)     
                
        for i in range(self.num_agents):
            if delta_reward_each_map[i].sum() / self.no_wall_size >= self.target_ratio:#(self.width * self.height)
                self.agent_ratio_step[i] = self.num_step
                # if self.use_complete_reward:
                #     info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))
        
        self.agent_reward = info['agent_explored_reward']
        self.agent_partial_reward = info['agent_explored_partial_reward']
        self.merge_reward = info['merge_explored_reward']
        self.merge_ratio = delta_reward_all_map.sum() / self.no_wall_size #(self.width * self.height)
        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_ratio_step'] = self.merge_ratio_step
        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]
        self.ft_info = copy.deepcopy(info)
        return obs, reward, done, info
    def ft_get_actions(self, args, step = None, mode = ""):
        '''
        frontier-based methods compute actions
        '''
        #input('next step')
        assert mode in ['apf', 'utility', 'nearest', 'rrt'], ('frontier global mode should be in [apf, utility, nearest, rrt]')
        info = self.ft_info
        explored = (info['explored_all_map']>0).astype(np.int32)
        obstacle = (info['obstacle_all_map']>0).astype(np.int32)

        H, W = explored.shape
        steps = [(-1,0),(1,0),(0,-1),(0,1)]

        map = np.ones((H, W)).astype(np.int32) * 3 # 3 for unknown area 
        map[explored == 1] = 0
        map[obstacle == 1] = 1 # 1 for obstacles 
        # frontier
        for x in range(H):
            for y in range(W):
                if map[x,y] == 0:
                    neighbors = [(x+dx, y+dy) for dx, dy in steps]
                    if sum([(map[u,v]==3) for u,v in neighbors])>0:
                        map[x,y] = 2 # 2 for targets(frontiers)
        map[:self.agent_view_size, :] = 1
        map[H-self.agent_view_size:, :] = 1
        map[:, :self.agent_view_size] = 1
        map[:, W-self.agent_view_size:] = 1
        unexplored = (map == 3).astype(np.int32)
        map[map == 3] = 0 # set unkown area to explrable areas

        current_agent_pos = info["current_agent_pos"]
        replan = [False for _ in range(self.num_agents)]
        if self.num_step >= 1:
            for agent_id in range(self.num_agents):
                if (map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] != 2) and (unexplored[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] == 0):
                    replan[agent_id] = True
        goals = []
        for agent_id in range(self.num_agents):
            if replan[agent_id] or len(self.ft_goals) == 0:
                if mode == 'apf':
                    apf = APF(args)
                    path = apf.schedule(map, current_agent_pos, steps, agent_id, self.apf_penalty[agent_id])
                    goal = path[-1]
                elif mode == 'utility':
                    goal = utility_goal(map, unexplored, current_agent_pos[agent_id], steps)
                elif mode == 'nearest':
                    goal = nearest_goal(map, current_agent_pos[agent_id], steps)
                elif mode == 'rrt':
                    goal = rrt_goal(map, unexplored, current_agent_pos[agent_id])
                goals.append(goal)
            else:
                goals.append(self.ft_goals[agent_id])
        self.ft_goals = goals.copy()

        '''for x in range(H):
            for y in range(W):
                o = ' '
                if map[x,y] == 2:
                    o = 'o'
                if map[x,y] == 1:
                    o = '#'
                s = 0
                for i in range(self.num_agents):
                    if current_agent_pos[i][0] == x and current_agent_pos[i][1] == y:
                        s += i+1
                if s>0:
                    o = chr(ord('a') + s - 1)
                s = 0
                for i in range(self.num_agents):
                    if goals[i][0] == x and goals[i][1] == y:
                        s += i+1
                if s>0:
                    o = chr(ord('A') + s-1)
                print(o, end='')
            print()'''

        actions = self.ft_get_short_term_action(map, current_agent_pos, goals)
        actions = np.array(actions, dtype=np.int32)
        goals = np.array(goals, dtype=np.int32)
        '''print('Actions', actions)
        print('Directions', self.agent_dir)
        print('Locations', current_agent_pos)
        print('Goals', [(x,y,unexplored[x,y],map[x,y]) for x,y in goals])
        print('replan', replan)'''
        return actions, goals

    def relative_pose2action(self, agent_dir, relative_pos):
        # first quadrant
        if relative_pos[0] < 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 3:
                    return 2 # forward
                if  agent_dir == 1:
                    return 0 # turn left
                if  agent_dir == 2:
                    return 1 # turn right
        # second quadrant
        if relative_pos[0] > 0 and relative_pos[1] > 0:
            if agent_dir == 0 or agent_dir == 1:
                return 2 # forward
            if  agent_dir == 2:
                return 0 # turn left
            if  agent_dir == 3:
                return 1 # turn right
        # third quadrant
        if relative_pos[0] > 0 and relative_pos[1] < 0:
            if agent_dir == 1 or agent_dir == 2:
                return 2 # forward
            if  agent_dir == 3:
                return 0 # turn left
            if  agent_dir == 0:
                return 1 # turn right
        # fourth quadrant
        if relative_pos[0] < 0 and relative_pos[1] < 0:
            if agent_dir == 2 or agent_dir == 3:
                return 2 # forward
            if  agent_dir == 0:
                return 0 # turn left
            if  agent_dir == 1:
                return 1 # turn right
        if relative_pos[0] == 0 and relative_pos[1] ==0:
            # turn around
            return 1
        if relative_pos[0] == 0 and relative_pos[1] > 0:
            if agent_dir == 0:
                return 2
            if agent_dir == 1:
                return 0
            else:
                return 1
        if relative_pos[0] == 0 and relative_pos[1] < 0:
            if agent_dir == 2:
                return 2
            if agent_dir == 1:
                return 1
            else:
                return 0
        if relative_pos[0] > 0 and relative_pos[1] == 0:
            if agent_dir == 1:
                return 2
            if agent_dir == 0:
                return 1
            else:
                return 0
        if relative_pos[0] < 0 and relative_pos[1] == 0:
            if agent_dir == 3:
                return 2
            if agent_dir == 0:
                return 0
            else:
                return 1
        return None

    def ft_get_short_term_action(self, map, current_agent_pos, goals):
        actions = []
        temp_map = map.copy().astype(np.float32)
        temp_map[map == 0] = 1 # free
        temp_map[map == 2] = 1 # frontiers
        temp_map[map == 1] = np.inf # obstacles
        for i in range(self.num_agents):
            goal = [goals[i][0], goals[i][1]]
            agent_pos = [current_agent_pos[i][0], current_agent_pos[i][1]]
            agent_dir = self.agent_dir[i]
            path = pyastar2d.astar_path(temp_map, agent_pos, goal, allow_diagonal=False)
            # print(temp_map[agent_pos[0], agent_pos[1]], temp_map[goal[0], goal[1]], map[agent_pos[0],agent_pos[1]], map[goal[0], goal[1]])
            if type(path) == type(None) or len(path) == 1:
                actions.append(1)
                continue
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            action = self.relative_pose2action(agent_dir, relative_pos)
            actions.append(action)
        return actions

    def get_short_term_action(self, inputs):
        actions = []
        temp_map = self.gt_map.astype(np.float32)
        temp_map[temp_map == 40] = np.inf
        for i in range(self.num_agents):
            goal = [inputs[i][1], inputs[i][0]]
            agent_pos = [self.agent_pos[i][1], self.agent_pos[i][0]]
            agent_dir = self.agent_dir[i]
            path = pyastar2d.astar_path(temp_map, agent_pos, goal, allow_diagonal=False)
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
            
