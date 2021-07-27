from onpolicy.utils.RRTStar.rrt_star import RRTStar
from .frontier import find_rectangle_obstacles
import numpy as np
import math
import pickle

class RRTPlanner(object):
    def __init__(self, traversible):
        self.map = (1 - traversible).astype(np.uint8)
        self.obstacles = find_rectangle_obstacles(self.map)
        self.H, self.W = self.map.shape
        self.start = None
        self.goal = None
    
    def set_goal(self, goal):
        self.goal = (goal[1], goal[0])
    
    def get_short_term_goal(self, start, step = None, agent = None):
        self.start = start
        rrt_star = RRTStar(start=start,
            goals=[self.goal],
            rand_area=((0, self.H), (0, self.W)),
            obstacle_list=self.obstacles,
            expand_dis=3.0 ,
            goal_sample_rate=10,
            max_iter=10000,
            connect_circle_dist=30.0)
        path = rrt_star.planning(animation=False, smooth_path = True)

        if math.hypot(path[-1][0] - self.goal[0], path[-1][1] - self.goal[1]) > 1e-5:
            return start[0], start[1], True
        to = path[1]
        d = math.hypot(to[0] - start[0], to[1] - start[1])
        # 5: short-term goal distance
        if d > 5:
            to[0] = start[0] + (to[0] - start[0]) / d * 5
            to[1] = start[1] + (to[1] - start[1]) / d * 5
        return to[0], to[1], False
