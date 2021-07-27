import numpy as np
import math
import pyastar2d

class AstarPlanner(object):
    def __init__(self, traversible):
        self.map = (1 - traversible).astype(np.float32)
        self.map = self.map * 1e6 + 1
        self.start = None
        self.goal = None
    
    def set_goal(self, goal):
        self.goal = (goal[1], goal[0])
    
    def get_short_term_goal(self, start):
        self.start = start
        path = pyastar2d.astar_path(self.map, self.start, self.goal, allow_diagonal=True)
        if len(path) == 1:
            return start[0], start[1], True
        w = min(len(path), 5)
        return path[w-1][0], path[w-1][1], False
