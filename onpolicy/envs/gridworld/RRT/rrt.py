
import random
import math
import copy

# Class of the node of the RRT
class TreeNode(object):
    def __init__(self, x, y):
        self.x = x    # Point coordinates
        self.y = y
        self.parent = None  # The parent index of the node


# Class of RRT
class RRT(object):
    def __init__(self, start, goal, obstacle_list, area):
        self.start = TreeNode(start[0], start[1])  # Start point
        self.end = TreeNode(goal[0], goal[1])      # Goal point
        self.min_rand = area[0]                    # Minimum boundary of region
        self.max_rand = area[1]                    # Maximum boundary of region
        self.expandDis = 0.5                       # The expand distance
        self.Sample = 1/625                        # The probability of selecting the destination
        self.obstacleList = obstacle_list          # Obstacle list
        self.nodeList = [self.start]               # Node list of the RRT

    def program(self):  # Function of path programing
        while True:
            # Generate a random node
            if random.random() > self.Sample:
                ran_node = self.get_random_node()
            else:
                ran_node = [self.end.x, self.end.y]
            # Find the nearest node to the random node
            min_index = self.get_nearest_node(self.nodeList, ran_node)
            nearest_node = self.nodeList[min_index]
            # Calculate the angle
            theta = math.atan2(ran_node[1] - nearest_node.y, ran_node[0] - nearest_node.x)
            # Create and add a new node to the RRT
            new_node = copy.deepcopy(nearest_node)
            new_node.x += self.expandDis * math.cos(theta)
            new_node.y += self.expandDis * math.sin(theta)
            new_node.parent = min_index
            # Test if there is collision
            if not self.collision_check(new_node, self.obstacleList):
                continue
            # Check if it is near the goal
            self.nodeList.append(new_node)
            dx = new_node.x - self.end.x
            dy = new_node.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                break
        # Generate the path from start to the goal
        path = [[self.end.x, self.end.y]]
        last_index = len(self.nodeList) - 1
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path, self.nodeList

    def get_random_node(self):   # Function of getting a random node
        node_x = random.uniform(self.min_rand, self.max_rand)
        node_y = random.uniform(self.min_rand, self.max_rand)
        node = [node_x, node_y]
        return node

    def collision_check(self, new_node, obstacle_list):  # Function of collision check
        flag = 1
        for (x, y, size) in obstacle_list:
            dx = x - new_node.x
            dy = y - new_node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= 0.75:
                flag = 0
        return flag

    def get_nearest_node(self, node_list, ran_node):   # Function of finding the nearest node
        d = [(node.x - ran_node[0]) ** 2 + (node.y - ran_node[1]) ** 2 for node in node_list]
        min_index = d.index(min(d))
        return min_index

