from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
import itertools as itt
from icecream import ic

class HumanEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        num_agents=2,
        size=9,
        numObjs=2,
        num_obstacles=4
    ):
        self.numObjs = numObjs
        # Reduce obstacles if there are too many
        if num_obstacles <= size/2 + 1:
            self.num_obstacles = int(num_obstacles)
        else:
            self.num_obstacles = int(size/2)

        super().__init__(
            num_agents=num_agents,
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = ['key', 'box', 'ball']

        objs = []
        objPos = []

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'box':
                obj = Box(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)

            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.num_obstacles):
            self.obstacles.append(Obstacle())
            pos = self.place_obj(self.obstacles[i_obst], max_tries=100)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        objIdx = self._rand_int(0, len(objs))
        self.targetType, self.target_color = objs[objIdx]
        self.target_pos = objPos[objIdx]

        descStr = '%s %s' % (self.target_color, self.targetType)
        self.mission = 'go to the %s' % descStr
        print(self.mission)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        
        rewards = []
        for agent_id in range(self.num_agents):
            ax, ay = self.agent_pos[agent_id]
            tx, ty = self.target_pos

            # Toggle/pickup action terminates the episode
            if action[agent_id] == self.actions.toggle:
                pass
                # done = True

            # Reward performing the done action next to the target object
            if action[agent_id] == self.actions.done:
                if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
                    reward += self._reward()
                done = True
            rewards.append(reward)

        dones = [done for agent_id in range(self.num_agents)]

        return obs, rewards, dones, info



