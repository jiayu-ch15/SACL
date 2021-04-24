from onpolicy.envs.gridworld.gym_minigrid.minigrid import *


class HumanEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        num_agents=2,
        size=8,
        numObjs=3
    ):
        self.numObjs = numObjs

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
        types = ['key', 'ball', 'box']

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
            elif objType == 'ball':
                obj = Ball(objColor)
            elif objType == 'box':
                obj = Box(objColor)

            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)

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
        
        for agent_id in range(self.num_agents):
            ax, ay = self.agent_pos[agent_id]
            tx, ty = self.target_pos

            # Toggle/pickup action terminates the episode
            if action[agent_id] == self.actions.toggle:
                done = True

            # Reward performing the done action next to the target object
            if action[agent_id] == self.actions.done:
                if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
                    reward = self._reward()
                done = True

        rewards = [[reward] for agent_id in range(self.num_agents)]
        dones = [done for agent_id in range(self.num_agents)]

        return obs, rewards, dones, info



