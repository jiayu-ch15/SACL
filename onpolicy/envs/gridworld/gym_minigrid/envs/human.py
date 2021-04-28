from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
from icecream import ic

class HumanEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        num_agents=2,
        num_preies=2,
        num_obstacles=4,
        direction_alpha=0.5,
        use_human_command=False,
        size=19
    ):
        self.num_preies = num_preies
        self.direction_alpha = direction_alpha
        self.use_human_command = use_human_command
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
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Types and colors of objects we can generate
        types = ['key', 'box', 'ball']

        objs = []
        objPos = []

        # Until we have generated all the objects
        while len(objs) < self.num_preies:
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
        
        # direction
        array_direction = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
        self.direction = []
        self.direction_encoder = []
        for agent_id in range(self.num_agents):
            direction = np.sign(self.target_pos - self.agent_pos[agent_id])
            direction_encoder = np.eye(4)[np.argmax(np.all(np.where(array_direction == direction, True, False), axis=1))]
            self.direction.append(direction)
            self.direction_encoder.append(direction_encoder)

        # text
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
                # done = True
            rewards.append(reward)

        dones = [done for agent_id in range(self.num_agents)]

        return obs, rewards, dones, info



