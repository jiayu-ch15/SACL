# Author: Boyuan Chen
# Berkeley Artifical Intelligence Research

# The project is largely based on m-byte918's javascript implementation of the game with a lot of bug fixes and optimization for python
# Original Ogar-Edited project https://github.com/m-byte918/MultiOgar-Edited

import gym
from gym import spaces
from .GameServer import GameServer
from .players import Player, Bot
from .gv import *
gv_init()
set_v('user_name', 'zoe')
set_v('obs_size', 1)
set_v('render', True)
if get_v('render'):
    from . import rendering
import numpy as np
from copy import deepcopy
import random

def max(a, b):

    if a>b:return a
    return b

def rand(a, b):

    return random.random() * (b - a) + a

class AgarEnv(gym.Env):
    def __init__(self, args, gamemode = 0, eval = None):
        super(AgarEnv, self).__init__()
        self.args = args
        self.total_step = args.total_step
        if eval:
            self.eval = eval
        else:
            self.eval = args.eval
        self.g = args.gamma
        self.num_agents = args.num_controlled_agent
        self.action_space = spaces.Box(low = -1, high = 1, shape=(3,)) # 11.5 TCH
        obs_size = get_v('obs_size')
        self.observation_space = spaces.Dict( {'t'+str(i):spaces.Box(low=-100, high=100, shape=(obs_size,)) for i in range(self.num_agents)} ) # 11.22 placeholder
        self.viewer = None
        self.gamemode = gamemode
        self.last_mass = [None for i in range(self.num_agents)]
        self.sum_r = np.zeros((self.num_agents, ))
        self.sum_r_g = np.zeros((self.num_agents, ))
        self.sum_r_g_i = np.zeros((self.num_agents, ))
        self.dir = []
        # factors for reward
        self.mass_reward_eps = 0.33
        self.killed_reward_eps = 0
        self.action_repeat = args.action_repeat

    def step(self, actions_):
        
        actions = deepcopy(actions_)
        reward = np.zeros((self.num_agents, ))
        done = np.zeros((self.num_agents, ))
        info = [{} for i in range(self.num_agents)]
        first = True
        for i in range(self.action_repeat):
        
            if not first:
                for j in range(self.num_agents):
                    actions[j * 3 + 2] = -1.
            first = False
            o,r = self.step_(actions) 
            reward += r
        
        self.m_g *= self.g
        done = (done != 0)
        self.total_step += self.args.num_processes
        
        self.killed += (self.t_killed != 0)
        for i in range(self.num_agents):
          
            info[i]['high_masks'] = True
            info[i]['bad_transition'] = False
            if self.killed[i] >= 1:
                done[i] = True
                info[i]['high_masks'] = False
                if self.killed[i] == 1:
                    info[i]['episode'] = {'r':self.sum_r[i], 'r_g': self.sum_r_g[i], 'r_g_i': self.sum_r_g_i[i], 'hit':self.hit[i], 'dis': self.sum_dis / self.s_n}
                else:info[i]['bad_transition'] = True
            elif self.s_n >= self.stop_step:
                done[i] = True
                info[i]['episode'] = {'r':self.sum_r[i], 'r_g': self.sum_r_g[i], 'r_g_i':self.sum_r_g_i[i], 'hit':self.hit[i], 'dis': self.sum_dis / self.s_n}

        if np.sum(done) == self.num_agents:
            for i in range(self.num_agents):
                info[i]['high_masks'] = True
                if self.killed[i] == 0 and self.s_n >= self.stop_step:info[i]['bad_transition']=True
                elif self.killed[i] == 1:info[i]['bad_transition'] = False
                elif self.killed[i] > 1:info[i]['bad_transition'] = True
                else:
                    print('bug in Env')
                    exit(0)
        
        return o, reward, done, info
    
    def step_(self, actions_):
        actions = deepcopy(actions_)
        act = []
        for i in range(self.num_agents):
            act.extend([actions[i * 3 + 0], actions[i * 3 + 1], actions[i * 3 + 2]])
            if actions[i * 3 + 2] > 0.5:act[-1] = 0
            else:act[-1] = 2
        actions = np.array(act).reshape(-1,3)
        for action, agent in zip(actions, self.agents):
            agent.step(deepcopy(action))
        for i in range(self.num_agents, len(self.server.players)):
            self.server.players[i].step()

        self.server.Update()
        t_rewards = np.array([self.parse_reward(self.agents[i], i) for i in range(self.num_agents)])
        rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    rewards[i] += t_rewards[j]
                elif True:#self.agents[i].in_view(self.agents[j].centerPos):
                    rewards[i] += t_rewards[j] * self.coop_eps[i]

        self.split = np.zeros(self.num_agents)
        observations = [self.parse_obs(self.agents[i], i, actions) for i in range(self.num_agents)]
        t_dis = self.agents[0].centerPos.clone().sub(self.agents[1].centerPos).sqDist() / self.server.config.r
        self.sum_dis += t_dis
        self.near = (t_dis <= 0.5)
        if self.killed[0] + self.killed[1] > 0.1:self.near = False
        self.last_action = deepcopy(actions.reshape(-1))
        self.sum_r += rewards
        self.sum_r_g += rewards * self.m_g
        self.sum_r_g_i += t_rewards * self.m_g
        observations = np.array(observations)
        self.s_n += 1
         
        observations = {'t'+str(i): observations[i] for i in range(self.num_agents)} # 11.22 placeholder
        return observations, rewards

    def reset(self):
        
        while 1:
            self.num_bots = 5
            self.num_players = self.num_bots +self.num_agents
            self.rewards_forced = [0 for i in range(self.num_agents)]
            self.stop_step = 2000 - random.randint(0, 100) * self.action_repeat
            self.last_mass = [None for i in range(self.num_agents)]
            self.killed = np.zeros(self.num_agents)
            self.t_killed = np.zeros(self.num_agents)
            #self.must_killed = [0 for i in range(self.num_agents)]
            self.sum_r = np.zeros((self.num_agents, ))
            self.sum_r_g = np.zeros((self.num_agents, ))
            self.sum_r_g_i = np.zeros((self.num_agents, ))
            self.sum_dis = 0.
            self.m_g = 1.
            self.last_action = [0 for i in range(3 * self.num_players)]
            self.s_n = 0
            self.kill_reward_eps = np.ones(self.num_agents) * 0.33 * 0#rand(0, 1)
            self.coop_eps = np.ones(self.num_agents) * 0.
            self.split = np.zeros(self.num_agents)
            self.hit = np.zeros((self.num_agents, 4))
            self.near = False
            if self.eval:
                self.kill_reward_eps = np.ones(self.num_agents) * 0.33 * (1 - self.eval['alpha'])
                self.coop_eps = np.ones(self.num_agents) * self.eval['beta']
                self.bot_speed = 1.0
            else:
                up  = min(1.0, max(0.0, (self.total_step - 5e6) / 5e6))
                low = min(1.0, max(0.0, (self.total_step - 1e7) / 5e6))
                #print(self.total_step)
                add = rand(low, up)
                self.bot_speed = 0.0
                self.bot_speed += add
            #print('speed in Env', self.bot_speed, self.total_step)
            set_v('bot_speed', self.bot_speed)
            self.server = GameServer(self)
            self.server.start(self.gamemode)
            self.agents = [Player(self.server) for _ in range(self.num_agents)]
            self.bots = [Bot(self.server) for _ in range(self.num_bots)]
            self.players = self.agents + self.bots
            self.server.addPlayers(self.players)
            self.viewer = None
            self.server.Update()
            observations = [self.parse_obs(self.agents[i], i) for i in range(self.num_agents)] # 11.5 TCH
            success = True
            for i in range(self.num_agents):
                if np.sum(observations[i]) == 0.:
                    success = False
            observations = np.array(observations)
            observations = {'t'+str(i): observations[i] for i in range(self.num_agents)} # 11.22 placeholder
            if success:break
        return observations

    def parse_obs(self, player, id, action = None):
        n = [10, 5, 5, 10, 10, 10, 5, 5, 10, 10]
        s_glo = 28
        s_size_i = [15, 7, 5, 15, 15, 1, 1, 1, 1, 1]
        #for i in range(len(s_size_i)):
        #    s_size_i[i] += self.num_agents
        s_size = np.sum(np.array(n) * np.array(s_size_i)) + s_glo # 564
        if len(player.cells) == 0:return np.zeros(s_size)
        obs = [[], [], [], [], []]
        for cell in player.viewNodes:
            t, feature = self.cell_obs(cell, player, id)
            obs[t].append(feature)
            
        for i in range(len(obs)):
            obs[i].sort(key = lambda x:x[0] ** 2 + x[1] ** 2)
        obs_f = np.zeros(s_size)
        bound = self.players[id].get_view_box()
        b_x = (bound[1] - bound[0]) / self.server.config.serverViewBaseX
        b_y = (bound[3] - bound[2]) / self.server.config.serverViewBaseY
        obs_f[-7] = player.centerPos.sqDist() / self.server.config.r
        obs_f[-8] = b_x
        obs_f[-9] = b_y
        obs_f[-10] = len(obs[0])
        obs_f[-11] = len(obs[1])
        obs_f[-12] = len(obs[2])
        obs_f[-13] = len(obs[3])
        obs_f[-14] = len(obs[4])
        obs_f[-15] = player.maxcell().radius / 400
        obs_f[-16] = player.mincell().radius / 400
        obs_f[-19:-16] = self.last_action[id * 3 : id * 3 + 3]
        obs_f[-20] = self.bot_speed
        obs_f[-21] = (self.killed[id] != 0)
        obs_f[-22] = (self.killed[1 - id] != 0)
        obs_f[-23] = sum([c.mass for c in player.cells]) / 50
        obs_f[-24] = sum([c.mass for c in self.agents[1 - id].cells]) / 50
        obs_f[-25] = 0#self.coop_eps[id]
        obs_f[-26] = 0#self.coop_eps[1 - id]
        obs_f[-27] = 0#self.kill_reward_eps[id]
        obs_f[-28] = 0#self.kill_reward_eps[1 - id]
        base = 0
        for j in range(10):
            lim = min(len(obs[j % 5]), n[j % 5])
            if j >= 5:
                for i in range(lim):
                    obs_f[base + i] = 1.
            else:
                for i in range(lim):
                    obs_f[base + i * s_size_i[j]:base + (i + 1)* s_size_i[j]] = obs[j][i] 
            base += s_size_i[j] * n[j]

        position_x = (player.centerPos.x) / self.server.config.borderWidth * 2 # [-1, 1]
        position_y = (player.centerPos.y) / self.server.config.borderHeight * 2 # [-1, 1]
        '''t_reward = min(1 - np.abs(position_x), 1 - np.abs(position_y))
        if t_reward <= 0.1:self.rewards_forced[id] = - 2 / 1000
        else:self.rewards_forced[id] = 0.'''
        obs_f[-4] = position_x
        obs_f[-3] = position_y
        if obs_f[-4] <= -0.99:obs_f[-2] = -1
        if obs_f[-3] <= -0.99:obs_f[-1] = -1
        if obs_f[-4] >=  0.99:obs_f[-2] =  1
        if obs_f[-3] >=  0.99:obs_f[-1] =  1
        obs_f[-5] = obs[0][0][-3]
        obs_f[-6] = obs[0][0][-2]
        return deepcopy(obs_f)

    def cell_obs(self, cell, player, id):
        if cell.cellType == 0:
            # player features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / self.server.config.splitVelocity # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / self.server.config.splitVelocity # [-1, 1]
            radius = cell.radius / 400 #need to think about mean though [0, infinite...]  # fixme
            log_radius = np.log(cell.radius / 100) #need to think about mean though   # fixme
            position_x = (cell.position.x) / self.server.config.borderWidth * 2 # [-1, 1]
            position_y = (cell.position.y) / self.server.config.borderHeight * 2 # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x) / self.server.config.serverViewBaseX * 2 # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y) / self.server.config.serverViewBaseY * 2 # [-1, 1]
            v_x = (cell.position.x - cell.last_position.x) / self.server.config.serverViewBaseX * 2 * 30
            v_y = (cell.position.y - cell.last_position.y) / self.server.config.serverViewBaseY * 2 * 30
            #canRemerge = onehot(cell.canRemerge, ndim=2) # len 2 onehot 0 or 1
            #ismycell = onehot(cell.owner == player, ndim=2) # len 2 onehot 0 or 1
            #owner = onehot(cell.owner.pID, ndim=self.num_players)
            features_player = np.array([[boost_x, boost_y, radius, log_radius, position_x, position_y, relative_position_x, relative_position_y]])
            features_player = [relative_position_x, relative_position_y, position_x, position_y, boost_x, boost_y, radius, log_radius, float(cell.canRemerge == True), relative_position_x ** 2 + relative_position_y ** 2, v_x, v_y, float(cell.radius * 1.15 > player.maxcell().radius), float(cell.radius * 1.15 > player.mincell().radius), cell.position.sqDist() / self.server.config.r] # 11.5 TCH
            if cell.owner == player:
                c_t = 0
                if boost_x or boost_y:self.split[id] = True
            elif cell.owner.pID < self.num_agents:c_t = 4
            else:c_t = 3
            return c_t, features_player

        elif cell.cellType == 1:
            # food features
            radius = (cell.radius - (self.server.config.foodMaxRadius + self.server.config.foodMinRadius) / 2) / (self.server.config.foodMaxRadius - self.server.config.foodMinRadius) *2  # fixme
            log_radius = np.log(cell.radius / ((self.server.config.foodMaxRadius + self.server.config.foodMinRadius) / 2))  # fixme
            position_x = (cell.position.x) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_food = np.array([[radius, log_radius, position_x, position_y, relative_position_x, relative_position_y]])
            features_food = [relative_position_x, relative_position_y, position_x, position_y,  radius, log_radius, relative_position_x ** 2 + relative_position_y ** 2] # 11.5 TCH
            return cell.cellType, features_food

        elif cell.cellType == 2:
            # virus features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / self.server.config.splitVelocity  # [-1, 1]
            radius = (cell.radius - (self.server.config.virusMaxRadius + self.server.config.virusMinRadius) / 2) / (self.server.config.virusMaxRadius - self.server.config.virusMinRadius) * 2  # fixme
            log_radius = np.log(cell.radius / ((self.server.config.virusMaxRadius + self.server.config.virusMinRadius) / 2))  # fixme
            position_x = (cell.position.x) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_virus = np.array([[boost_x, boost_y, radius, log_radius, position_x, position_y, relative_position_x, relative_position_y]])
            features_virus = [relative_position_x, relative_position_y, position_x, position_y, relative_position_x ** 2 + relative_position_y ** 2] # 11.5 TCH
            return cell.cellType, features_virus

        elif cell.cellType == 3:
            return None # 1.3 TBC
            # ejected mass features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / self.server.config.splitVelocity  # [-1, 1]
            position_x = (cell.position.x - self.server.config.borderWidth / 2) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y - self.server.config.borderHeight / 2) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x + 0 * self.server.config.serverViewBaseX / 2) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y + 0 * self.server.config.serverViewBaseY / 2) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_food = np.array([[boost_x, boost_y, position_x, position_y, relative_position_x, relative_position_y]])
            return cell.cellType, features_food

    def parse_reward(self, player, id):
        mass_reward, kill_reward, killed_reward = self.calc_reward(player, id)
        if mass_reward < 0:mass_reward = 0
        if mass_reward > 0:
            if kill_reward:
                self.hit[id][2] += 1
            elif self.split[id]:self.hit[id][0] += 1
            elif self.near:self.hit[id][3] += 1
            else:self.hit[id][1] += 1
                
        #mass_reward = int(mass_reward > 0)
        #if random.random() <= 0.03:
        #    self.must_killed[id] = True
        if len(player.cells) == 0:# or self.must_killed[id]:
            self.t_killed[id] += 1
        # reward for being --- big, not dead, eating part of others, killing all of others, not be eaten by someone
        reward = mass_reward * self.mass_reward_eps + \
                 kill_reward * self.kill_reward_eps[id] + \
                 killed_reward * self.killed_reward_eps
        if self.eval:reward -= kill_reward * self.kill_reward_eps[id]
        return reward

    def calc_reward(self, player, id):
        mass_reward = sum([c.mass for c in player.cells])
        if self.last_mass[id] is None:# or self.killed[id]:
            #self.killed[id] = False
            mass_reward = 0
        else:mass_reward -= self.last_mass[id]
        self.last_mass[id] = sum([c.mass for c in player.cells])
        #print('mass_reward', mass_reward)
        kill_reward = player.killreward
        killedreward = player.killedreward
        return mass_reward, kill_reward, killedreward

    def add_dir(self, a):

        self.dir = deepcopy(a)

    def render(self, playeridx, mode = 'human', name = ""):
    
        if not get_v('render'):
            return
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.server.config.serverViewBaseX, self.server.config.serverViewBaseY)
            self.render_border()
            self.render_grid()

        bound = self.players[playeridx].get_view_box()
        print('bound in Env', bound)
        self.viewer.set_bounds(*bound)
        # self.viewer.set_bounds(-7000, 7000, -7000, 7000)

        self.geoms_to_render = []
        # self.viewNodes = sorted(self.viewNodes, key=lambda x: x.size)
        self.render_dir(self.players[playeridx].centerPos)
        for node in self.players[playeridx].viewNodes:
            self.add_cell_geom(node)

        self.geoms_to_render = sorted(self.geoms_to_render, key=lambda x: x.order)
        for geom in self.geoms_to_render:
            self.viewer.add_onetime(geom)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array', name = name)

    def render_border(self):
        map_left = - self.server.config.borderWidth / 2
        map_right = self.server.config.borderWidth / 2
        map_top = - self.server.config.borderHeight / 2
        map_bottom = self.server.config.borderHeight / 2
        line_top = rendering.Line((map_left, map_top), (map_right, map_top))
        line_top.set_color(0, 0, 0)
        self.viewer.add_geom(line_top)
        line_bottom = rendering.Line((map_left, map_bottom), (map_right, map_bottom))
        line_bottom.set_color(0, 0, 0)
        self.viewer.add_geom(line_bottom)
        line_left = rendering.Line((map_left, map_top), (map_left, map_bottom))
        line_left.set_color(0, 0, 0)
        self.viewer.add_geom(line_left)
        map_right = rendering.Line((map_right, map_top), (map_right, map_bottom))
        map_right.set_color(0, 0, 0)
        self.viewer.add_geom(map_right)
        cellwall = rendering.make_circle(radius = self.server.config.r, res = 50, filled = False)
        cellwall.set_color(0, 0, 0)
        self.viewer.add_geom(cellwall)

    def render_grid(self):
        map_left = - self.server.config.borderWidth / 2
        map_right = self.server.config.borderWidth / 2
        map_top = - self.server.config.borderHeight / 2
        map_bottom = self.server.config.borderHeight / 2
        for i in range(0, int(map_right), 100):
            line = rendering.Line((i, map_top), (i, map_bottom))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)
            line = rendering.Line((-i, map_top), (-i, map_bottom))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)

        for i in range(0, int(map_bottom), 100):
            line = rendering.Line((map_left, i), (map_right, i))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)
            line = rendering.Line((map_left, -i), (map_right, -i))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)

    def render_dir(self, center):

        for i in range(len(self.dir)):
            line = rendering.Line((0, 0), (self.dir[i][0] * 500, self.dir[i][1] * 500))
            line.set_color((len(self.dir) - i) / len(self.dir), i / len(self.dir), 0)
            line.order = i
            xform = rendering.Transform()
            line.add_attr(xform)
            xform.set_translation(center.x, center.y)
            self.geoms_to_render.append(line)

    def add_cell_geom(self, cell):
        if cell.cellType == 0:
            cellwall = rendering.make_circle(radius=cell.radius)
            cellwall.set_color(cell.color.r * 0.75 / 255.0, cell.color.g * 0.75 / 255.0 , cell.color.b * 0.75 / 255.0)
            xform = rendering.Transform()
            cellwall.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            cellwall.order = cell.radius
            self.geoms_to_render.append(cellwall)

            geom = rendering.make_circle(radius=cell.radius - max(10, cell.radius * 0.1))
            geom.set_color(cell.color.r / 255.0, cell.color.g / 255.0, cell.color.b / 255.0)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            if cell.owner.maxradius < self.server.config.virusMinRadius:
                geom.order = cell.owner.maxradius + 0.0001
            elif cell.radius < self.server.config.virusMinRadius:
                geom.order = self.server.config.virusMinRadius - 0.0001
            else: #cell.owner.maxradius < self.server.config.virusMaxRadius:
                geom.order = cell.owner.maxradius + 0.0001

            self.geoms_to_render.append(geom)

            # self.viewer.add_onetime(geom)
        elif cell.cellType == 2:
            geom = rendering.make_circle(radius=cell.radius)
            geom.set_color(cell.color.r / 255.0, cell.color.g / 255.0, cell.color.b / 255.0, 0.6)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            geom.order = cell.radius
            self.geoms_to_render.append(geom)

        else:
            geom = rendering.make_circle(radius=cell.radius)
            geom.set_color(cell.color.r / 255.0, cell.color.g / 255.0, cell.color.b / 255.0)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            geom.order = cell.radius
            self.geoms_to_render.append(geom)


    def close(self):
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

def onehot(d, ndim):
    v = [0. for i in range(ndim)]
    v[d] = 1.
    return v
