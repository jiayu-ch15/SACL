from pyglet.window import key
import numpy as np
from gv import *
gv_init()
set_v('user_name', 'tzg')
set_v('obs_size', 1)
set_v('render', True)
from Env import AgarEnv
import time

render = True
num_agents = 2
class Args():

    def __init__(self):
        
        self.num_controlled_agent = num_agents
        self.num_processes = 64
        self.action_repeat = 1
        self.total_step = 1e8
        self.gamma = 0.99
        
env = AgarEnv(Args(), eval = {'alpha':1.,'beta':0.})
env.seed(0)

step = 1
window = None
action = np.zeros((num_agents, 3))
def on_mouse_motion(x, y, dx, dy):
    action[0][0] = (x / 1920 - 0.5) * 2
    action[0][1] = (y / 1080 - 0.5) * 2

def on_key_press(k, modifiers):

    if k == key.SPACE:
        action[0][2] = 1
    else:
        action[0][2] = 0

start = time.time()
for episode in range(1):
    observation = env.reset()
    while True:
        time.sleep(0.05)
        if step % 40 == 0:
            print('step', step)
            print(step / (time.time() - start))
        if render:
            env.render(0)
            if not window:
                window = env.viewer.window
                window.on_key_press = on_key_press
                window.on_mouse_motion = on_mouse_motion
        a = action.reshape(-1)
        observations, rewards, done, info = env.step(a)
        print(step, rewards)
        action[0][2] = 0
        step+=1
env.close()

