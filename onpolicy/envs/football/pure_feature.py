import gfootball
import numpy as np
import copy
import random


num_agents = 4
random.seed(300)
if __name__ == "__main__":
    env = gfootball.env.create_environment(
        env_name="academy_counterattack_easy",
        number_of_left_players_agent_controls=num_agents,
        number_of_right_players_agent_controls=0,
        representation="simple115v2"
    )
    env.reset()
    differ = np.zeros(shape=(num_agents,115), dtype=bool)
    dones = False
    for k in range(9999):
        if not dones:
            obs, _, dones, _ = env.step(np.random.randint(low=0, high=18, size=(num_agents,)))
        else:
            obs = env.reset()
            dones = False
        if k > 0:
            # for i in range(3):
            differ = np.logical_or(obs_ != obs, differ)
            print(np.where(differ))

        obs_ = copy.deepcopy(obs)
