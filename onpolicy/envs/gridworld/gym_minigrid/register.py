from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    num_agents,
    num_preies,
    num_obstacles,
    direction_alpha,
    entry_point,
    reward_threshold=0.95
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        kwargs={'num_agents': num_agents, 'num_preies': num_preies, 'num_obstacles': num_obstacles, 'direction_alpha': direction_alpha},
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
