from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    num_agents,
    grid_size,
    max_steps,
    agent_view_size,
    num_obstacles,
    agent_pos,
    entry_point,
    reward_threshold=0.95,
    use_merge = True,
    use_same_location = True,
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        kwargs={'num_agents': num_agents,
        'grid_size': grid_size,
        'max_steps': max_steps,
        'agent_view_size': agent_view_size,
        'num_obstacles': num_obstacles,
        'agent_pos': agent_pos,
        'use_merge': use_merge,
        'use_same_location' : use_same_location,
        },
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
