from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    grid_size,
    max_steps,
    agent_view_size,
    num_obstacles,
    num_agents,
    agent_pos,
    entry_point,
    reward_threshold = 0.95,
    direction_alpha = 0.1,
    use_merge = True,
    use_same_location = True,
    use_complete_reward = True, 
    use_human_command = False,
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        kwargs={
        'grid_size': grid_size,
        'max_steps': max_steps,
        'agent_view_size': agent_view_size,
        'num_obstacles': num_obstacles,
        'num_agents': num_agents,
        'agent_pos': agent_pos,
        'direction_alpha': direction_alpha,
        'use_merge': use_merge,
        'use_same_location': use_same_location,
        'use_complete_reward': use_complete_reward,
        'use_human_command': use_human_command, 
        },
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
