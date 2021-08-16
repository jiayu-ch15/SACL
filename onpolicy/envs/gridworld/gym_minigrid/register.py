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
    reward_threshold=0.95,
    use_merge = True,
    use_local = True,
    use_single = True,
    use_same_location = True,
    use_complete_reward = True,
    use_multiroom = False,
    use_time_penalty = False,
    use_fc_net = False,
    use_orientation = False,
    use_agent_id = False,
    use_irregular_room = False,
    astar_cost_mode = "normal"
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
        'use_merge': use_merge,
        'use_local': use_local,
        'use_fc_net':use_fc_net,
        'use_orientation':use_orientation,
        'use_single': use_single,
        'use_same_location': use_same_location,
        'use_complete_reward': use_complete_reward,
        'use_multiroom': use_multiroom,
        'use_irregular_room': use_irregular_room,
        'use_time_penalty': use_time_penalty,
        'use_agent_id': use_agent_id,
        'astar_cost_mode' : astar_cost_mode
        },
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
