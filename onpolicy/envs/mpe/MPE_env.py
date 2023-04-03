from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, 
        scenario.observation, scenario.info,
        done_callback=scenario.done if hasattr(scenario, "done") else None,
        action_space=scenario.action_space if hasattr(scenario, "action_space") else None,
        observation_space=scenario.observation_space if hasattr(scenario, "observation_space") else None,
        share_observation_space=scenario.share_observation_space if hasattr(scenario, "share_observation_space") else None,
    )

    return env
