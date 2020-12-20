from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

import argparse
import sys

def get_args(args):
    parser = argparse.ArgumentParser(description='smarts', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_agents', type=int, default=3, help="number of cars")
    parser.add_argument('--num_lanes', type=int, default=3, help="number of lanes")
    all_args = parser.parse_args()

    return all_args

def main(args):
    all_args = get_args(args)
    num_agents = all_args.num_agents
    num_lanes = all_args.num_lanes

    trajectory_boid_agent = t.BoidAgentActor(
        name="trajectory-boid",
        agent_locator="scenarios.straight.agent_prefabs:trajectory-boid-agent-v0",
    )

    pose_boid_agent = t.BoidAgentActor(
        name="pose-boid",
        agent_locator="scenarios.straight.agent_prefabs:pose-boid-agent-v0",
    )

    traffic = t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(begin=("west", lane_idx, 0), end=("east", lane_idx, "max"),),
                rate=50,
                actors={t.TrafficActor("car"): 1},
            )
            for lane_idx in range(num_lanes)
        ]
    )

    # missions = [
    # t.Mission(t.Route(begin=("west", 0, 10), end=("east", 0, "max"))),
    # t.Mission(t.Route(begin=("west", 1, 10), end=("east", 1, "max"))),
    # t.Mission(t.Route(begin=("west", 2, 10), end=("east", 2, "max")))
    # ]

    missions = [t.Mission(t.Route(begin=("west", mission_idx, 10), end=("east", mission_idx, "max"))) for mission_idx in range(num_agents)]

    scenario = t.Scenario(
        traffic={"all": traffic},
        ego_missions=missions,
        bubbles=[
            t.Bubble(
                zone=t.PositionalZone(pos=(50, 0), size=(40, 20)),
                margin=5,
                actor=trajectory_boid_agent,
            ),
            t.Bubble(
                zone=t.PositionalZone(pos=(150, 0), size=(50, 20)),
                margin=5,
                actor=pose_boid_agent,
                keep_alive=True,
            ),
        ],
    )

    gen_scenario(scenario, output_dir=str(Path(__file__).parent))

if __name__ == "__main__":
    main(sys.argv[1:])
