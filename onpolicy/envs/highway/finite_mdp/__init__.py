from gym.envs.registration import register

register(
    id='finite-mdp-v0',
    entry_point='finite_mdp.envs:FiniteMDPEnv',
)