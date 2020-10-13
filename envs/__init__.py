
from envs.starcraft2.StarCraft2 import StarCraft2Env
from envs.starcraft2.smac_maps import get_map_params
from envs.agar.Env import AgarEnv
from envs.ssd.Cleanup import CleanupEnv
from envs.ssd.Harvest import HarvestEnv
from envs.hanabi.rl_env import HanabiEnv
from envs.mpe.MPE import MPEEnv
from envs.hns.envs.hide_and_seek import HideAndSeekEnv
from envs.hns.envs.blueprint_construction import BlueprintConstructionEnv
from envs.hns.envs.box_locking import BoxLockingEnv
from envs.hns.envs.shelter_construction import ShelterConstructionEnv

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

