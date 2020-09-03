
from .starcraft2.StarCraft2 import StarCraft2Env
from .starcraft2.smac_maps import get_map_params
from .agar.Env import AgarEnv
from .ssd.Cleanup import CleanupEnv
from .ssd.Harvest import HarvestEnv
from .hanabi.rl_env import HanabiEnv
from .mpe.MPE import MPEEnv
from .hns.envs.hide_and_seek import HideAndSeekEnv
from .hns.envs.blueprint_construction import BlueprintConstructionEnv
from .hns.envs.box_locking import BoxLockingEnv
from .hns.envs.shelter_construction import ShelterConstructionEnv

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

