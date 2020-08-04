
from .starcraft2.StarCraft2 import StarCraft2Env
from .starcraft2.smac_maps import get_map_params
from .agar.Env import AgarEnv
from .ssd.Cleanup import CleanupEnv
from .ssd.Harvest import HarvestEnv

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

