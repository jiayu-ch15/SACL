
import socket
hostname = socket.gethostname()

from envs.mpe.MPE_env import MPEEnv
from envs.agar.Agar_Env import AgarEnv
from envs.hanabi.Hanabi_Env import HanabiEnv
from envs.ssd.Cleanup_Env import CleanupEnv
from envs.ssd.Harvest_Env import HarvestEnv

if hostname == "ubuntu-SYS-4028GR-TR2":
    from envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from envs.starcraft2.smac_maps import get_map_params
    from envs.hns.HNS_Env import HNSEnv
    
    from envs.hns.envs.hide_and_seek import HideAndSeekEnv
    from envs.hns.envs.blueprint_construction import BlueprintConstructionEnv
    from envs.hns.envs.box_locking import BoxLockingEnv
    from envs.hns.envs.shelter_construction import ShelterConstructionEnv

    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(['train_sc.py'])
elif "c4130" in hostname:
    from envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from envs.starcraft2.smac_maps import get_map_params
    
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(['train_sc.py'])

