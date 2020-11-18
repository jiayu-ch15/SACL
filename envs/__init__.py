
import socket
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])
hostname = socket.gethostname()

from envs.mpe.MPE_env import MPEEnv
from envs.agar.Agar_Env import AgarEnv
from envs.hanabi.Hanabi_Env import HanabiEnv
from envs.ssd.Cleanup_Env import CleanupEnv
from envs.ssd.Harvest_Env import HarvestEnv

if hostname == "ubuntu-SYS-4028GR-TR2":
    # StarCraftII
    from envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from envs.starcraft2.smac_maps import get_map_params

    # HideAndSeek
    from envs.hns.HNS_Env import HNSEnv  

elif "c4130" in hostname or hostname == "aa-TRX40-AORUS-XTREME":
    # StarCraftII
    from envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from envs.starcraft2.smac_maps import get_map_params
    
else:
    print("your hostname is {}, and it is not included in the known list, If u want to train StarCraftII or HideAndSeek task, add your hostname to envs/__init__.py file.\n".format(hostname))

