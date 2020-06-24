# MAPPO-SC (Multi-agent PPO for StarCraftII)

## 1.Install

```Bash
git clone https://github.com/zoeyuchao/mappo-sc.git
cd ~/mappo-sc
conda create -n mappo-sc python==3.6.2
conda activate mappo-sc
pip install -r requirements.txt
```

## 2. Train StarCraft

1. Download StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   ```Bash
   unzip SC2.4.10.zip
   # password is iagreetotheeula
   echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
   ```

   If you want stable id, you can copy the `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

2. Enjoy 

- config.py: all hyper-parameters
  
  - default: use cuda, GRU and share policy
  
- train.py: all train code

  - Here is an example:

  ```Bash
  conda activate mappo-sc
  python train.py --map_name="3m" -algorithm_name="mappo" 
  ```

  - You can use tensorboardX to see the training curve in fold `results`:
  
  ```Bash
  tensorboard --logdir=./results/ 
  ```

