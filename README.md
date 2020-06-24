# MAPPO-SC (Multi-agent PPO for StarCraftII)

## 1.Install

```Bash
git clone https://github.com/zoeyuchao/mappo-sc.git
cd ~/mappo-sc
conda create -n mappo-sc python==3.6.2
conda activate mappo-sc
pip install -r requirements.txt
```

## 2.Train

- config.py: all hyper-parameters

  - default: use cuda, GRU and share policy

- train.py: all train code

  an example:

  ```Bash
  conda activate mappo-sc
  python train.py --map_name="3m" -algorithm_name="mappo" 
  ```

  

