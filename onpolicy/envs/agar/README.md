# Agar.io

## 1. Introduction

Agar is a online game where players can control cells using a cursor to move ,  "space" to split and "w" to eject.  When cell A is 1.15 times larger than B, A is able to eat B and get all his mass. However, a cell would move slower and slower as he grows, so he can split toward small cells to get a rush speed which helps him to eat them. On top of split, cells can coopertate to eat other players. The more you eat, the bigger you grow, the higher score you get. We hope our agents can cooperate with each other and get a best score.

## 2. Observation space

## 2.1 discription of cells around the agent

    1. position: absolute position of the cell.(2D)
    
    2. relative_position: relative position of the cell.(2D)
    
    3. v: speed of the cell(2D)
    
    4. boost_x: additional speed a cell gets when he split towards .(2D)
    
    5. radius, log_radius: radius of cell.(1D)
    
    6. canRemerge: whether the cell can remerge immediately.(1D)
    
    7. d: the distance between cell and the center of the agent, that is relative_position_x ** 2 + relative_position_y ** 2(1D)
    
    8. eaten_max: whether the ball can be eaten by the maximal balls of the observing agent(1D)
    
    9.eaten_min: whether the ball can be eaten by the minimal balls of the observing agent(1D)

### 2.2 global information of the agent

	1. center_position: when agent splites into several cells, it means the center position of these cells.(2D)
	
	2. center_speed: speed of its center(2D)
	
	3. bound: the length and width of the agent’s observation scope(1D)
	
	4. maxcell: the radius of the biggest cell(1D)
	
	5. mincell: the radius of smallest cell(1D)
	
	6. last_action: the last action of agent(3D)
	
	7. bot_speed: the speed of bots(script-based agents) (1D)
	
	8. killed: whether the other learn-based agent is killed.(1D)
	
	9. num_type: there are 3 typies of agent, num_type denotes amounts of them(1D)

## 3. action space

   1. move: move in x and y direction 

   2. split: cells larger than a threhold can split towards a target and get a rush ahead for a short while

## 4. reward

   1. mass_reward: increase of cells’ mass.

   2. Kill_reward: reward of kill(set to 0)

   3. Killed_reward: reward of being killed(set to 0)
