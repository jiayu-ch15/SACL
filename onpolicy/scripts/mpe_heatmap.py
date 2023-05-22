import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb

boundary_max = 2.1
grid_size = 20
num_adversaries = 3
num_good_agents = 1
num_landmarks = 2
map_delta = 2 * boundary_max / grid_size
start_pos = np.array([- boundary_max, - boundary_max])
offset_velocity = (num_adversaries + num_good_agents) * 2
starts = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/task_heatmap/run1/models/tasks_0.npy')
xlabels = np.around(np.linspace(-2.1,2.1,num=grid_size), decimals=2)
ylabels = np.around(np.linspace(-2.1,2.1,num=grid_size), decimals=2)

adv_data = starts[:,offset_velocity : offset_velocity + num_adversaries * 2]
good_data = starts[:,offset_velocity + num_adversaries * 2 : offset_velocity + num_adversaries * 2 + num_good_agents * 2]
landmark_data = starts[:, offset_velocity + num_adversaries * 2 + num_good_agents * 2 : offset_velocity + num_adversaries * 2 + num_good_agents * 2 + num_landmarks * 2]

heat_map_one = np.zeros(shape=(grid_size, grid_size))
entitys = ['adv','good', 'landmark']
entity_num = [num_adversaries, num_good_agents, num_landmarks]

heat_map = {}
for entity_id, entity in enumerate(entitys):
    for entity_one_num in range(entity_num[entity_id]):
        heat_map[entity + '_' + str(entity_one_num)] = heat_map_one.copy()

# adv heat map
for adv_id in range(num_adversaries):
    for thread in range(adv_data.shape[0]):
        index = ((adv_data[thread, adv_id * 2 : (adv_id + 1) * 2] - start_pos) // map_delta).astype(int)
        heat_map['adv_' + str(adv_id)][index[0],index[1]] += 1
    heat_map['adv_' + str(adv_id)] = heat_map['adv_' + str(adv_id)]

# good heat map
for good_id in range(num_good_agents):
    for thread in range(good_data.shape[0]):
        index = ((good_data[thread, good_id * 2 : (good_id + 1) * 2] - start_pos) // map_delta).astype(int)
        heat_map['good_' + str(good_id)][index[0],index[1]] += 1
    heat_map['good_' + str(good_id)] = heat_map['good_' + str(good_id)]

# landmark heat map
for landmark_id in range(num_landmarks):
    for thread in range(landmark_data.shape[0]):
        index = ((landmark_data[thread, landmark_id * 2 : (landmark_id + 1) * 2] - start_pos) // map_delta).astype(int)
        heat_map['landmark_' + str(landmark_id)][index[0],index[1]] += 1
    heat_map['landmark_' + str(landmark_id)] = heat_map['landmark_' + str(landmark_id)]

# adv_data=np.random.uniform(1.0, 2.0, 20000).reshape(-1,2)
# heat_map = np.zeros(shape=(grid_size, grid_size))
# for idx in range(adv_data.shape[0]):
#     index = ((adv_data[idx] - start_pos) // map_delta).astype(int)
#     heat_map[index[0],index[1]] += 1

sns.set(rc={"figure.figsize":(8, 8)})
ax = sns.heatmap(
    data=np.flip(heat_map['adv_0'].astype(int),axis=0), fmt="s", 
    square=True, linewidths=2, cbar=False, cmap="coolwarm",
    xticklabels=xlabels, yticklabels=np.flip(ylabels),
    # vmin=10
)
ax.set_title("Heatmap of the predator", fontsize=22)

plt.tight_layout()
# plt.show()
plt.savefig('sacl_hm.pdf')