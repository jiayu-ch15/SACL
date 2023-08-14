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
tasks = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl_greedy_heatmap/wandb/run-20230806_032953-35if43kh/files/35M/tasks_1749.npy')
use_prioritized = 0
if use_prioritized:
    weights = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl_FPS_heatmap/wandb/run-20230806_033019-1vcid8wp/files/8M/scores_399.npy')
    weights = weights / np.mean(weights)
    probs = weights / np.sum(weights)
    sample_idx = np.random.choice(len(tasks), 10000, replace=True, p=probs)
    starts = np.array([tasks[idx] for idx in sample_idx])
else:
    starts = tasks
xlabels = np.around(np.linspace(-2.1,2.1,num=grid_size), decimals=2)
ylabels = np.around(np.linspace(-2.1,2.1,num=grid_size), decimals=2)

adv_data = []
good_data = []
landmark_data = []
for agent_idx in range(num_adversaries):
    adv_data.append(starts[:, agent_idx * 4 : (agent_idx + 1) * 4 - 2])
adv_data = np.concatenate(adv_data, axis=1)
for agent_idx in range(num_good_agents):
    good_data.append(starts[:, 3 * 4 + agent_idx * 4 : 3 * 4 + (agent_idx + 1) * 4 - 2])
good_data = np.concatenate(good_data, axis=1)

adv_data_unif=np.random.uniform(1.0, 2.0, 3000).reshape(-1,2)
good_data_unif=np.random.uniform(-2.0, -1.0, 3000).reshape(-1,2)

# adv_data = np.concatenate([adv_data[:,:2], adv_data_unif],axis=0)
# good_data = np.concatenate([good_data[:,:2], good_data_unif],axis=0)

adv_data = np.concatenate([adv_data[:,:2]],axis=0)
good_data = np.concatenate([good_data[:,:2]],axis=0)

heat_map_one = np.zeros(shape=(grid_size, grid_size))
entitys = ['adv', 'good', 'landmark']
entity_num = [num_adversaries, num_good_agents, num_landmarks]

heat_map = {}
for entity_id, entity in enumerate(entitys):
    for entity_one_num in range(entity_num[entity_id]):
        heat_map[entity + '_' + str(entity_one_num)] = heat_map_one.copy()

# adv heat map
for thread in range(adv_data.shape[0]):
    index = ((adv_data[thread, :2] - start_pos) // map_delta).astype(int)
    heat_map['adv_0'][index[0],index[1]] += 1

# good heat map
for thread in range(good_data.shape[0]):
    index = ((good_data[thread, :2] - start_pos) // map_delta).astype(int)
    heat_map['good_0'][index[0],index[1]] += 1

# # landmark heat map
# for landmark_id in range(num_landmarks):
#     for thread in range(landmark_data.shape[0]):
#         index = ((landmark_data[thread, landmark_id * 2 : (landmark_id + 1) * 2] - start_pos) // map_delta).astype(int)
#         heat_map['landmark_' + str(landmark_id)][index[0],index[1]] += 1
#     heat_map['landmark_' + str(landmark_id)] = heat_map['landmark_' + str(landmark_id)]

# adv_data=np.random.uniform(1.0, 2.0, 20000).reshape(-1,2)
# heat_map = np.zeros(shape=(grid_size, grid_size))
# for idx in range(adv_data.shape[0]):
#     index = ((adv_data[idx] - start_pos) // map_delta).astype(int)
#     heat_map[index[0],index[1]] += 1

sns.set(rc={"figure.figsize":(8, 8)})
ax = sns.heatmap(
    # data=np.flip(heat_map.astype(int),axis=0), fmt="s",
    data=np.flip(heat_map['adv_0'].astype(int),axis=0), fmt="s", 
    square=True, linewidths=2, cbar=False, cmap="coolwarm",
    xticklabels=xlabels, yticklabels=np.flip(ylabels),
    vmin=0, vmax=100
)
ax.set_title("Heatmap of the predator", fontsize=22)

plt.tight_layout()
# plt.show()
plt.savefig('../greedy_35M_buffer_hm_predator.pdf')

plt.figure()
ax = sns.heatmap(
    # data=np.flip(heat_map.astype(int),axis=0), fmt="s",
    data=np.flip(heat_map['good_0'].astype(int),axis=0), fmt="s", 
    square=True, linewidths=2, cbar=False, cmap="coolwarm",
    xticklabels=xlabels, yticklabels=np.flip(ylabels),
    vmin=0, vmax=100
)
ax.set_title("Heatmap of the prey", fontsize=22)

plt.tight_layout()
# plt.show()
plt.savefig('../greedy_35M_buffer_hm_prey.pdf')