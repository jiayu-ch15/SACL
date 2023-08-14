import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

boundary_max = 2.1
grid_size = 20
num_adversaries = 3
num_good_agents = 1
num_landmarks = 2
map_delta = 2 * boundary_max / grid_size
start_pos = np.array([- boundary_max, - boundary_max])
offset_velocity = (num_adversaries + num_good_agents) * 2
FPS_tasks = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl_FPS_heatmap/wandb/run-20230805_142527-bjelts70/files/35M/tasks_1749.npy')
greedy_tasks = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl_greedy_heatmap/wandb/run-20230805_143528-1wk813cj/files/35M/tasks_1749.npy')
random_tasks = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl_random_heatmap/wandb/run-20230805_143138-ahknjc7y/files/35M/tasks_1749.npy')

# tsne = TSNE(n_components=2)
pca = PCA(n_components=2)
tasks_space = np.random.uniform(-2,2,size=(10000,FPS_tasks.shape[-1]))
adv1_pos = np.random.uniform(-2,2,size=(10000, 2))
adv1_vel = np.random.uniform(-1,1,size=(10000, 2))
adv2_pos = np.random.uniform(-2,2,size=(10000, 2))
adv2_vel = np.random.uniform(-1,1,size=(10000, 2))
adv3_pos = np.random.uniform(-2,2,size=(10000, 2))
adv3_vel = np.random.uniform(-1,1,size=(10000, 2))
good_pos = np.random.uniform(-2,2,size=(10000, 2))
good_vel = np.random.uniform(-1.3,1.3,size=(10000, 2))
landmark1_pos = np.random.uniform(-2,2,size=(10000, 2))
landmark2_pos = np.random.uniform(-2,2,size=(10000, 2))
tasks_space = np.concatenate([adv1_pos, adv1_vel, adv2_pos, adv2_vel,
                              adv3_pos, adv3_vel, good_pos, good_vel,
                              landmark1_pos, landmark2_pos
                              ], axis=1)
pca.fit(tasks_space)
tasks_space = pca.transform(tasks_space)
FPS_tasks = pca.transform(FPS_tasks[:,:-1])
greedy_tasks = pca.transform(greedy_tasks[:,:-1])
random_tasks = pca.transform(random_tasks[:,:-1])
plt.style.use("ggplot")
p = sns.color_palette()
f, axes = plt.subplots(1, 1, figsize=(5, 5))
plt.scatter(tasks_space[:,0], tasks_space[:,1], s=1, color='red', marker='o', label='state space')
# plt.scatter(FPS_tasks[:,0], FPS_tasks[:,1], s=1, color='cornflowerblue', marker='o', label='state buffer')
plt.scatter(greedy_tasks[:,0], greedy_tasks[:,1], s=1, color='cornflowerblue', marker='o', label='state buffer')
# plt.scatter(random_tasks[:,0], random_tasks[:,1], s=1, color='cornflowerblue', marker='o', label='state buffer')

plt.legend(loc='upper left', fontsize=15)
plt.tight_layout()
# plt.show()
plt.savefig('../greedy_PCA.pdf')