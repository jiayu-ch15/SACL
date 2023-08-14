import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb
data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_new@40M/run1/logs/cross_play_returns.npy')
# labels=["SP", "FSP"]
# data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/easy@40M/run1/logs/cross_play_returns.npy')
# labels=["SACL", "SP", "FSP", "PSRO", "R-NaD"]
# data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/hard@40M/run1/logs/cross_play_returns.npy')
labels=["SACL", "SP", "FSP", "PSRO", "NeuRD"]
num_exp = len(labels)
padding = 3
mean = np.zeros(shape=(num_exp,num_exp))
std = np.zeros(shape=(num_exp,num_exp))
annotation = np.zeros((num_exp, num_exp), dtype=object)
for i in range(mean.shape[0]):
    for j in range(mean.shape[1]):
        mean[i,j]=np.mean(data[i*padding:(i+1)*padding,j*padding:(j+1)*padding])
        std[i,j]=np.std(data[i*padding:(i+1)*padding,j*padding:(j+1)*padding])
        annotation[i,j] = f"{mean[i,j]:.2f}\n({std[i,j]:.2f})"

sns.set(rc={"figure.figsize":(5.5, 5.5)})
ax = sns.heatmap(
    data=np.log(mean), annot=annotation, fmt="s", 
    square=True, linewidths=2, cbar=False, cmap="coolwarm",
    xticklabels=labels, yticklabels=labels,
)
ax.set_title("Predator Reward")

plt.tight_layout()
# plt.show()
plt.savefig("../hard@40M.pdf")