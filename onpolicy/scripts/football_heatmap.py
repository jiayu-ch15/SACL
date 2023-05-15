import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb
data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/ps@sacl_vs_sp_deterministic_new@40M/run1/logs/cross_play_win_rate.npy')
# labels=["sacl", "sp"]
labels=["sacl", "sp"]
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

sns.set(rc={"figure.figsize":(8, 8)})
ax = sns.heatmap(
    data=mean, annot=annotation, fmt="s", 
    square=True, linewidths=2, cbar=False, cmap="coolwarm",
    xticklabels=labels, yticklabels=labels,
)
ax.set_title("Red Reward")

plt.tight_layout()
# plt.show()
plt.savefig("../ps_deterministic_new@40M.pdf")