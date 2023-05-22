import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb
data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/3v1@100M_True/run1/logs/cross_play_win_rate.npy')
labels=["sacl", "sp", "fsp", "psro", "NeuRD"]
# labels=["sacl", "psro"]
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
    xticklabels=labels, yticklabels=labels, annot_kws={"fontsize":20}
)
ax.set_title("Win rate of the red team", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
# plt.show()
plt.savefig("../3v1@100M.pdf")