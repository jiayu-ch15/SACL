import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb
# data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/eval-corner-xp@50M/run3/logs/cross_play_returns.npy')
# labels=["unif", "mean_var", "individual_var", "sp_unif", "bias", "1var-1bias", "rb_var", "TDerror"]
# data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/num_ensemble@50M/run1/logs/cross_play_returns.npy')
# labels=["ensemble3", "rb_variance(ensemble1)", "ensmeble5", "3adv_var_ensemble4"]
# data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/metric@75M/run1/logs/cross_play_returns.npy')
# labels=["random", "sp_unif", "ensemble_var", "TDerror", "1var_07bias"]
data = np.load('/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/main@40M/run1/logs/cross_play_returns.npy')
labels=["sacl", "sp", "fsp", "psro"]
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
ax.set_title("Predator Reward")

plt.tight_layout()
# plt.show()
plt.savefig("../main@40M.pdf")