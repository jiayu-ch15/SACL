import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb

episode = 0
# model_dir='3v1_1var/wandb/run-20230505_150104-25sa3piw'
model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/3v1_debug/run1/models"
task = np.load(model_dir + '/tasks_{}.npy'.format(episode))
score = np.load(model_dir + '/scores_{}.npy'.format(episode))

# 3v1
ball_x_y = task[:,:2]
left_GK = task[:, 3:5]
left_1 = task[:,5:7]
left_2 = task[:,7:9]
left_3 = task[:,9:11]
right_GK = task[:, 11:13]
right_1 = task[:, 13:]

# pass_shoot
# ball_x_y = task[:,:2]
# left_GK = task[:, 3:5]
# left_1 = task[:,5:7]
# left_2 = task[:,7:9]
# right_GK = task[:, 9:11]
# right_1 = task[:, 11:]

# groundtruth
# 3v1
task_truth = np.array([0.62,0.0,0.0,-1.0,0.0,0.6,0.0,0.7,0.2,0.7,-0.2, -1.0, 0.0, -0.75, 0.0])
# pass_shoot
task_truth = np.array([0.7, -0.28, 0.0, -1.0, 0.0,0.7, 0.0,0.7, -0.3,-1.0, 0.0, -0.75, 0.3])
# run_pass_shoot
task_truth = np.array([0.7, -0.28, 0.0, -1.0, 0.0,0.7, 0.0,0.7, -0.3,-1.0, 0.0, -0.75, 0.1])

def plot_heatmap(task, episode, model_dir, name='ball', task_truth=np.array([0.62,0.0])):
    num = 20
    x_max = np.max(task[:,0],axis=0)
    x_min = np.min(task[:,0],axis=0)
    y_max = np.max(task[:,1],axis=0)
    y_min = np.min(task[:,1],axis=0)
    x = np.linspace(start=x_min,stop=x_max,num=num,endpoint=True)
    y = np.linspace(start=y_min,stop=y_max,num=num,endpoint=True)
    heatmap = np.zeros(shape=(num,num))
    for task_one in task:
        index_x = 0
        for x_idx in range(len(x)-1):
            if task_one[0] >= x[x_idx] and task_one[0] < x[x_idx + 1]:
                index_x = x_idx
                break
        for y_idx in range(len(y)-1):
            if task_one[1] >= y[y_idx] and task_one[1] < y[y_idx + 1]:
                index_y = y_idx
                break
        heatmap[index_x,index_y] += 1
    
    xlabels = ['x={}'.format(round(xlabel,2)) for xlabel in x]
    ylabels = ['y={}'.format(round(ylabel,2)) for ylabel in y]

    sns.set(rc={"figure.figsize":(8, 8)})
    ax = sns.heatmap(
        data=heatmap.T, fmt="s", 
        square=True, linewidths=2, cbar=False, cmap="coolwarm",
        xticklabels=xlabels, yticklabels=ylabels,
    )
    ax.set_title("Heatmap of the {}".format(name))

    plt.tight_layout()
    # plt.show()
    plt.savefig(model_dir + "/{}@{}.png".format(name,episode))

    heatmap_truth = np.zeros(shape=(num,num))
    x_truth_in_task = False
    for x_idx in range(len(x)-1):
        if task_truth[0] > x[x_idx] and task_truth[0] <= x[x_idx + 1]:
            index_x = x_idx
            x_truth_in_task = True
            break
    for y_idx in range(len(y)-1):
        if task_truth[1] > y[y_idx] and task_truth[1] <= y[y_idx + 1]:
            index_y = y_idx
            break
    if x_truth_in_task:
        heatmap_truth[index_x,index_y] = 10000
    else:
        heatmap_truth[0, index_y] = 10000

    # groundtruth
    sns.set(rc={"figure.figsize":(8, 8)})
    ax = sns.heatmap(
        data=heatmap_truth.T, fmt="s", 
        square=True, linewidths=2, cbar=False, cmap="coolwarm",
        xticklabels=xlabels, yticklabels=ylabels,
        vmin=10
    )
    ax.set_title("Heatmap of the {}_groundtruth".format(name))

    plt.tight_layout()
    # plt.show()
    plt.savefig(model_dir + "/{}_groundtruth@{}.png".format(name,episode))

# pass shoot 
# plot_heatmap(task=ball_x_y,episode=episode, model_dir=model_dir,name='ball', task_truth=np.array([0.7, -0.28])) # appro 0.63
# plot_heatmap(task=left_GK,episode=episode, model_dir=model_dir,name='left_GK', task_truth=np.array([-1.0,0.0]))
# plot_heatmap(task=left_1,episode=episode, model_dir=model_dir,name='left_1', task_truth=np.array([0.7,0.0]))
# plot_heatmap(task=left_2,episode=episode, model_dir=model_dir,name='left_2', task_truth=np.array([0.7,-0.3]))
# plot_heatmap(task=right_GK,episode=episode, model_dir=model_dir,name='right_GK', task_truth=np.array([1.0,0.0]))
# plot_heatmap(task=right_1,episode=episode, model_dir=model_dir,name='right_1', task_truth=np.array([0.75,-0.3]))


# run pass shoot
# plot_heatmap(task=ball_x_y,episode=episode, model_dir=model_dir,name='ball', task_truth=np.array([0.7, -0.28])) # appro 0.63
# plot_heatmap(task=left_GK,episode=episode, model_dir=model_dir,name='left_GK', task_truth=np.array([-1.0,0.0]))
# plot_heatmap(task=left_1,episode=episode, model_dir=model_dir,name='left_1', task_truth=np.array([0.7,0.0]))
# plot_heatmap(task=left_2,episode=episode, model_dir=model_dir,name='left_2', task_truth=np.array([0.7,-0.3]))
# plot_heatmap(task=right_GK,episode=episode, model_dir=model_dir,name='right_GK', task_truth=np.array([1.0,0.0]))
# plot_heatmap(task=right_1,episode=episode, model_dir=model_dir,name='right_1', task_truth=np.array([0.75,-0.1]))

# 3v1
plot_heatmap(task=ball_x_y,episode=episode, model_dir=model_dir,name='ball', task_truth=np.array([0.6, 0.0])) # appro 0.63
plot_heatmap(task=left_GK,episode=episode, model_dir=model_dir,name='left_GK', task_truth=np.array([-1.0,0.0]))
plot_heatmap(task=left_1,episode=episode, model_dir=model_dir,name='left_1', task_truth=np.array([0.6,0.0]))
plot_heatmap(task=left_2,episode=episode, model_dir=model_dir,name='left_2', task_truth=np.array([0.7,0.2]))
plot_heatmap(task=left_3,episode=episode, model_dir=model_dir,name='left_3', task_truth=np.array([0.7,-0.2]))
plot_heatmap(task=right_GK,episode=episode, model_dir=model_dir,name='right_GK', task_truth=np.array([1.0,0.0]))
plot_heatmap(task=right_1,episode=episode, model_dir=model_dir,name='right_1', task_truth=np.array([0.75,0.0]))