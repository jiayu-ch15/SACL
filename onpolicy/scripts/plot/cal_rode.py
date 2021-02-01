from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
from matplotlib.pyplot import MultipleLocator
import pandas

def read_tensorboard_data(tensorboard_path, val_name):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    val = ea.scalars.Items(val_name)
    return val
    
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            L.append(os.path.join(root, file))  
    return L  

if __name__ == "__main__":
    map_names = ['2s_vs_1sc','2s3z',\
'5m_vs_6m','3s5z','1c3s5z','27m_vs_30m','bane_vs_bane','3s_vs_5z','6h_vs_8z',\
'corridor','3s5z_vs_3s6z','10m_vs_11m','MMM2','2c_vs_64zg']

    for map_name in map_names:
        ####################################RODE##############################
        val_name = 'test_battle_won_mean'
        tensorboard_path = './RODE/' + map_name + '/tb_logs/'

        data_list = file_name(tensorboard_path)
        
        win_rate = []

        for dl in data_list:
            val = read_tensorboard_data(dl, val_name)
            y = []
            for v in val:
                y.append(v.value)
            win_rate.append(np.array(y))

        y_seed = []
        for y in win_rate:
            y_seed.append(np.median(y[-10:]))

        y_seed = np.array(y_seed)

        median_seed = np.median(y_seed, axis=0)
        std_seed = np.std(y_seed, axis=0)

        print(map_name)
        print(median_seed)
        print(std_seed)
        


 