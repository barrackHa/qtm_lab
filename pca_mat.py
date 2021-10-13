import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

mat = sio.loadmat('data_files/AIM_model_movements0004.mat')
num_of_frames = mat['AIM_model_movements0004']['Frames'][0][0].flatten()[0]
delta_t_sec = 1 / num_of_frames
frame_rate = mat['AIM_model_movements0004']['FrameRate'][0][0].flatten()[0]
data = mat['AIM_model_movements0004']['Trajectories'][0][0]['Labeled'][0][0]

lables = data['Labels'][0][0][0].flatten()

lables_df = pd.DataFrame(data=lables, index=None, columns=None)
data_count = data['Count'][0][0].flatten()[0]

# data.shape -> (37, 4, 42000)
data = data['Data'][0][0]
data_df_list = [ pd.DataFrame(data=data[i], index=None, columns=None) for i in range(len(lables)) ]

print(lables.shape)
print(data[0].flatten().shape)

samples_for_pca = dict()
for i in range(len(lables)):
    samples_for_pca[lables[i][0]] = data[i].flatten()

