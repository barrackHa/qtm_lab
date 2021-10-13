import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('data_files/AIM_model_movements0004.mat')

# print(mat['AIM_model_movements0004'].dtype)

# print(mat['AIM_model_movements0004']['File'])
file_timestamp = mat['AIM_model_movements0004']['Timestamp'][0][0].flatten()
# print(file_timestamp)
# print(mat['AIM_model_movements0004']['StartFrame'])
num_of_frames = mat['AIM_model_movements0004']['Frames'][0][0].flatten()[0]
delta_t_sec = 1 / num_of_frames
# print(type(delta_t_sec))
frame_rate = mat['AIM_model_movements0004']['FrameRate'][0][0].flatten()[0]
# print(frame_rate)
# print(mat['AIM_model_movements0004']['Events'])
# print(mat['AIM_model_movements0004']['Trajectories'][0][0]['Labeled'][0][0].dtype)
# print(mat['AIM_model_movements0004']['Trajectories'][0][0]['Unidentified'])

data = mat['AIM_model_movements0004']['Trajectories'][0][0]['Labeled'][0][0]
# 'Count', 'O'), ('Labels', 'O'), ('Data', 'O'), ('Type', 'O'), ('TrajectoryType', 'O')]
lables = data['Labels'][0][0][0].flatten()
# print(len(lables))
lables_df = pd.DataFrame(data=lables, index=None, columns=None)
data_count = data['Count'][0][0].flatten()[0]
# print(data_count)
# print(data['Data'][0][0][0][:][0].shape)

# print(data['Type'][0][0].shape)
# print(data['TrajectoryType'][0][0])

# data.shape -> (37, 4, 42000)
data = data['Data'][0][0]
data_df_list = [ pd.DataFrame(data=data[i], index=None, columns=None) for i in range(len(lables)) ]
# print(data_df[0][0])
# data = data.reshape(42000,37,4)
marker = data[1]
tmp = pd.DataFrame(data=marker.T[0], index=None, columns=None)
# print(tmp.head())
# print(data[1][1][0])
# print(data[1][2][0])
# print(data_df_list[0].head())

# fig = plt.figure()
# x,y,z,c = data_df[0]
# # syntax for 3-D projection
# ax = plt.axes(projection ='3d')
# ax.scatter(x, y, z, c=c)

# plt.show()