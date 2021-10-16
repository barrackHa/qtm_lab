from pathlib import Path
from matplotlib import colors
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load .mat file
file_name = 'Barak_test'
data_folder = Path.cwd() / Path("data_files/")
file_to_open = data_folder / file_name
mat = sio.loadmat(str(file_to_open))

# Guess table name and extract info
table_name = list(mat.keys())[-1]
M = mat[table_name]
file_timestamp = M['Timestamp'][0][0].flatten()
num_of_frames = M['Frames'][0][0].flatten()[0]
frame_rate = M['FrameRate'][0][0].flatten()[0]
delta_t_sec = 1 / frame_rate

## data's keys:
## ['Count', 'O'), ('Labels', 'O'), ('Data', 'O'), ('Type', 'O'), ('TrajectoryType', 'O')]
labeled_traj = M['Trajectories'][0][0]['Labeled'][0][0]

lables = labeled_traj['Labels'][0][0][0].flatten()
lables_df = pd.DataFrame(data=lables, index=None, columns=None)
data_count = labeled_traj['Count'][0][0].flatten()[0]

data = labeled_traj['Data'][0][0]
data_df_list = [ pd.DataFrame(data=data[i], index=None, columns=None) for i in range(len(lables)) ]

new_data = data.reshape((8, 14000))
new_df = pd.DataFrame(
    data=new_data, 
    index=['p_i_{}'.format(c) for c in 'xyzr'*2], 
    columns=None
)


print(data.shape)
print(new_data.shape)

# print(data_df_list[1].head(n=8))
print(new_df.head(n=8))