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


colors = ['blue', 'red']
all_v = np.array([[],[],[]])
ax = plt.axes(projection ='3d')
for marker in data:
    marker_df = pd.DataFrame(data=marker.T[0], index=None, columns=None)

    # Split by axes
    [x,y,z] = data[0][0:3]

    # Compute local velocity and acceleration acoording to :
    # velocity = dx/dt, acceleration = dv/dt
    derive_by_t = lambda vec:  np.diff(vec) / delta_t_sec
    velocities = [v_x, v_y, v_z] = list(map(derive_by_t, [x,y,z]))
    all_v = np.array([
        np.concatenate((all_v[0], v_x)), 
        np.concatenate((all_v[1], v_y)), 
        np.concatenate((all_v[2], v_z))
    ])

    ax.scatter(v_x, v_y, v_z, s=0.1, c=colors.pop())

pca = PCA(n_components=2)
x = StandardScaler().fit_transform(all_v)
principalComponents = pca.fit_transform(x)
print(principalComponents)
plt.show()
