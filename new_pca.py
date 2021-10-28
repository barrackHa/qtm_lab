from pathlib import Path
from matplotlib import colors
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load .mat file
file_name = 'test.mat'
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

# Every marker (label) has x,y and z coordinates and an error estimate - r 
comp_per_marker = 'xyzr'

## data's keys:
labeled_traj = M['Trajectories'][0][0]['Labeled'][0][0]

lables = labeled_traj['Labels'][0][0][0].flatten()
lables_df = pd.DataFrame(data=lables, index=None, columns=None)
data_count = labeled_traj['Count'][0][0].flatten()[0]

data = labeled_traj['Data'][0][0]
data_df_list = [ pd.DataFrame(data=data[i], index=None, columns=None) for i in range(len(lables)) ]

new_data = data.reshape((len(lables)*4 , int(num_of_frames)))
# print(new_data.shape, data.shape)
# print(new_data.T)
new_df = pd.DataFrame(
    data=new_data.T, 
    index=None, 
    columns=np.array(
        [[ 
            '{}_{}'.format(i[0], c) for c in comp_per_marker] 
            for i in lables ]
    ).flatten()
)

derive_by_t = lambda vec:  np.diff(vec) / delta_t_sec
# velocities = [v_x, v_y, v_z] = list(map(derive_by_t, [x,y,z]))
# accelerations = [a_x, a_y, a_z] = list(map(derive_by_t, (v_x, v_y, v_z)))

# print(new_data[3].shape)

# print(data_df_list[1].head(n=8))

# print(new_df.iloc[0].to_numpy())

velocity_indices = []
for l in lables:
    for c in comp_per_marker[:-1]:
        l = l[0]
        new_cul_name, cul_to_derive = l + '_V' + c, l + '_' + c 
        new_df[new_cul_name] = new_df[cul_to_derive].diff().div(delta_t_sec)
        velocity_indices.append(new_cul_name)

# print(new_df[velocity_indices].values[1:])

# print(DataFrame(data = (new_df['A_v'] / new_df['B_v'])).describe())
X = new_df[velocity_indices].values[1:]
print(X)
# Standardizing the features

X = StandardScaler().fit_transform(X)

exit()
pca = PCA(n_components=len(velocity_indices))
principalComponents = pca.fit_transform(X)

print(principalComponents.shape)
pca_df = pd.DataFrame(data=principalComponents) #, columns=pc_labales)
print(pca_df.head())

explained_variance_sum = sum(pca.explained_variance_)
pc_labales = [
    'pc_{}({}%)'.format(str(i+1), 
        int(100 * (pca.explained_variance_[i] / explained_variance_sum))
    ) \
    for i in range(len(pca.explained_variance_))
]
# print(new_df[velocity_indices].head())

# print(pd.DataFrame(data=X, columns=velocity_indices).head())

print(new_df[velocity_indices].head())

# print(pca.components_)
# print(pca.explained_variance_)
plt.bar(
    pc_labales, 
    100 * (pca.explained_variance_ / explained_variance_sum)
)

fig = plt.figure()
plt.plot(
    np.arange(1, len(pca.explained_variance_)+1 ), 
    np.cumsum(pca.explained_variance_), 
    'ro-', linewidth=2
)
plt.title('Scree Plot')

pca_df.plot.scatter(x=pc_labales[0], y=pc_labales[1])
plt.show()

