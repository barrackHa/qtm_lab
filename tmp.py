import numpy as np
from fft_mat import QTM

# file_name = 'Barak_test'
file_name = 'x_axis_broom_barak'
# file_name = 'test'
qtm = QTM(file_name)
print(qtm.v_pca.components_.T)
print(qtm.v_pca.explained_variance_ratio_)
print(qtm.get_fft_analysis())
qtm.plot_v_pca_data_points(show=False)
qtm.plot_v_pca_explained_variance(show=True)