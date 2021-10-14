from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


class QTM():
    """
    A class representing a QTM model  given by a .MAT file
    """
    def __init__(self, file_name):
        self.mat = self.__load_mat_file__(file_name)

        # Guess table name and extract info
        self.table_name = list(self.mat.keys())[-1]
        M = self.M = self.mat[self.table_name]
        self.file_timestamp = M['Timestamp'][0][0].flatten()
        self.num_of_frames = M['Frames'][0][0].flatten()[0]
        self.frame_rate = M['FrameRate'][0][0].flatten()[0]
        self.delta_t_sec = 1 / self.frame_rate
        self.labeled_traj = M['Trajectories'][0][0]['Labeled'][0][0]
        self.lables = self.labeled_traj['Labels'][0][0][0].flatten()
        self.lables_df = pd.DataFrame(data=self.lables, index=None, columns=None)
        self.data_count = self.labeled_traj['Count'][0][0].flatten()[0]
        self.data = self.labeled_traj['Data'][0][0]
        self.data_df_list = [ 
            pd.DataFrame(data=self.data[i], index=None, columns=None) \
            for i in range(len(self.lables)
        )]



    def __load_mat_file__(self, file_name):
        """
        Loads a .mat file into a dict 
        """
        data_folder = Path.cwd() / Path("data_files/")
        file_to_open = data_folder / file_name
        return sio.loadmat(str(file_to_open))


if __name__ == '__main__':
    file_name = 'Barak_test'
    qtm = QTM(file_name)
    data = qtm.data
    delta_t_sec = qtm.delta_t_sec
    lables = qtm.lables

    joint_index = 0
    marker = data[joint_index]
    marker_df = pd.DataFrame(data=marker.T[0], index=None, columns=None)

    # Split by axes
    [x,y,z] = data[0][0:3]

    # Compute local velocity and acceleration acoording to :
    # velocity = dx/dt, acceleration = dv/dt
    derive_by_t = lambda vec:  np.diff(vec) / delta_t_sec
    velocities = [v_x, v_y, v_z] = list(map(derive_by_t, [x,y,z]))
    accelerations = [a_x, a_y, a_z] = list(map(derive_by_t, (v_x, v_y, v_z)))

    # Furier 
    v_fft_decomp = list(map(np.abs, map(fft, velocities)))
    a_fft_decomp = list(map(abs,map(fft, accelerations)))

    # Create plots
    dim = 3
    plot_linewidth = 0.5
    fig, axs = plt.subplots(4, dim, sharex=False)
    fig.suptitle('{}'.format(lables[joint_index]))
    t_v_f = fftfreq(velocities[0].shape[0], delta_t_sec)
    t_a_f = fftfreq(accelerations[0].shape[0], delta_t_sec)

    for i in range(dim): 
        axs[0,i].plot(velocities[i], linewidth=plot_linewidth)
        axs[1,i].plot(
            t_v_f,
            v_fft_decomp[i], 
            c='green', linewidth=plot_linewidth
        )
        axs[2,i].plot(accelerations[i], c='orange', linewidth=plot_linewidth)
        axs[3,i].plot(
            t_a_f,
            a_fft_decomp[i], 
            c='red', linewidth=plot_linewidth
        )

        axs[0,i].set_title(['X component','Y component','Z component'][i])
        
    axs[0,0].set(ylabel='velocity')
    axs[1,0].set(ylabel='v_fft_decomp')
    axs[2,0].set(ylabel='acceleration')
    axs[3,0].set(ylabel='a_fft_decomp')

    plt.show()
    
    # syntax for 3-D projection
    # nax = plt.axes(projection ='3d')
    # nax.scatter(x, y, z, s=0.1)
    # plt.show()

