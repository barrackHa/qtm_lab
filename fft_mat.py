from os import error
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
        self.__labels = self.labeled_traj['Labels'][0][0][0]
        self.labels_df = pd.DataFrame(data=self.labels , index=None, columns=['Lable_Name'])
        self.data_count = self.labeled_traj['Count'][0][0].flatten()[0]
        self.__data = self.labeled_traj['Data'][0][0]
        self.__velocities = None
        self.__accelerations = None
        self.__v_pca = None
        self.__v_pca_principal_components = None
        self.__fft_analysis = None
        self.data_df_list = [ 
            pd.DataFrame(data=self.data[i], index=['x','y','z','r'], columns=None) \
            for i in range(len(self.labels)
        )]
        return

    @property
    def data(self):
        """ Get a copy of the raw data """
        return self.__data.copy()

    @property
    def velocities(self):
        """ Get the velocities analysis """
        if self.__velocities is None:
            self.analys_velocities()    
        return self.__velocities

    @property
    def accelerations(self):
        """ Get the accelerations analysis """
        if self.__accelerations is None:
            self.analys_accelerations()    
        return self.__accelerations

    @property
    def v_pca(self):
        """ Get the velocities analysis """
        if self.__v_pca is None:
            self.fit_velocity_pca()    
        return self.__v_pca

    @property
    def v_pca_principal_components(self):
        """ Get the velocities data transformed into PC basis """
        if self.__v_pca is None:
            return self.fit_velocity_pca()
        return self.__v_pca_principal_components

    @property
    def labels(self):
        """ Returns a list of qtm labels in str form """
        return [l[0] for l in self.__labels]

    @property
    def derive_by_t(self):
        """Return a derivation opertor"""
        return lambda vec: np.diff(vec) / self.delta_t_sec

    def derivation_operator_factory(self):
        # TODO:// add more derivation options
        return self.derive_by_t

    def __load_mat_file__(self, file_name):
        """ Loads a .mat file into a dict """
        data_folder = Path.cwd() / Path("data_files/")
        file_to_open = data_folder / file_name
        return sio.loadmat(str(file_to_open))

    def analys_velocities(self):
        """ Analys velocities from data """
        derivation_operator = self.derivation_operator_factory()
        self.__velocities = np.array(list(map(
            lambda lbl_3d: np.array(list(map(derivation_operator, lbl_3d))),
            map(lambda lbl_trej: lbl_trej[0:3], self.__data)
        )))
        return self.__velocities
    
    def analys_accelerations(self):
        """ Analys velocities from data """
        derivation_operator = self.derivation_operator_factory()
        self.__accelerations = np.array(list(map(
            lambda lbl_v_3d: np.array(list(map(derivation_operator, lbl_v_3d))),
            map(lambda lbl_vels: lbl_vels[0:3], self.__velocities)
        )))
        return self.__accelerations

    def get_joint_velocities_decomp(self, joint):
        """
        @Param joint is ither a label or a joint index as shown by listed by self.labels
        Returns the velocities decomposition by axis for the specified joint.
        """
        try:
            [x,y,z] = self.data[joint][0:3]
        except:
            try:
                idx = self.labels.index(str(joint))
                [x,y,z] = self.data[idx][0:3]
            except:
                raise Exception(
                    '"joint" must be a valid key or key index from:\n{}'\
                        .format(self.labels)
                )
        # Return a list of velocities [v_x, v_y, v_z]
        return np.array(list(map(self.derive_by_t, [x,y,z])))

    def get_joint_accelerations_decomp(self, joint):
        """
        @Param joint is ither a label or a joint index as listed by self.labels
        Returns the accelerations decomposition by axis for the specified joint.
        """
        [v_x, v_y, v_z] = self.get_joint_velocities_decomp(joint)
        return np.array(list(map(self.derive_by_t, [v_x, v_y, v_z])))

    def fit_velocity_pca(self):
        """ Run PCA on the velocity data """
        num_of_lables, lables_dim, num_of_frames = self.velocities.shape
        pca_input = self.velocities.reshape(num_of_lables * lables_dim , num_of_frames).T
        pca_input = StandardScaler().fit_transform(pca_input)
        pca = PCA(n_components=(num_of_lables * lables_dim))
        principalComponents = pca.fit_transform(pca_input)
        self.__v_pca = pca
        self.__v_pca_principal_components = principalComponents
        return principalComponents

    def plot_v_pca_explained_variance(self, show=False):
        # TODO:// write save to file options
        explained_variance_sum = sum(self.v_pca.explained_variance_)
        pc_labales = [
            'pc_{}({}%)'.format(str(i+1), 
                int(100 * (self.v_pca.explained_variance_[i] / explained_variance_sum))
            ) \
            for i in range(len(self.v_pca.explained_variance_))
        ]
        fig = plt.figure() 
        fig.suptitle('{}'.format('PCA Analysis'))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.bar(
            pc_labales, 
            100 * (self.v_pca.explained_variance_ / explained_variance_sum)
        )
        ax1.set(
            ylabel='Percentage of variance explained\n by each of the\n selected components',
            title='PC Bars'
        )
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(
            np.arange(0, len(self.v_pca.explained_variance_)+1 ), 
            np.cumsum(
                np.concatenate((
                    np.array([0], dtype='float64'),
                    self.v_pca.explained_variance_ratio_
            ))), 
            'ro-', linewidth=2
        )
        ax2.set(title='PC Scree Plot')

        if show:
            plt.show()
        return

    def plot_v_pca_data_points(self, show=False):
        # TODO:// write save to file options and a description
        principalComponents = self.v_pca_principal_components
        X, Y = principalComponents.T[0:2]
        fig = plt.figure()
        fig.suptitle('{}'.format('Velocity Data Points In PC Space'))
        ax = fig.add_subplot(111)
        ax.set(xlabel='pc1', ylabel='pc2', xlim = [-10, 10], ylim = [-10, 10])
        ax.scatter(X,Y, s=2)
        if show:
            plt.show()
        return

    def get_fft_analysis(self, velocity=True, acceleration=False):
        fft_dict = {'velocity': {}, 'acceleration': {}}
        if velocity:
            fft_dict['velocity']['fft'] = list(map(np.abs,map(fft, self.velocities)))
            fft_dict['velocity']['fftfreq'] = fftfreq(self.velocities[0].shape[0], self.delta_t_sec)
        if acceleration:
            fft_dict['acceleration']['fft'] = list(map(np.abs,map(fft, self.accelerations)))
            fft_dict['acceleration']['fftfreq'] =  fftfreq(self.accelerations[0].shape[0], self.delta_t_sec)
        self.__fft_analysis = fft_dict
        print()
        return fft_dict

    def plot_fft(self, markers):
        l = len(self.labels) - 1
        print(self.__fft_analysis['velocity']['fft'][0][0].shape)
        print(self.__fft_analysis['velocity']['fftfreq'])
        for mark in set(markers):
            mark = int(mark)
            if mark < 0 or l < mark :
                raise Exception(
                    'Marker index must be between 0 and {} but got {}'.
                    format(l, mark)
                )
            fig = plt.figure() 
            fig.add_axes() 
            ax = fig.add_subplot(111)
            ax.plot(
                self.__fft_analysis['velocity']['fftfreq'],
                self.__fft_analysis['velocity']['fft'][0][0],
                plot_linewidth=0.5
            )
        plt.show()
        return

if __name__ == '__main__':
    # file_name = 'circ_motion_1D_2labls'
    # file_name = 'Barak_test'
    file_name = 'x_axis_broom_barak'
    qtm = QTM(file_name)
    data = qtm.data
    
    delta_t_sec = qtm.delta_t_sec
    lables = qtm.labels

    joint_index = 0
    marker = data[joint_index]
    # marker_df = pd.DataFrame(data=marker.T[0], index=['x','y','z','r'], columns=None)

    # Split by axes
    qtm.get_joint_velocities_decomp(joint=0)
    arr = [x,y,z] = qtm.data[joint_index][0:3]

    # Compute local velocity and acceleration acoording to :
    # velocity = dx/dt, acceleration = dv/dt
    derive_by_t = lambda vec:  np.diff(vec) / delta_t_sec
    velocities = [v_x, v_y, v_z] = list(map(derive_by_t, [x,y,z]))
    tmp = [vt_x, vt_y, vt_z] = qtm.get_joint_velocities_decomp(joint_index)
    
    new_velocities = qtm.analys_velocities()
    # print(qtm.data)
    # print(new_velocities)
    # print((qtm.velocities[joint_index] == np.array(tmp)).all())

    accelerations = [a_x, a_y, a_z] = list(map(derive_by_t, (v_x, v_y, v_z)))
    [at_x, at_y, at_z] = qtm.get_joint_accelerations_decomp(joint_index)

    # print(a_x == at_x, a_y == at_y, (a_z == at_z).all())
    # Fourier 
    v_fft_decomp = list(map(np.abs, map(fft, velocities)))
    # class_v_fft_decomp = qtm.get_fft_analysis(joint_index, 'v')
    # print(
    #     (v_fft_decomp[0] == class_v_fft_decomp[0]).all(),
    #     (v_fft_decomp[1] == class_v_fft_decomp[1]).all(),
    #     (v_fft_decomp[2] == class_v_fft_decomp[2]).all()
    # )

    # print(
        # class_v_fft_decomp[0],
        # class_v_fft_decomp[1],
        # class_v_fft_decomp[2]
    #     vt_x, vt_y, vt_z, tmp.T
    # )

    a_fft_decomp = list(map(np.abs,map(fft, accelerations)))

    # print(len(qtm.labels))
    lables_dim = qtm.data.shape[1]
    num_of_lables = qtm.data.shape[0]

    # print(qtm.data.reshape(num_of_lables * lables_dim , int(qtm.num_of_frames)).T)
    print(qtm.data)
    qtm.fit_velocity_pca()
    qtm.plot_v_pca_explained_variance(show=True)
    qtm.plot_v_pca_data_points(show=True)
    
    # fft_analysis = qtm.get_fft_analysis(velocity=True, acceleration=False)
    # qtm.plot_fft([0])
    exit()
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

