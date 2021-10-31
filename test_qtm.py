import pytest
import numpy as np
from fft_mat import QTM

def test_pass():
    assert 1 == 1

def test_fails():
    with pytest.raises(AssertionError):
        assert 1 == 2

@pytest.fixture
def mat_4_by_4_file():
    return QTM('f')

@pytest.fixture
def circ_motion_1D_2labls_mat():
    """
    circ_motion_1D_2labls_mat a label starting at [0,0,0]. It moves only on 'x' axis,
    1 mm per frame for 4 frames to [4,0,0] and than goes back to [-4,0,0] at the same pace.
    So, the velocity is 700 mm per sec in ither the positive or negative direction.
    The raltion between the labels is: B = A + [1,0,0] 
    """
    return QTM('circ_motion_1D_2labls')

def test_4by4_shape(mat_4_by_4_file):
    # print(mat_4_by_4_file.data.shape)
    assert mat_4_by_4_file.data.shape == (2,4,10)

def test_velocity_pca(circ_motion_1D_2labls_mat):
    test_mat = circ_motion_1D_2labls_mat
    assert test_mat.data.shape == (2,4,61)
    assert test_mat.velocities.shape == (2,3,60)

    yORz_velocity = np.array([0]*60)
    x_velocity = np.array([700 for _ in range(60)]) \
                * np.array(([1]*6 + [-1]*6)*6)[3:-9] 

    expected = np.array([
        [x_velocity, yORz_velocity, yORz_velocity],
        [x_velocity, yORz_velocity, yORz_velocity]
    ])
    assert (expected == test_mat.velocities).all()
    principalComponents = test_mat.fit_velocity_pca()

    # Make sure the values return in the currect shape
    assert principalComponents.shape == (60,6)

    # Make sure the variance is explaind by 1 pc (file was built this way)
    assert test_mat.v_pca.explained_variance_ratio_[0] == 1

    
    
    # principalComponents = circ_motion_1D_2labls_mat.fit_velocity_pca()
    # print(type(principalComponents))

if __name__ == '__main__':
    pytest.main()