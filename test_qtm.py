import pytest
from fft_mat import QTM

def test_test():
    assert 1 == 1

def test_fails():
    with pytest.raises(AssertionError):
        assert 1 == 2

@pytest.fixture
def mat_4_by_4_file():
    return QTM('f')

@pytest.fixture
def circ_motion_1D_2labls_mat():
    return QTM('circ_motion_1D_2labls')

def test_4by4_shape(mat_4_by_4_file):
    # print(mat_4_by_4_file.data.shape)
    assert mat_4_by_4_file.data.shape == (2,4,10)

def test_velocity_pca(circ_motion_1D_2labls_mat):
    assert circ_motion_1D_2labls_mat.data.shape == (2,4,61)
    assert circ_motion_1D_2labls_mat.velocities.shape == (2,3,60)
    
    # principalComponents = circ_motion_1D_2labls_mat.fit_velocity_pca()
    # print(type(principalComponents))

if __name__ == '__main__':
    pytest.main()