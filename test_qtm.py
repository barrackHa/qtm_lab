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

def test_4by4_shape(mat_4_by_4_file):
    print(mat_4_by_4_file.data.shape)
    assert mat_4_by_4_file.data.shape == (2,4,10)

if __name__ == '__main__':
    pytest.main()