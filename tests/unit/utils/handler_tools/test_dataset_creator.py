import numpy as np
import pytest
from src.utils.handler_tools import dataset_creator


@pytest.mark.parametrize("y, \
                         y_one_hot_expected",
                         [(np.array(['good', 'bad', 'good']),
                           np.array([[0, 1], [1, 0], [0, 1]])),
                          (np.array(['bad', 'good', 'bad']),
                           np.array([[1, 0], [0, 1], [1, 0]])),
                          (np.zeros(1),
                           np.array([[1]]))
                          ])
def test__one_hot(y, y_one_hot_expected):
    """
    Test one_hot function of dataset_creator
    """
    # ARRANGE

    # ACT
    y_one_hot = dataset_creator.one_hot_encode(y=y)

    # ASSERT
    assert (y_one_hot == y_one_hot_expected).all()


@pytest.mark.skip(reason="no way of currently testing this")
def test__load_dataset(tmp_path):
    d = tmp_path / "context.hdf"
    d.mkdir()
    dataset_name = "simple_select"
    train, valid, test = dataset_creator.load_dataset(data_path=tmp_path, dataset_name=dataset_name)


def test__load_dataset_error(tmp_path):
    dataset_name = "nonexistent_dataset"
    with pytest.raises(AssertionError):
        dataset_creator.load_dataset(data_path=tmp_path, dataset_name=dataset_name)

