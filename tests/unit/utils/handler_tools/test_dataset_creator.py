import numpy as np
import pytest
from src.utils.handler_tools import dataset_creator


def test__train_valid_test_split():
    """
    Test train_valid_test_split() function
    """
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))
    splits_expected = 0.7, 0.1, 0.2

    # ACT
    train, valid, test = dataset_creator.train_valid_test_split(X=X, y=y, splits=splits_expected)

    # ASSERT
    assert set(X) == set(train.X).union(valid.X).union(test.X)
    assert set(y) == set(train.y).union(valid.y).union(test.y)
    length = len(X)
    assert set(range(length)) == set(train.idx).union(valid.idx).union(test.idx)
    splits_output = len(train.X) / length, len(valid.X) / length, len(test.X) / length
    assert splits_output == splits_expected


def test__train_valid_test_split_errors():
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))

    # ACT
    with pytest.raises(ValueError):
        dataset_creator.train_valid_test_split(X=X, y=y, splits=(1, 0, 0))


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

