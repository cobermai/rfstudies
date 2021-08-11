import numpy as np
import pytest
from src.xbox2_specific.datasets import simple_select


@pytest.mark.skip(reason="no way of currently testing this")
def test__select_data(tmp_path):
    # ARRANGE
    X_expected = np.array([1, 2, 3])
    y_expected = np.array([4, 5, 6])

    # Create dummy file in tmp path

    # ACT
    X, y = simple_select.select_data(tmp_path)
    # ASSERT
    assert X == X_expected
    assert y == y_expected


def test__scale_data():
    """
    Test scale_data() function
    """
    # ARRANGE

    X = np.array([[[0, 0, 0], [1, 1, 1]]])
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = simple_select.scale_data(X)
    print(X_output)
    # ASSERT
    assert (X_output == X_expected).all()


def test__train_valid_test_split():
    """
    Test train_valid_test_split() function
    """
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))
    splits_expected = 0.7, 0.1, 0.2

    # ACT
    train, valid, test = simple_select.train_valid_test_split(X, y, splits_expected)

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
        simple_select.train_valid_test_split(X, y, (1, 0, 0))
