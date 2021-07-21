"""
tests the main module
"""
import numpy as np
import pytest
from src import main


def test__scale_data():
    # ARRANGE

    X = np.array([[[0, 0, 0], [1, 1, 1]]])
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = main.scale_data(X)

    # ASSERT
    assert (X_output == X_expected).all()


def test__train_valid_test_split():
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))
    splits_expected = 0.7, 0.1, 0.2

    # ACT
    train, valid, test = main.train_valid_test_split(X, y, splits_expected)

    # ASSERT
    assert set(X) == set(train.X).union(valid.X).union(test.X)
    assert set(y) == set(train.y).union(valid.y).union(test.y)
    id = np.arange(10)
    assert set(id) == set(train.id).union(valid.id).union(test.id)
    length = len(X)
    splits_output = len(train.X) / length, len(valid.X) / length, len(test.X) / length
    assert splits_output == splits_expected


def test__train_valid_test_split_errors():
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))

    # ACT
    with pytest.raises(ValueError):
        main.train_valid_test_split(X, y, (0, 0, 0))
