"""
tests the main module
"""
import numpy as np
from src import main


def test__scale_data():
    # ARRANGE

    X = np.array([[[0, 0, 0], [1, 1, 1]]])
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = main.scale_data(X)

    # ASSERT
    assert (X_output == X_expected).all()


def test__train_validation_test_split():
    # ARRANGE

    # something that looks like input data
    X = np.array([[[0, 0, 0], [1, 1, 1]]])

    # something that looks like labels
    y = np.array([[[True], [False], [True], [False],
                   [True], [False], [True], [False]
                   [True], [False]]])

    # expected split data outputs
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])
    y_expected = np.array([[[True], [False], [True], [False],
                            [True], [False], [True]]])

    # Don't know if we need to test this
    idx_expected = 1, 3, 4, 7, 5, 4, 9

    splits_expected = 0.7, 0.2, 0.1

    # ACT
    # Maybe force random seed to check correct random selection to test correct split?
    X_output, y_output, idx_output = main.train_validation_test_split(X, y, splits_expected)

    splits_output = len(X_train) / len(X), len(X_valid) / len(X), len(X_test) / len(X)

    # ASSERT

    assert X_output == X_expected and y_output == y_expected and \
           idx_output == idx_expected and splits_output == splits_expected
