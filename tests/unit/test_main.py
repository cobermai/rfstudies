"""
tests the main module
"""
import argparse
import numpy as np
import pytest
from src import main

#
# def test__parse_input_arguments():
#     args = main.parse_input_arguments()
#
#
# @pytest.mark.parametrize("y, \
#                          y_one_hot_expected",
#                          [(np.array(['good', 'bad', 'good']),
#                            np.array([[0, 1], [1, 0], [0, 1]])),
#                           (np.array(['bad', 'good', 'bad']),
#                            np.array([[1, 0], [0, 1], [1, 0]])),
#                           (np.zeros(1),
#                            np.array([[1]]))
#                           ])
# def test__one_hot(y, y_one_hot_expected):
#     """
#     Test one_hot function of main
#     """
#     # ARRANGE
#
#     # ACT
#     y_one_hot = main.one_hot(y)
#
#     # ASSERT
#     assert (y_one_hot == y_one_hot_expected).all()
#
#
# def test__scale_data():
#     """
#     Test scale_data() function
#     """
#     # ARRANGE
#
#     X = np.array([[[0, 0, 0], [1, 1, 1]]])
#     X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])
#
#     # ACT
#     X_output = main.scale_data(X)
#
#     # ASSERT
#     assert (X_output == X_expected).all()
#
#
# def test__train_valid_test_split():
#     """
#     Test train_valid_test_split() function
#     """
#     # ARRANGE
#     X = np.array(range(10, 20))
#     y = np.array(range(20, 30))
#     splits_expected = 0.7, 0.1, 0.2
#
#     # ACT
#     train, valid, test = main.train_valid_test_split(X, y, splits_expected)
#
#     # ASSERT
#     assert set(X) == set(train.X).union(valid.X).union(test.X)
#     assert set(y) == set(train.y).union(valid.y).union(test.y)
#     length = len(X)
#     assert set(range(length)) == set(train.idx).union(valid.idx).union(test.idx)
#     splits_output = len(train.X) / length, len(valid.X) / length, len(test.X) / length
#     assert splits_output == splits_expected
#
#
# def test__train_valid_test_split_errors():
#     # ARRANGE
#     X = np.array(range(10, 20))
#     y = np.array(range(20, 30))
#
#     # ACT
#     with pytest.raises(ValueError):
#         main.train_valid_test_split(X, y, (0, 0, 0))
