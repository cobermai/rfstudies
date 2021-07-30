from pathlib import Path
from collections import namedtuple
import numpy as np
import pytest
from src.model import classifier


def test__one_hot():
    # ARRANGE
    data = namedtuple("data", ["X", "y", "idx"])
    X_train = np.empty(1)
    y_train = np.array(['good', 'bad', 'good'])
    idx_train = np.empty(1)
    X_valid = np.empty(1)
    y_valid = np.array(['bad', 'good', 'bad'])
    idx_valid = np.empty(1)

    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)

    clf = classifier.Classifier(train_data=train,
                     valid_data=valid,
                     classifier_name="fcn",
                     output_directory=Path("~/PycharmProjects/mlframework/tests/output"))

    y_train_hot_expected = np.array([[0, 1], [1, 0], [0, 1]])
    y_valid_hot_expected = np.array([[1, 0], [0, 1], [1, 0]])

    # ACT
    y_train_hot, y_valid_hot = clf.one_hot()

    # ASSERT
    assert (y_train_hot == y_train_hot_expected).all()
    assert (y_valid_hot == y_valid_hot_expected).all()


    # def test__eval_classifications():
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
    # def test__calc_class_imbalance(y):
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
