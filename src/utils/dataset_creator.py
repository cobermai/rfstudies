import typing
from collections import namedtuple
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from src.utils.handler_tools.xbox2_datasets import trend_bd_next_pulse

def train_valid_test_split(X, y, splits: tuple) -> typing.Tuple:
    """
    splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test)
    :return: train, valid, test: data of type named tuple
    """
    if splits[0] == 1:
        raise ValueError('Training set fraction cannot be 1')

    idx = np.arange(len(X))
    X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = \
        train_test_split(X, y, idx, train_size=splits[0])
    X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
        train_test_split(X_tmp, y_tmp, idx_tmp, train_size=splits[1] / (1 - (splits[0])))

    data = namedtuple("data", ["X", "y", "idx"])
    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)
    test = data(X_test, y_test, idx_test)

    return train, valid, test


def one_hot_encode(y):
    """
    transform the labels from integers to one hot vectors
    """
    enc = OneHotEncoder(categories='auto')
    return enc.fit_transform(y.reshape(-1, 1)).toarray()


def load_dataset(data_path, dataset_name):
    if dataset_name == "trend_bd_next_pulse":
        X, y = trend_bd_next_pulse.select_data(context_data_file_path=data_path / "context.hdf")
        X_scaled = trend_bd_next_pulse.scale_data(X)
    else:
        raise AssertionError

    y_hot = one_hot_encode(y)
    train, valid, test = train_valid_test_split(X=X_scaled, y=y_hot, splits=(0.7, 0.2, 0.1))

    return train, valid, test
