import typing
from pathlib import Path
from collections import namedtuple
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.xbox2_specific.datasets import simple_select
from src.xbox2_specific.datasets import XBOX_event_bd20ms


def one_hot_encode(y):
    """
    Transforms the labels from integers to one hot vectors
    :param y: array with labels to encode
    :return: array of one hot encoded labels
    """
    enc = OneHotEncoder(categories='auto')
    return enc.fit_transform(y.reshape(-1, 1)).toarray()


def scale_data(X):
    """
    function scales data for prediction with standard scaler
    :param X: data array of shape (event, sample, feature)
    :return: X_scaled: scaled data array of shape (event, sample, feature)
    """
    X_scaled = np.zeros_like(X)
    for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
        X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
    return X_scaled


def train_valid_test_split(X, y, splits: tuple) -> typing.Tuple:
    """
    Splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test)
    :return: train, valid, test: Tuple with data of type named tuple
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


def load_dataset(data_path: Path, dataset_name: str) -> typing.Tuple:
    """
    Loads the specified data set, does one hot encoding on labels and splits data into train, valid and test set
    :param data_path: Path to input data
    :param dataset_name: Name of the data set
    :return: Tuple of named tuples containing training, validation and test set
    """
    if dataset_name == "simple_select":
        X, y = simple_select.select_data(context_data_file_path=data_path / "context.hdf")
        X_scaled = scale_data(X)
        y_hot = one_hot_encode(y)
        train, valid, test = train_valid_test_split(X=X_scaled, y=y_hot, splits=(0.7, 0.2, 0.1))
    elif dataset_name == "XBOX_event_bd20ms":
        X, y = XBOX_event_bd20ms.select_data(context_data_file_path=data_path / "context.hdf")
        X_scaled = scale_data(X)
        y_hot = one_hot_encode(y)
        train, valid, test = train_valid_test_split(X=X_scaled, y=y_hot, splits=(0.7, 0.2, 0.1))
    else:
        raise AssertionError

    return train, valid, test
