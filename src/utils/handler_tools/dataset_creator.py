import typing
from pathlib import Path
from collections import namedtuple
from typing import Optional
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.xbox2_specific.datasets import simple_select
from src.xbox2_specific.datasets import XBOX2_event_bd20ms
from src.xbox2_specific.datasets import XBOX2_trend_bd20ms


def train_valid_test_split(X, y, splits: Optional[tuple] = None) -> typing.Tuple:
    """
    Splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test)
    :return: train, valid, test: Tuple with data of type named tuple
    """
    if splits is None:
        splits = (0.7, 0.2, 0.1)

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
        train, valid, test = train_valid_test_split(X=X, y=y)
    elif dataset_name == "XBOX_event_bd20ms":
        X, y = XBOX2_event_bd20ms.select_data(context_data_file_path=data_path / "context.hdf")
        train, valid, test = train_valid_test_split(X=X, y=y)
    elif dataset_name == "XBOX_trend_bd20ms":
        X, y = XBOX2_trend_bd20ms.select_data(context_data_file_path=data_path / "context.hdf")
        train, valid, test = train_valid_test_split(X=X, y=y)
    else:
        raise AssertionError

    return train, valid, test
