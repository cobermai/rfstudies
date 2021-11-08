from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
import typing
from typing import Optional
import numpy as np
import xarray as xr


data = namedtuple("data", ["X", "y", "idx"])


class DatasetCreator(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    @staticmethod
    @abstractmethod
    def select_events(data_path: Path) -> xr.DataArray:
        """
        abstract method to select events for dataset
        """

    @staticmethod
    @abstractmethod
    def select_features(data_array: xr.DataArray) -> xr.DataArray:
        """
        abstract method to select features for dataset
        """

    @staticmethod
    @abstractmethod
    def select_labels(data_array: xr.DataArray) -> xr.DataArray:
        """
        abstract method to select labels for dataset
        """

    @staticmethod
    @abstractmethod
    def train_valid_test_split(X_data_array: xr.DataArray, y_data_array: xr.DataArray,
                               splits: Optional[tuple] = None,
                               manual_split: list = None) -> tuple:
        """
        abstract method to split data set into training, validation and test set
        """

    @staticmethod
    @abstractmethod
    def scale_data(train: data, valid: data, test: data,
                   manual_scale: Optional[list] = None) -> tuple:
        """
        Function scales data for with sklearn standard scaler.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training of type named tuple
        :param valid: data for validation of type named tuple
        :param test: data for testing of type named tuple
        :param manual_scale: list that specifies groups which are scaled separately
        :return: train, valid, test: Tuple with data of type named tuple
        """

    @staticmethod
    @abstractmethod
    def one_hot_encode(train: data, valid: data, test: data,) -> tuple:
        """
        Function transforms the labels from integers to one hot vectors.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training of type named tuple
        :param valid: data for validation of type named tuple
        :param test: data for testing of type named tuple
        :return: train, valid, test: Tuple with data of type named tuple
        """


def load_dataset(creator: DatasetCreator, data_path: Path,
                 splits: Optional[tuple] = None,
                 manual_split: Optional[tuple] = None,
                 manual_scale: Optional[list] = None) -> typing.Tuple:
    """
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param data_path: path to datafile
    :param splits: train, valid, test split fractions
    :param manual_scale: tuple of lists that describes groups of the data which is scaled separately
    :param manual_split: list that describes a manual split of the data
    :return: train, valid, test: tuple with data of type named tuple
    """
    data_array = creator.select_events(data_path=data_path)

    X_data_array = creator.select_features(data_array=data_array)

    y_data_array = creator.select_labels(data_array=data_array)

    train, valid, test = creator.train_valid_test_split(X_data_array=X_data_array,
                                                        y_data_array=y_data_array,
                                                        splits=splits,
                                                        manual_split=manual_split)
    train, valid, test = creator.scale_data(train, valid, test, manual_scale=manual_scale)
    train, valid, test = creator.one_hot_encode(train, valid, test)

    return train, valid, test


def da_to_numpy_for_ml(train, valid, test) -> tuple:
    """
    Function that takes raw values of xarray, replaces NaN with zero and infinity with large finite numbers
    :param data_array: xarray DataArray
    :return: numpy array ready for machine learning algorithms
    """
    train_X = np.nan_to_num(train.X.values)
    train_y = np.nan_to_num(train.y.values)
    valid_X = np.nan_to_num(valid.X.values)
    valid_y = np.nan_to_num(valid.y.values)
    test_X = np.nan_to_num(test.X.values)
    test_y = np.nan_to_num(test.y.values)
    train_np = data(train_X, train_y, train.idx)
    valid_np = data(valid_X, valid_y, valid.idx)
    test_np = data(test_X, test_y, test.idx)
    return train_np, valid_np, test_np
