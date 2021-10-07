from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
import typing
from typing import Optional
import xarray as xr


data = namedtuple("data", ["X", "y", "idx"])


class DatasetCreator(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    @staticmethod
    @abstractmethod
    def select_events(data_path: Path) -> list:
        """
            abstract method to select events for dataset
            """

    @staticmethod
    @abstractmethod
    def select_features(data_path: Path, selection: list) -> xr.DataArray:
        """
            abstract method to select features for dataset
            """

    @staticmethod
    @abstractmethod
    def select_labels(data_path: Path, selection: list)  -> xr.DataArray:
        """
        abstract method to select labels for dataset
        """

    @staticmethod
    @abstractmethod
    def train_valid_test_split(da_X: xr.DataArray, da_y: xr.DataArray,
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
                 manual_split: Optional[tuple] = None, manual_scale: Optional[list] = None) -> typing.Tuple:
    """
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param data_path: path to datafile
    :param manual_scale: tuple of lists that describes groups of the data which is scaled separately
    :param manual_split: list that describes a manual split of the data
    :return: train, valid, test: tuple with data of type named tuple
    """
    selection = creator.select_events(data_path=data_path)

    X_DataArray = creator.select_features(data_path=data_path, selection=selection)

    y_DataArray = creator.select_labels(data_path=data_path, selection=selection)

    train, valid, test = creator.train_valid_test_split(X_DataArray=X_DataArray, y_DataArray=y_DataArray, manual_split=manual_split)
    train, valid, test = creator.scale_data(train, valid, test, manual_scale=manual_scale)
    train, valid, test = creator.one_hot_encode(train, valid, test)

    return train, valid, test
