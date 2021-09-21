from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
import typing
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr


data = namedtuple("data", ["X", "y", "idx"])


class DatasetCreator(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    @staticmethod
    @abstractmethod
    def select_events(context_data_file_path: Path) -> pd.DataFrame:
        """
            abstract method to select events for dataset
            """

    @staticmethod
    @abstractmethod
    def select_features(df: pd.DataFrame) -> pd.DataFrame:
        """
            abstract method to select features for dataset
            """

    @staticmethod
    @abstractmethod
    def select_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        abstract method to select labels for dataset
        """

    @staticmethod
    @abstractmethod
    def train_valid_test_split(df_X: pd.DataFrame, df_y: pd.DataFrame,
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
    df_event_selection = creator.select_events(context_data_file_path=data_path)

    df_X = creator.select_features(df=df_event_selection)

    df_y = creator.select_labels(df=df_event_selection)

    train, valid, test = creator.train_valid_test_split(df_X=df_X, df_y=df_y, manual_split=manual_split)
    train, valid, test = creator.scale_data(train, valid, test, manual_scale=manual_scale)
    train, valid, test = creator.one_hot_encode(train, valid, test)

    return train, valid, test
