from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
import typing
from typing import Optional
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.hdf_tools import hdf_to_df_selection


class DatasetCreator(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    @staticmethod
    def select_events(context_data_file_path: Path) -> pd.DataFrame:
        """
        abstract method to select events for dataset
        """

    @staticmethod
    def select_features(df: pd.DataFrame) -> np.ndarray:
        """
        abstract method to select features for dataset
        """

    @staticmethod
    def select_labels(df: pd.DataFrame) -> np.ndarray:
        """
        abstract method to select labels for dataset
        """

    @staticmethod
    def train_valid_test_split(X: np.ndarray, y: np.ndarray,
                               splits: Optional[tuple] = None) -> tuple:
        """
        Function splits data into training, testing and validation set using random sampling. Note that this function
        can be overwritten in the concrete dataset selection.
        :param X: input data array of shape (event, sample, feature)
        :param y: output data array of shape (event)
        :param splits: tuple specifying splitting fractions (training, validation, test)
        :return: train, valid, test: Tuple with data of type named tuple
        """

    @staticmethod
    def scale_data(train: typing.NamedTuple, valid: typing.NamedTuple, test: typing.NamedTuple) -> tuple:
        """
        Function scales data for with sklearn standard scaler.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training of type named tuple
        :param valid: data for validation of type named tuple
        :param test: data for testing of type named tuple
        :return: train, valid, test: Tuple with data of type named tuple
        """

    @staticmethod
    def one_hot_encode(train: typing.NamedTuple, valid: typing.NamedTuple, test: typing.NamedTuple) -> tuple:
        """
        Function transforms the labels from integers to one hot vectors.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training of type named tuple
        :param valid: data for validation of type named tuple
        :param test: data for testing of type named tuple
        :return: train, valid, test: Tuple with data of type named tuple
        """



def load_dataset(creator: DatasetCreator, data_path: Path) -> typing.Tuple:
    """
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param data_path: path to datafile
    :return: train, valid, test: tuple with data of type named tuple
    """
    df_event_selection = creator.select_events(data_path)

    X = creator.select_features(df=df_event_selection)

    y = creator.select_labels(df=df_event_selection)

    train, valid, test = creator.train_valid_test_split(X=X, y=y)

    train, valid, test = creator.scale_data(train, valid, test)
    train, valid, test = creator.one_hot_encode(train, valid, test)

    return train, valid, test
