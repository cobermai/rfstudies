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
    @abstractmethod
    def select_events(context_data_file_path: Path) -> pd.DataFrame:
        """
        abstract method to select events for dataset
        """

    @staticmethod
    @abstractmethod
    def select_features(df: pd.DataFrame) -> np.ndarray:
        """
        abstract method to select features for dataset
        """

    @staticmethod
    @abstractmethod
    def select_labels(df: pd.DataFrame) -> np.ndarray:
        """
        abstract method to select labels for dataset
        """

    @staticmethod
    def scale_data(X: np.ndarray) -> np.ndarray:
        """
        Function scales data for with sklearn standard scaler.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param X: data array of shape (event, sample, feature)
        :return: X_scaled: scaled data array of shape (event, sample, feature)
        """
        X_scaled = np.zeros_like(X)
        for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
            X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
        return X_scaled

    @staticmethod
    def one_hot_encode(y: np.ndarray) -> np.ndarray:
        """
        Function transforms the labels from integers to one hot vectors.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param y: array with labels to encode
        :return: array of one hot encoded labels
        """
        enc = OneHotEncoder(categories='auto')
        return enc.fit_transform(y.reshape(-1, 1)).toarray()

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
        if splits is None:
            splits = (0.7, 0.2, 0.1)
        if (splits[0] >= 1) or (splits[0] < 0):
            raise ValueError('Training fraction cannot be >= 1 or negative')
        if (splits[1] >= 1) or (splits[1] < 0):
            raise ValueError('Validation fraction cannot be >= 1 or negative')
        if (splits[2] >= 1) or (splits[2] < 0):
            raise ValueError('Test fraction cannot be >= 1 or negative')
        if not np.allclose(splits[0] + splits[1] + splits[2], 1):
            raise ValueError('Splits must sum to 1')

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


def load_dataset(creator: DatasetCreator, data_path: Path,
                 manual_scale: list = None, manual_split: list = None) -> typing.Tuple:
    """
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param data_path: path to datafile
    :param manual_split: list that describes a manual split of the data
    :param manual_scale: list that describes groups of the data which is scaled separately
    :return: train, valid, test: tuple with data of type named tuple
    """
    df_event_selection = creator.select_events(data_path=data_path)

    X = creator.select_features(df=df_event_selection)

    y = creator.select_labels(df=df_event_selection)

    train, valid, test = creator.train_valid_test_split(X=X, y=y, manual_split=manual_split)

    train, valid, test = creator.scale_data(train, valid, test)
    train, valid, test = creator.one_hot_encode(train, valid, test)

    return train, valid, test
