from abc import ABC, abstractmethod
import typing
import h5py
from pathlib import Path
from collections import namedtuple
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.hdf_tools import hdf_to_df_selection
from src.xbox2_specific.datasets import XBOX2_event_bd20ms
from src.xbox2_specific.datasets import XBOX2_trend_bd20ms

class DatasetCreator(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    def read_hdf_dataset(self, file: h5py.File, key: str):
        """
        Read dataset from hdf file
        :param file: h5py File object with read access
        :param key: string specifying key for dataset to read
        :return: array containing contents of h5py dataset
        """
        dataset = file[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("Specified key does not yield a hdf dataset")
        return dataset[:]

    @abstractmethod
    def select_events(self, context_data_file_path: Path) -> typing.List[bool]:
        """
        abstract method to select events for dataset
        """
        pass

    @abstractmethod
    def select_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        abstract method to select features for dataset
        """
        pass

    @abstractmethod
    def select_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        abstract method to select labels for dataset
        """
        pass

    def scale_data(self, X: np.ndarray) -> np.ndarray:
        """
        Function scales data for prediction with standard scaler. Note that this function can be overwritten in the
        concrete dataset selection.
        :param X: data array of shape (event, sample, feature)
        :return: X_scaled: scaled data array of shape (event, sample, feature)
        """
        X_scaled = np.zeros_like(X)
        for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
            X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
        return X_scaled

    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        Function transforms the labels from integers to one hot vectors. Note that this function can be overwritten in
        the concrete dataset selection.
        :param y: array with labels to encode
        :return: array of one hot encoded labels
        """
        enc = OneHotEncoder(categories='auto')
        return enc.fit_transform(y.reshape(-1, 1)).toarray()

    def train_valid_test_split(self, X: np.ndarray, y: np.ndarray, splits: Optional[tuple] = None) -> typing.Tuple:
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


def load_dataset(creator: DatasetCreator, hdf_dir: Path) -> typing.Tuple:
    """
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param hdf_dir: input directory with hdf files
    :return: train, valid, test: tuple with data of type named tuple
    """
    event_selection = creator.select_events(context_data_file_path=hdf_dir / "context.hdf")
    df_selected = hdf_to_df_selection(hdf_dir / "context.hdf", selection=event_selection)

    X = creator.select_features(df=df_selected)
    y = creator.select_labels(df=df_selected)

    X_scaled = creator.scale_data(X)
    y_hot = creator.one_hot_encode(y)

    train, valid, test = creator.train_valid_test_split(X=X_scaled, y=y_hot)

    return train, valid, test
