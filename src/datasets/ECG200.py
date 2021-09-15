"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.io import arff
from src.utils.dataset_creator import DatasetCreator
import typing
from src.utils.hdf_tools import hdf_to_df_selection
from src.xbox2_specific.utils import dataset_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ECG200(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def select_events(file_path: Path) -> pd.DataFrame:
        """
        selection of events in data
        :param file_path: path to data file
        :return selection: boolean filter for selecting breakdown events
        """

        data_train = arff.loadarff(file_path / "ECG200_TRAIN.arff")
        df_train = pd.DataFrame(data_train[0])

        data_test = arff.loadarff(file_path / "ECG200_TEST.arff")
        df_test = pd.DataFrame(data_test[0]).add_suffix('_test')

        df = pd.concat([df_train, df_test], axis=1)
        return df

    @staticmethod
    def select_features(df: pd.DataFrame) -> np.ndarray:
        """
        returns features of selected events for modeling
        :param df: dataframe with selected events
        :return X: label of selected events
        """
        X = df.filter(regex="att").values
        X = X[..., np.newaxis]
        return X

    @staticmethod
    def select_labels(df: pd.DataFrame) -> np.ndarray:
        """
        returns labels of selected events for supervised machine learning
        :param df: dataframe with selected events
        :return y: label of selected events
        """
        return df.filter(regex="target").values

    @staticmethod
    def train_valid_test_split(X: np.ndarray, y: np.ndarray,
                               splits: typing.Optional[tuple] = None) -> tuple:

        idx = np.arange(len(X[:, 0:96]))
        X_train, X_valid, y_train, y_valid, idx_train, idx_valid = \
            train_test_split(X[:, 0:96], y[:, 0], idx, train_size=0.9)

        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X_train, y_train, idx_train)
        valid = data(X_valid, y_valid, idx_valid)
        test = data(X[:, 96:], y[:, 1], idx)

        return train, valid, test


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

        X_scaled = np.zeros_like(train.X)
        for feature_index in range(len(train.X[0, 0, :])):  # Iterate through feature
            scaler = StandardScaler().fit(train.X[:, :, feature_index])

            train.X[:, :, feature_index] = scaler.transform(train.X[:, :, feature_index])
            valid.X[:, :, feature_index] = scaler.transform(valid.X[:, :, feature_index])
            test.X[:, :, feature_index] = scaler.transform(test.X[:, :, feature_index])

        return train, valid, test


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
        enc = OneHotEncoder(categories='auto').fit(train.y.reshape(-1, 1))
        train = train._replace(y=enc.transform(train.y.reshape(-1, 1)).toarray())
        valid = valid._replace(y=enc.transform(valid.y.reshape(-1, 1)).toarray())
        test = test._replace(y=enc.transform(test.y.reshape(-1, 1)).toarray())

        return train, valid, test