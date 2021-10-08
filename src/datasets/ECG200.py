"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
from collections import namedtuple
from typing import Optional, NamedTuple
import numpy as np
import xarray as xr
from scipy.io import arff
from src.utils.dataset_creator import DatasetCreator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = namedtuple("data", ["X", "y", "idx"])


class ECG200(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def select_events(data_path: Path) -> xr.DataArray:
        """
        selection of events in data
        :param data_path: path to context data file
        :return: data array with data of selected events
        """

        data_train = arff.loadarff(data_path / "ECG200_TRAIN.arff")
        data_test = arff.loadarff(data_path / "ECG200_TEST.arff")

        data_set = xr.Dataset("train", data_train[0],
                              "train", data_test[0])

        data_array = data_set
        # df_test = xr.Dataset(data_test[0]).add_suffix('_test')

        # data_array = pd.concat([df_train, df_test], axis=1)
        return data_array

    @staticmethod
    def select_features(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns features of selected events for modeling
        :param data_array: xarray DataArray with data
        :return: xarray DataArray with features of selected events
        """
        X = data_array.filter(regex="att").values
        X = X[..., np.newaxis]
        return X

    @staticmethod
    def select_labels(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns labels of selected events for supervised machine learning
        :param data_array: xarray data array of data from selected events
        :return: labels of selected events
        """
        return data_array.filter(regex="target").values

    @staticmethod
    def train_valid_test_split(X_data_array: xr.DataArray, y_data_array: xr.DataArray,
                               splits: Optional[tuple] = None,
                               manual_split: Optional[tuple] = None) -> tuple:
        """
        Function splits data into training, testing and validation set using random sampling. Note that this function
        can be overwritten in the concrete dataset selection.
        :param X_data_array: input data array of shape (event, sample, feature)
        :param y_data_array: output data array of shape (event)
        :param splits: tuple specifying splitting fractions (training, validation, test)
        :param manual_split: tuple of lists specifying which runs to put in different sets (train, valid, test).
        :return: Tuple with data of type named tuple
        """
        idx = np.arange(len(X_data_array[:, 0:96]))
        X_train, X_valid, y_train, y_valid, idx_train, idx_valid = \
            train_test_split(X_data_array[:, 0:96], y_data_array[:, 0], idx, train_size=0.9)

        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X_train, y_train, idx_train)
        valid = data(X_valid, y_valid, idx_valid)
        test = data(X_data_array[:, 96:], y_data_array[:, 1], idx)

        return train, valid, test

    @staticmethod
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
        mean = train.X.mean()
        std = train.X.std()

        train = train._replace(X=((train.X - mean) / std))
        valid = valid._replace(X=((valid.X - mean) / std))
        test = test._replace(X=((test.X - mean) / std))

        return train, valid, test

    @staticmethod
    def one_hot_encode(train: data, valid: data, test: data) -> tuple:
        """
        Function transforms the labels from integers to one hot vectors.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training with type named tuple which has attributes X, y and idx
        :param valid: data for validation with type named tuple which has attributes X, y and idx
        :param test: data for testing with type named tuple which has attributes X, y and idx
        :return: train, valid, test: Tuple containing data with type named tuple which has attributes X, y and idx
        """
        enc = OneHotEncoder(categories='auto').fit(train.y.reshape(-1, 1))
        train = train._replace(y=enc.transform(train.y.reshape(-1, 1)).toarray())
        valid = valid._replace(y=enc.transform(valid.y.reshape(-1, 1)).toarray())
        test = test._replace(y=enc.transform(test.y.reshape(-1, 1)).toarray())

        return train, valid, test
