"""Selection of data for ECG200 dataset. """
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

        data_train_list = data_train[0]
        data_train_array = np.empty(shape=(len(data_train_list), (len(data_train_list[0]))))
        for ind, signal in enumerate(data_train_list):
            data_train_array[ind, :] = list(signal)
        ind_train = list(range(len(data_train_list)))

        data_test_list = data_test[0]
        data_test_array = np.empty(shape=(len(data_test_list), (len(data_test_list[0]))))
        for ind, signal in enumerate(data_test_list):
            data_test_array[ind, :] = list(signal)
        ind_test = list(range(ind_train[-1] + 1, ind_train[-1] + 1 + len(data_train_list)))

        data = np.concatenate([data_train_array, data_test_array])

        is_train = np.ones(shape=(len(data)), dtype=bool)
        is_train[ind_test] = False

        data_array = xr.DataArray(data=data,
                                  dims=["event", "sample"])

        data_array = data_array.assign_coords(is_train=("event", is_train))

        return data_array

    @staticmethod
    def select_features(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns features of selected events for modeling
        :param data_array: xarray DataArray with data
        :return: xarray DataArray with features of selected events
        """
        X_data_array = data_array[:, 0:96]
        return X_data_array

    @staticmethod
    def select_labels(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns labels of selected events for supervised machine learning
        :param data_array: xarray data array of data from selected events
        :return: labels of selected events
        """
        y_data_array = data_array[:, -1]
        return y_data_array

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
        idx = np.arange(len(X_data_array))
        X_train, X_valid, y_train, y_valid, idx_train, idx_valid = \
            train_test_split(X_data_array[X_data_array["is_train"] == True],
                             y_data_array[y_data_array["is_train"] == True],
                             idx[X_data_array["is_train"] == True],
                             train_size=0.9)

        train = data(X_train, y_train, idx_train)
        valid = data(X_valid, y_valid, idx_valid)
        test = data(X_data_array[X_data_array["is_train"] == False],
                    y_data_array[y_data_array["is_train"] == False],
                    idx[X_data_array["is_train"] == True])

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
        mean = train.X.values.mean()
        std = train.X.values.std()
        X_train = (train.X.values - mean) / std
        X_train = X_train[..., np.newaxis]
        train = train._replace(X=X_train)
        X_valid = (valid.X.values - mean) / std
        X_valid = X_valid[..., np.newaxis]
        valid = valid._replace(X=X_valid)
        X_test = (test.X.values - mean) / std
        X_test = X_test[..., np.newaxis]
        test = test._replace(X=X_test)
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
        enc = OneHotEncoder(categories='auto').fit(train.y.values.reshape(-1, 1))
        train = train._replace(y=enc.transform(train.y.values.reshape(-1, 1)).toarray())
        valid = valid._replace(y=enc.transform(valid.y.values.reshape(-1, 1)).toarray())
        test = test._replace(y=enc.transform(test.y.values.reshape(-1, 1)).toarray())

        return train, valid, test
