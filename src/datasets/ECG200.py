"""Selection of data for ECG200 dataset. """
from collections import namedtuple
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy.io import arff
from sklearn.model_selection import train_test_split
from tensorflow import one_hot

from src.dataset_creator import DatasetCreator

data = namedtuple("data", ["X", "y", "idx"])


class ECG200(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    For more information about the dataset visit http://www.timeseriesclassification.com/description.php?Dataset=ECG200
    """

    @staticmethod
    def select_events(data_path: Path) -> xr.DataArray:
        """
        selection of events in data
        :param data_path: path to "ECG200_TRAIN.arff" and "ECG200_TEST.arff" file
        :return: data array with data of selected events
        """

        def read_arff(file_path: Path, encoding: str):
            with open(file_path, 'rt', encoding=encoding) as f:
                data_read = f.read()

            stream = StringIO(data_read)
            return arff.loadarff(stream)

        data_train = read_arff(data_path / "ECG200_TRAIN.arff", encoding="utf-8")
        data_test = read_arff(data_path / "ECG200_TEST.arff", encoding="utf-8")

        data_train_list = data_train[0]
        data_train_array = np.empty(shape=(len(data_train_list), (len(data_train_list[0]))))
        for index, signal in enumerate(data_train_list):
            data_train_array[index, :] = list(signal)
        index_train = np.arange(len(data_train_list))

        data_test_list = data_test[0]
        data_test_array = np.empty(shape=(len(data_test_list), (len(data_test_list[0]))))
        for index, signal in enumerate(data_test_list):
            data_test_array[index, :] = list(signal)
        index_test = np.arange(len(index_train), len(index_train) + len(data_test_list))

        data_combined = np.concatenate([data_train_array, data_test_array])

        is_train = np.ones(shape=(len(data_combined)), dtype=bool)
        is_train[index_test] = False

        data_array = xr.DataArray(data=data_combined,
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
        # Select all data but last column which is labels
        X_data_array = data_array[:, :-1]
        return X_data_array

    @staticmethod
    def select_labels(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns labels of selected events for supervised machine learning
        :param data_array: xarray data array of data from selected events
        :return: labels of selected events
        """
        # Select last column which is the class labels
        y_data_array = data_array[:, -1]
        # Set reassign label -1 to 0
        y_data_array[y_data_array == -1] = 0
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
        :param splits: inherited from superclass but unused
        :param manual_split: inherited from superclass but unused
        :return: Tuple with data of type named tuple
        """
        idx = np.arange(len(X_data_array))
        X_train, X_valid, y_train, y_valid, idx_train, idx_valid = \
            train_test_split(X_data_array[X_data_array["is_train"] == True],
                             y_data_array[y_data_array["is_train"] == True],
                             idx[X_data_array["is_train"] == True],
                             train_size=0.9)
        X_train = X_train.expand_dims({"feature": 1}, axis=2)
        X_valid = X_valid.expand_dims({"feature": 1}, axis=2)

        X_test = X_data_array[X_data_array["is_train"] == False]
        X_test = X_test.expand_dims({"feature": 1}, axis=2)
        y_test = y_data_array[y_data_array["is_train"] == False]
        idx_test = idx[X_data_array["is_train"] == False]

        train = data(X_train, y_train, idx_train)
        valid = data(X_valid, y_valid, idx_valid)
        test = data(X_test, y_test, idx_test)

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
        :param manual_scale: inherited from superclass but unused
        :return: train, valid, test: Tuple with data of type named tuple
        """
        mean = train.X.mean()
        std = train.X.std()
        X_train = (train.X - mean) / std
        train = train._replace(X=X_train)
        X_valid = (valid.X - mean) / std
        valid = valid._replace(X=X_valid)
        X_test = (test.X - mean) / std
        test = test._replace(X=X_test)
        return train, valid, test

    @staticmethod
    def one_hot_encode(train: data, valid: data, test: data) -> tuple:
        """
        Function transforms the labels to one hot vectors.
        :param train: data for training with type named tuple which has attributes X, y and idx
        :param valid: data for validation with type named tuple which has attributes X, y and idx
        :param test: data for testing with type named tuple which has attributes X, y and idx
        :return: train, valid, test: Tuple containing data with type named tuple which has attributes X, y and idx
        """
        n_labels = len(np.unique(train.y))
        train_y_one_hot = train.y.expand_dims({"dummy": n_labels}, axis=1)
        train_y_one_hot.values = one_hot(train.y, n_labels)
        valid_y_one_hot = valid.y.expand_dims({"dummy": n_labels}, axis=1)
        valid_y_one_hot.values = one_hot(valid.y, n_labels)
        test_y_one_hot = test.y.expand_dims({"dummy": n_labels}, axis=1)
        test_y_one_hot.values = one_hot(test.y, n_labels)

        train = train._replace(y=train_y_one_hot)
        valid = valid._replace(y=valid_y_one_hot)
        test = test._replace(y=test_y_one_hot)

        return train, valid, test
