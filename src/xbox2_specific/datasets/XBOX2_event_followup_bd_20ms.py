"""Selecting from context data and prepare dataset XBOX2_event_bd20ms for machine learning. """
from collections import namedtuple
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import h5py
from tensorflow import one_hot
from sklearn.model_selection import train_test_split
from src.utils.dataset_creator import DatasetCreator
from src.xbox2_specific.utils import dataset_utils

data = namedtuple("data", ["X", "y", "idx"])


class XBOX2EventFollowupBD20msSelect(DatasetCreator):
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
        # select events with breakdown in 20 ms
        bd_selection_list = ["is_bd_in_20ms"]
        selection = dataset_utils.select_events_from_list(context_data_file_path=data_path / "context.hdf",
                                                          selection_list=bd_selection_list)

        # read features into data_array
        feature_list = ["PEI Amplitude",
                        "PSI Amplitude",
                        "PSR Amplitude",
                        "PKI Amplitude",
                        "DC Up",
                        "DC Down"]
        with h5py.File(data_path / "context.hdf", 'r') as file:
            ext_link_index = dataset_utils.read_hdf_dataset(file, "event_ext_link_index")[selection]
        data_array = dataset_utils.event_ext_link_hdf_to_da_timestamp(file_path=data_path / "EventDataExtLinks.hdf",
                                                                      ext_link_index=ext_link_index,
                                                                      feature_list=feature_list)

        # read label and metadata
        label_name = "is_bd_in_20ms"
        with h5py.File(data_path / "context.hdf", 'r') as file:
            is_bd_in_20ms = dataset_utils.read_hdf_dataset(file, label_name)[selection]
            timestamp = dataset_utils.read_hdf_dataset(file, "Timestamp")[selection]
            run_no = dataset_utils.read_hdf_dataset(file, "run_no")[selection]
            is_followup = dataset_utils.determine_followup(is_bd_in_20ms, timestamp, threshold=60)

        # add label to data_array
        data_array = data_array.assign_coords(is_bd_in_20ms=("event", is_bd_in_20ms))
        # add meta data
        data_array = data_array.assign_coords(run_no=("event", run_no))
        data_array = data_array.assign_coords(timestamp=("event", timestamp))
        data_array = data_array.assign_coords(is_followup=("event", is_followup))

        return data_array

    @staticmethod
    def select_features(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns features of selected events for modeling
        :param data_array: xarray DataArray with data
        :return: xarray DataArray with features of selected events
        """
        X_data_array = data_array[(data_array["is_followup"] == True) | (data_array['is_bd_in_20ms'] == False)]
        X_data_array = X_data_array.drop_vars('is_bd_in_20ms')
        return X_data_array

    @staticmethod
    def select_labels(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns labels of selected events for supervised machine learning
        :param data_array: xarray data array of data from selected events
        :return: labels of selected events
        """
        label_name = "is_bd_in_20ms"
        y_data_array = data_array[label_name]
        y_data_array = y_data_array[(y_data_array["is_followup"] == True) | (y_data_array["is_bd_in_20ms"] == False)]
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

        if manual_split is None:
            idx = np.arange(len(X_data_array["event"]))
            X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = \
                train_test_split(X_data_array, y_data_array, idx, train_size=splits[0])
            X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
                train_test_split(X_tmp, y_tmp, idx_tmp, train_size=splits[1] / (1 - (splits[0])))
        else:
            train_runs = manual_split[0]
            valid_runs = manual_split[1]
            test_runs = manual_split[2]

            idx = np.arange(len(X_data_array["event"]))

            def get_data_in_runs(X: xr.DataArray, y: xr.DataArray, idx: np.ndarray, runs):
                X_in_runs = X[X["run_no"].isin(runs)]
                y_in_runs = y[y["run_no"].isin(runs)]
                idx_in_runs = idx[X["run_no"].isin(runs)]
                return X_in_runs, y_in_runs, idx_in_runs

            X_train, y_train, idx_train = get_data_in_runs(X_data_array, y_data_array, idx, train_runs)
            X_valid, y_valid, idx_valid = get_data_in_runs(X_data_array, y_data_array, idx, valid_runs)
            X_test, y_test, idx_test = get_data_in_runs(X_data_array, y_data_array, idx, test_runs)

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
        :param manual_scale: list that specifies groups which are scaled separately
        :return: train, valid, test: Tuple with data of type named tuple
        """

        def standard_scale(X, axis=1):
            """
            Function for standard scaling array data
            :param X: numpy-like array with data to be scaled
            :return: array with standard scaled values
            """
            mean = np.mean(X, axis=axis)
            std = np.std(X, axis=axis)
            return (X - mean) / std

        if manual_scale is None:
            # standard scale training, valid and test separately. Scaling is done for each signal.
            train_X_scaled = standard_scale(train.X, axis=1)
            valid_X_scaled = standard_scale(valid.X, axis=1)
            test_X_scaled = standard_scale(test.X, axis=1)

            train = train._replace(X=train_X_scaled)
            valid = valid._replace(X=valid_X_scaled)
            test = test._replace(X=test_X_scaled)
        else:  # standard scale by run using mean and std of train set
            train_X = train.X
            valid_X = valid.X
            test_X = test.X

            # placeholders for scaled values
            train_X_scaled = train_X
            valid_X_scaled = valid_X
            test_X_scaled = test_X

            # standard scale each run included in manual scale separately
            for run in manual_scale:
                if any(train_X["run_no"] == run):
                    train_X_run = train_X[train_X["run_no"] == run]
                    train_X_run_scaled = standard_scale(train_X_run, axis=0)
                    train_X_scaled[train_X["run_no"] == run] = train_X_run_scaled
                if any(valid_X["run_no"] == run):
                    valid_X_run = valid_X[valid_X["run_no"] == run]
                    valid_X_run_scaled = standard_scale(valid_X_run, axis=0)
                    valid_X_scaled[valid_X["run_no"] == run] = valid_X_run_scaled
                if any(test_X["run_no"] == run):
                    test_X_run = test_X[test_X["run_no"] == run]
                    test_X_run_scaled = standard_scale(test_X_run, axis=0)
                    test_X_scaled[test_X["run_no"] == run] = test_X_run_scaled
            train = train._replace(X=train_X_scaled)
            valid = valid._replace(X=valid_X_scaled)
            test = test._replace(X=test_X_scaled)

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
        def expand_and_onehot(y: xr.DataArray):
            """
            :param y: array of class labels
            :return: namedtuples train, valid and test with onehot encoded y
            """
            n_labels = len(np.unique(y))
            y_one_hot = y.expand_dims({"dummy": n_labels}, axis=1)
            y_one_hot.values = one_hot(y, n_labels)
            return y_one_hot

        train_y_one_hot = expand_and_onehot(train.y)
        valid_y_one_hot = expand_and_onehot(valid.y)
        test_y_one_hot = expand_and_onehot(test.y)

        train = train._replace(y=train_y_one_hot)
        valid = valid._replace(y=valid_y_one_hot)
        test = test._replace(y=test_y_one_hot)

        return train, valid, test