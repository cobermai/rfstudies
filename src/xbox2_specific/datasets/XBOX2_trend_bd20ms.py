"""Selecting from context data and prepare dataset XBOX2_trend_bd20ms for machine learning. """
from collections import namedtuple
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from src.utils.dataset_creator import DatasetCreator
from src.xbox2_specific.utils import dataset_utils

data = namedtuple("data", ["X", "y", "idx"])


class XBOX2TrendBD20msSelect(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """
    @staticmethod
    def select_events(data_path: Path) -> list[bool]:
        """
        selection of events in data
        :param data_path: path to context data file
        :return df: pandas dataframe with data from selected events
        """
        # select only events with breakdown in 20 ms + some healthy events
        bd_selection_list = ["is_bd_in_20ms"]
        selection = dataset_utils.select_events_from_list(context_data_file_path=data_path / "context.hdf",
                                                          selection_list=bd_selection_list)
        return selection

    @staticmethod
    def select_features(data_path: Path, selection: list[bool]) -> xr.DataArray:
        """
        returns features of selected events for modeling
        :param data_path: path to data
        :param selection: boolean list for indexing which events to select
        :return df_X: features of selected events
        """
        feature_list = ["run_no",
                        "PrevTrendData/Loadside win", "PrevTrendData/Tubeside win",
                        "PrevTrendData/Collector", "PrevTrendData/Gun", "PrevTrendData/IP before PC",
                        "PrevTrendData/PC IP", "PrevTrendData/WG IP", "PrevTrendData/IP Load",
                        "PrevTrendData/IP before structure", "PrevTrendData/US Beam Axis IP",
                        "PrevTrendData/Klystron Flange Temp", "PrevTrendData/Load Temp",
                        "PrevTrendData/PC Left Cavity Temp", "PrevTrendData/PC Right Cavity Temp",
                        "PrevTrendData/Bunker WG Temp", "PrevTrendData/Structure Input Temp",
                        "PrevTrendData/Chiller 1", "PrevTrendData/Chiller 2", "PrevTrendData/Chiller 3",
                        "PrevTrendData/PKI FT avg", "PrevTrendData/PSI FT avg", "PrevTrendData/PSR FT avg",
                        "PrevTrendData/PSI max", "PrevTrendData/PSR max", "PrevTrendData/PEI max",
                        "PrevTrendData/DC Down min", "PrevTrendData/DC Up min",
                        "PrevTrendData/PSI Pulse Width"]

        # Get selected features
        with h5py.File(data_path / "context.hdf") as file:
            data = np.empty(shape=(len(np.arange(sum(selection))), 1, len(feature_list)))
            for feature_ind, feature in enumerate(feature_list):
                data[:, 0, feature_ind] = dataset_utils.read_hdf_dataset(file, feature)[selection]
            run_no = dataset_utils.read_hdf_dataset(file, "run_no")[selection]
            Timestamp = dataset_utils.read_hdf_dataset(file, "Timestamp")[selection]

        # Create xarray DataArray
        dim_names = ["event", "samples", "feature"]
        da_X = xr.DataArray(data=data,
                            dims=dim_names,
                            coords={"event": Timestamp,
                                    "feature": feature_list
                                    })

        da_X = da_X.assign_coords(run_no=("event", run_no))
        return da_X

    @staticmethod
    def select_labels(data_path: Path, selection: list) -> xr.DataArray:
        """
        returns labels of selected events for supervised machine learning
        :param data_path: path to data
        :param selection: boolean list for indexing which events to select
        :return df_y: label of selected events
        """
        label_name = "is_healthy"

        with h5py.File(data_path / "context.hdf") as file:
            is_healthy = dataset_utils.read_hdf_dataset(file, label_name)[selection]
            run_no = dataset_utils.read_hdf_dataset(file, "run_no")[selection]

        is_healthy = is_healthy[..., np.newaxis]
        dim_names = ["event", "feature"]
        da_y = xr.DataArray(data=is_healthy,
                            dims=dim_names,
                            coords={"feature": [label_name]
                                    })
        da_y = da_y.assign_coords(run_no=("event", run_no))
        return da_y

    @staticmethod
    def train_valid_test_split(da_X: xr.DataArray, da_y: xr.DataArray,
                               splits: Optional[tuple] = None,
                               manual_split: Optional[tuple] = None) -> tuple:
        """
        Function splits data into training, testing and validation set using random sampling. Note that this function
        can be overwritten in the concrete dataset selection.
        :param da_X: input data array of shape (event, sample, feature)
        :param da_y: output data array of shape (event)
        :param splits: tuple specifying splitting fractions (training, validation, test)
        :param manual_split: tuple of lists specifying which runs to put in different sets (train, valid, test).
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

        if manual_split is None:
            idx = np.arange(len(da_X["event"]))
            X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = \
                train_test_split(da_X, da_y, idx, train_size=splits[0])
            X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
                train_test_split(X_tmp, y_tmp, idx_tmp, train_size=splits[1] / (1 - (splits[0])))
        else:
            train_runs = manual_split[0]
            valid_runs = manual_split[1]
            test_runs = manual_split[2]

            idx = np.arange(len(da_X["event"]))

            X_train = da_X[da_X["run_no"].isin(train_runs)]
            y_train = da_y[da_y["run_no"].isin(train_runs)]
            idx_train = idx[da_X["run_no"].isin(train_runs)]
            X_valid = da_X[da_X["run_no"].isin(valid_runs)]
            y_valid = da_y[da_y["run_no"].isin(valid_runs)]
            idx_valid = idx[da_X["run_no"].isin(valid_runs)]
            X_test = da_X[da_X["run_no"].isin(test_runs)]
            y_test = da_y[da_y["run_no"].isin(test_runs)]
            idx_test = idx[da_X["run_no"].isin(test_runs)]

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
        if manual_scale is None:  # standard scale all using mean and std of train set
            mean = np.mean(train.X, axis=0)
            std = np.std(train.X, axis=0)

            train_X_scaled = dataset_utils.da_to_numpy_for_ml((train.X - mean) / std)
            valid_X_scaled = dataset_utils.da_to_numpy_for_ml((valid.X - mean) / std)
            test_X_scaled = dataset_utils.da_to_numpy_for_ml((test.X - mean) / std)

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
            for i in manual_scale:
                if any(train_X["run_no"] == i):
                    train_X_i = train_X[train_X["run_no"] == i]
                    mean_train = np.mean(train_X_i, axis=0)
                    std_train = np.std(train_X_i, axis=0)
                    train_X_i_scaled = dataset_utils.da_to_numpy_for_ml((train_X_i - mean_train) / std_train)
                    train_X_scaled[train_X["run_no"] == i] = train_X_i_scaled
                if any(valid_X["run_no"] == i):
                    valid_X_i = valid_X[valid_X["run_no"] == i]
                    mean_valid = np.mean(valid_X_i, axis=0)
                    std_valid = np.std(valid_X_i, axis=0)
                    valid_X_i_scaled = dataset_utils.da_to_numpy_for_ml((valid_X_i - mean_valid) / std_valid)
                    valid_X_scaled[valid_X["run_no"] == i] = valid_X_i_scaled
                if any(test_X["run_no"] == i):
                    test_X_i = test_X[test_X["run_no"] == i]
                    mean_test = np.mean(test_X_i, axis=0)
                    std_test = np.std(test_X_i, axis=0)
                    test_X_i_scaled = dataset_utils.da_to_numpy_for_ml((test_X_i - mean_test) / std_test)
                    test_X_scaled[test_X["run_no"] == i] = test_X_i_scaled

            train_X_scaled = dataset_utils.da_to_numpy_for_ml(train_X_scaled)
            valid_X_scaled = dataset_utils.da_to_numpy_for_ml(valid_X_scaled)
            test_X_scaled = dataset_utils.da_to_numpy_for_ml(test_X_scaled)

            train = train._replace(X=train_X_scaled)
            valid = valid._replace(X=valid_X_scaled)
            test = test._replace(X=test_X_scaled)

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
        train_y = dataset_utils.da_to_numpy_for_ml(train.y)
        valid_y = dataset_utils.da_to_numpy_for_ml(valid.y)
        test_y = dataset_utils.da_to_numpy_for_ml(test.y)

        enc = OneHotEncoder(categories='auto').fit(train_y.reshape(-1, 1))

        train = train._replace(y=enc.transform(train_y.reshape(-1, 1)).toarray())
        valid = valid._replace(y=enc.transform(valid_y.reshape(-1, 1)).toarray())
        test = test._replace(y=enc.transform(test_y.reshape(-1, 1)).toarray())

        return train, valid, test
