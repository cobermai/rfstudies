"""example code how to select from context data and prepare data for machine learning. """
from collections import namedtuple
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from src.utils.dataset_creator import DatasetCreator
from src.utils.hdf_tools import hdf_to_df_selection
from src.xbox2_specific.utils import dataset_utils


data = namedtuple("data", ["X", "y", "idx"])


class SimpleSelect(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """
    @staticmethod
    def select_events(context_data_file_path: Path) -> pd.DataFrame:
        """
        selection of events in data
        :param context_data_file_path: path to context data file
        :return df: dataframe containing the selected events
        """
        selection_list = ["is_bd_in_40ms", "is_bd_in_20ms", "is_bd"]
        selection = dataset_utils.select_events_from_list(context_data_file_path, selection_list)
        df = hdf_to_df_selection(context_data_file_path, selection=selection)
        df = df[pd.Index(df["run_no"] > 0)]  # Only choose stable runs
        return df

    @staticmethod
    def select_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        returns features of selected events for modeling
        :param df: dataframe with selected events
        :return X_df: label of selected events
        """
        selection_list = ["Timestamp",
                          "PrevTrendData__Timestamp",
                          "is_bd", "is_healthy",
                          "is_bd_in_20ms",
                          "is_bd_in_40ms"]
        feature_names = df.columns.difference(pd.Index(selection_list))

        df_X = df[feature_names]
        return df_X

    @staticmethod
    def select_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        returns labels of selected events for supervised machine learning
        :param df: dataframe with selected events
        :return y_df: label of selected events
        """
        df_y = dataset_utils.get_labels(df=df, label="is_healthy")
        df_y = pd.concat([df_y, dataset_utils.get_labels(df=df, label="run_no")], axis=1)
        return df_y

    @staticmethod
    def train_valid_test_split(df_X: pd.DataFrame, df_y: pd.DataFrame,
                               splits: Optional[tuple] = None,
                               manual_split: Optional[tuple] = None) -> tuple:
        """
        Function splits data into training, testing and validation set using random sampling. Note that this function
        can be overwritten in the concrete dataset selection.
        :param df_X: input data array of shape (event, sample, feature)
        :param df_y: output data array of shape (event)
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
            idx = df_X.index
            X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = \
                train_test_split(df_X, df_y, idx, train_size=splits[0])
            X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
                train_test_split(X_tmp, y_tmp, idx_tmp, train_size=splits[1] / (1 - (splits[0])))
        else:
            train_runs = manual_split[0]
            valid_runs = manual_split[1]
            test_runs = manual_split[2]

            X_train = df_X[pd.Index(df_X["run_no"].isin(train_runs))]
            y_train = df_y[pd.Index(df_y["run_no"].isin(train_runs))]
            idx_train = df_X.index[pd.Index(df_X["run_no"].isin(train_runs))]
            X_valid = df_X[pd.Index(df_X["run_no"].isin(valid_runs))]
            y_valid = df_y[pd.Index(df_y["run_no"].isin(valid_runs))]
            idx_valid = df_X.index[pd.Index(df_X["run_no"].isin(valid_runs))]
            X_test = df_X[pd.Index(df_X["run_no"].isin(test_runs))]
            y_test = df_y[pd.Index(df_y["run_no"].isin(test_runs))]
            idx_test = df_X.index[pd.Index(df_X["run_no"].isin(test_runs))]

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
            mean = train.X.mean()
            std = train.X.std()

            train_X_scaled = df_to_numpy_for_ml((train.X - mean) / std)
            valid_X_scaled = df_to_numpy_for_ml((valid.X - mean) / std)
            test_X_scaled = df_to_numpy_for_ml((test.X - mean) / std)

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
                train_X_i = train_X[pd.Index(train_X["run_no"] == i)]
                valid_X_i = valid_X[pd.Index(valid_X["run_no"] == i)]
                test_X_i = test_X[pd.Index(test_X["run_no"] == i)]

                mean = train_X_i.mean()
                std = train_X_i.std()

                train_X_i_scaled = df_to_numpy_for_ml((train_X_i - mean) / std)
                valid_X_i_scaled = df_to_numpy_for_ml((valid_X_i - mean) / std)
                test_X_i_scaled = df_to_numpy_for_ml((test_X_i - mean) / std)

                train_X_scaled[pd.Index(train_X["run_no"] == i)] = train_X_i_scaled
                valid_X_scaled[pd.Index(valid_X["run_no"] == i)] = valid_X_i_scaled
                test_X_scaled[pd.Index(test_X["run_no"] == i)] = test_X_i_scaled

            train = train._replace(X=train_X_scaled)
            valid = valid._replace(X=valid_X_scaled)
            test = test._replace(X=test_X_scaled)

        return train, valid, test

    @staticmethod
    def one_hot_encode(train: data, valid: data, test: data) -> tuple:
        """
        Function transforms the labels from integers to one hot vectors.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training of type named tuple
        :param valid: data for validation of type named tuple
        :param test: data for testing of type named tuple
        :return: train, valid, test: Tuple with data of type named tuple
        """
        train_y = df_to_numpy_for_ml(train.y.loc[:, train.y.columns != "run_no"])
        valid_y = df_to_numpy_for_ml(valid.y.loc[:, valid.y.columns != "run_no"])
        test_y = df_to_numpy_for_ml(test.y.loc[:, test.y.columns != "run_no"])

        enc = OneHotEncoder(categories='auto').fit(train_y.reshape(-1, 1))
        train = train._replace(y=enc.transform(train_y.reshape(-1, 1)).toarray())
        valid = valid._replace(y=enc.transform(valid_y.reshape(-1, 1)).toarray())
        test = test._replace(y=enc.transform(test_y.reshape(-1, 1)).toarray())

        return train, valid, test


def df_to_numpy_for_ml(df):
    out = df.to_numpy(dtype=float)
    out = out[..., np.newaxis]
    out = np.nan_to_num(out)

    return out
