"""example code how to select from context data and prepare data for machine learning. """
import typing
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.hdf_tools import hdf_to_df_selection

def select_data(context_data_file_path: Path) -> typing.Tuple:
    """
    :param context_data_file_path: path to hdf5 context data file
    :return: data prepared from machine learning X of shape (event, sample, feature),
    y of shape (event, sample, feature)
    """
    with h5py.File(context_data_file_path, "r") as file:
        is_bd_in_two_pulses = file["is_bd_in_40ms"][:]
        is_bd_in_next_pulse = file["is_bd_in_20ms"][:]
        is_bd = file["is_bd"][:]

        time_diff = file["Timestamp"][:] - file["PrevTrendData/Timestamp"][:]
        time_diff_threshold = pd.to_timedelta(2, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold

        # only define healthy pulses with a time difference to the previous trend data of < 2s
        is_healthy = file["clic_label/is_healthy"][:] | filter_timestamp_diff

        # select all breakdown and directly preceding pulses
        selection = (is_bd_in_two_pulses | is_bd_in_next_pulse | is_bd)

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    df = hdf_to_df_selection(context_data_file_path, selection=selection)

    clm_for_training = df.columns.difference(pd.Index(["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_healthy",
                                                       "is_bd_in_20ms", "is_bd_in_40ms"]))
    X = df[clm_for_training].to_numpy(dtype=float)
    y = df["is_healthy"].to_numpy(dtype=int)
    return X, y

def scale_data(X):
    """
    function scales data for prediction with standard scaler
    :param X: data array of shape (event, sample, feature)
    :return: X_scaled: scaled data array of shape (event, sample, feature)
    """
    X_scaled = np.zeros_like(X)

    for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
        X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
    return X_scaled

def train_test_split(X, y, splits: tuple) -> typing.Tuple:
    """
    splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test) remainder is test
    :return: : tuple containing data split into training, testing and validation
    in the form X_train, X_valid, X_test, y_train, y_valid, y_test, idx_train, idx_valid, idx_test
    """
    # maybe check for sum of splits == 1?
    idx = np.arange(len(X))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = \
        train_test_split(X, y, idx, test_size=splits(0))

    X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
        train_test_split(X_temp, y_temp, idx_temp, teest_size=splits(2)/(1-(splits(0)))

    # idea for only a single output
    # split_data = X_train, X_valid, X_test, \
    #        y_train, y_valid, y_test, \
    #        idx_train, idx_valid, idx_test
    #
    # return split_datarain_rain_"""example code how to select from context data and prepare data for machine learning. """
import typing
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.hdf_tools import hdf_to_df_selection

def select_data(context_data_file_path: Path) -> typing.Tuple:
    """
    :param context_data_file_path: path to hdf5 context data file
    :return: data prepared from machine learning X of shape (event, sample, feature),
    y of shape (event, sample, feature)
    """
    with h5py.File(context_data_file_path, "r") as file:
        is_bd_in_two_pulses = file["is_bd_in_40ms"][:]
        is_bd_in_next_pulse = file["is_bd_in_20ms"][:]
        is_bd = file["is_bd"][:]

        event_ts = file["Timestamp"][:]
        trend_ts = file["PrevTrendData/Timestamp"][:]
        time_diff = event_ts - trend_ts
        time_diff_threshold = pd.to_timedelta(2, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold

        # only define healthy pulses with a time difference to the previous trend data of < 2s
        is_healthy = file["clic_label/is_healthy"][:] | filter_timestamp_diff

        # select all breakdown and directly preceding pulses
        selection = (is_bd_in_two_pulses | is_bd_in_next_pulse | is_bd)

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    df = hdf_to_df_selection(context_data_file_path, selection=selection)

    clm_for_training = df.columns.difference(pd.Index(["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_healthy",
                                                       "is_bd_in_20ms", "is_bd_in_40ms"]))
    X = df[clm_for_training].to_numpy(dtype=float)
    y = df["is_healthy"].to_numpy(dtype=int)
    return X, y

def scale_data(X):
    """
    function scales data for prediction with standard scaler
    :param X: data array of shape (event, sample, feature)
    :return: X_scaled: scaled data array of shape (event, sample, feature)
    """
    X_scaled = np.zeros_like(X)

    for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
        X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
    return X_scaled

def train_test_split(X, y, splits: tuple) -> typing.Tuple:
    """
    splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test) remainder is test
    :return: : tuple containing data split into training, testing and validation
    in the form X_train, X_valid, X_test, y_train, y_valid, y_test, idx_train, idx_valid, idx_test
    """
    # maybe check for sum of splits == 1?
    idx = np.arange(len(X))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = \
        train_test_split(X, y, idx, test_size=splits(0))

    X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
        train_test_split(X_temp, y_temp, idx_temp, teest_size=splits(2)/(1-(splits(0)))

    # idea for only a single output
    # split_data = X_train, X_valid, X_test, \
    #        y_train, y_valid, y_test, \
    #        idx_train, idx_valid, idx_test
    #
    # return split_datarain_rain_

    return X_train, X_valid, X_test, \
           y_train, y_valid, y_test, \
           idx_train, idx_valid, idx_test


if __name__ == '__main__':
    c_path = Path("~/cernbox_projects_local/CLIC_data_transfert/Xbox2_hdf/context.hdf").expanduser()

    X, y = select_data(context_data_file_path=c_path)

    X_train, X_valid, X_test, y_train, y_valid, y_test, idx_train, idx_valid, idx_test = train_test_split(X, y)


    return X_train, X_valid, X_test, \
           y_train, y_valid, y_test, \
           idx_train, idx_valid, idx_test


if __name__ == '__main__':
    c_path = Path("~/cernbox_projects_local/CLIC_data_transfert/Xbox2_hdf/context.hdf").expanduser()

    X, y = select_data(context_data_file_path=c_path)

    X_train, X_valid, X_test, y_train, y_valid, y_test, idx_train, idx_valid, idx_test = train_test_split(X, y)
