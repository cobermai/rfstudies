"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
import typing
from collections import namedtuple
import h5py
import numpy as np
import pandas as pd
from src.utils.hdf_tools import hdf_to_df_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def select_data(context_data_file_path: Path) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    returns all breakdown events (no distinction between runs) and 2.5% of the healthy events (chosen randomly).
    filters out healthy events where the prev trend data is further away than two seconds.
    :return: X and y prepared or machine learning
    """
    with h5py.File(context_data_file_path, "r") as file:
        is_bd_in_two_pulses = file["is_bd_in_40ms"][:]
        is_bd_in_next_pulse = file["is_bd_in_20ms"][:]
        is_bd = file["is_bd"][:]

        #
        event_ts = file["Timestamp"][:]
        trend_ts = file["PrevTrendData/Timestamp"][:]
        time_diff = event_ts - trend_ts
        time_diff_threshold = pd.to_timedelta(2, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold

        # only define healthy pulses with a time difference to the previous trend data of < 2s
        is_healthy = file["clic_label/is_healthy"][:] & filter_timestamp_diff

        # select all breakdown and directly preceding pulses
        selection = (is_bd_in_two_pulses | is_bd_in_next_pulse | is_bd)

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    df = hdf_to_df_selection(context_data_file_path, selection=selection)

    clm_for_training = df.columns.difference(pd.Index(["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_healthy",
                                                       "is_bd_in_20ms", "is_bd_in_40ms"]))
    X = df[clm_for_training].to_numpy(dtype=float)
    y = df["is_healthy"].to_numpy(dtype=bool)
    return X, y


def scale_data(X):
    """
    function scales data for prediction with standard scaler
    :param X: data array of shape (event, sample, feature)
    :return: X_scaled: scaled data array of shape (event, sample, feature)
    """
    X = X[..., np.newaxis]
    X = np.nan_to_num(X)

    X_scaled = np.zeros_like(X)
    for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
        X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
    return X_scaled

def train_valid_test_split(X, y, splits: tuple) -> typing.Tuple:
    """
    Splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test)
    :return: train, valid, test: Tuple with data of type named tuple
    """
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
