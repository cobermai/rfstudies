"""example code how to select from context data and prepare data for machine learning. """
import typing
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple
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

def train_valid_test_split(X, y, splits: tuple) -> typing.Tuple:
    """
    splits data into training, testing and validation set using random sampling
    :param X: input data array of shape (event, sample, feature)
    :param y: output data array of shape (event)
    :param splits: tuple specifying splitting fractions (training, validation, test)
    :return: train, valid, test: tuple containing split data as named tuples
    """
    if sum(splits)==1 and len(splits)==3 and all(splits>0):
        raise Warning('Splits should sum to 1 and have three float values between 0 and 1')
    if splits(0)==0:
        raise ValueError('Training set fraction cannot be 1')

    idx = np.arange(len(X))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = \
        train_test_split(X, y, idx, test_size=1-splits(0))

    X_valid, X_test, y_valid, y_test, idx_valid, idx_test = \
        train_test_split(X_temp, y_temp, idx_temp, test_size=splits(1)/(1-(splits(0))))

    data = namedtuple("data", ["X", "y", "id"])
    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)
    test = data(X_test, y_test, idx_test)

    return train, valid, test


if __name__ == '__main__':
    c_path = Path("~/cernbox_projects_local/CLIC_data_transfert/Xbox2_hdf/context.hdf").expanduser()

    X, y = select_data(context_data_file_path=c_path)

    train_test_split(X, y, (0.7, 0.2, 0.1))
