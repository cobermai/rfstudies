"""example code how to select from context data and prepare data for machine learning. """
import typing
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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


if __name__ == '__main__':
    c_path = Path("~/cernbox_projects_local/CLIC_data_transfert/Xbox2_hdf/context.hdf").expanduser()

    X, y = select_data(context_data_file_path=c_path)
