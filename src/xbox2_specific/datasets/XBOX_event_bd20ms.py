from pathlib import Path
import typing
from collections import namedtuple
import h5py
import numpy as np
import pandas as pd
from src.utils.hdf_tools import hdf_to_df_selection


def select_data(context_data_file_path: Path) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    returns all breakdown events (no distinction between runs) and 2.5% of the healthy events (chosen randomly).
    filters out healthy events where the prev trend data is further away than two seconds.
    :return: X and y prepared or machine learning
    """
    with h5py.File(context_data_file_path, "r") as file:
        is_bd_in_next_pulse = file["is_bd_in_20ms"][:]

        event_ts = file["Timestamp"][:]
        trend_ts = file["PrevTrendData/Timestamp"][:]
        time_diff = event_ts - trend_ts
        time_diff_threshold = pd.to_timedelta(2, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold

        # only define healthy pulses with a time difference to the previous trend data of < 2s
        is_healthy = file["clic_label/is_healthy"][:] & filter_timestamp_diff

        # select all breakdown and directly preceding pulses
        selection = is_bd_in_next_pulse

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    df = hdf_to_df_selection(context_data_file_path, selection=selection)

    clm_for_training = df.columns.difference(pd.Index(["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_healthy",
                                                       "is_bd_in_20ms", "is_bd_in_40ms"]))
    X = df[clm_for_training].to_numpy(dtype=float)
    X = X[..., np.newaxis]
    X = np.nan_to_num(X)
    y = df["is_healthy"].to_numpy(dtype=bool)
    return X, y


