from pathlib import Path
import typing
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
        # load relevant data from context file
        is_bd_in_next_pulse = file["is_bd_in_20ms"][:]
        event_timestamp = file["Timestamp"][:]
        trend_timestamp = file["PrevTrendData/Timestamp"][:]

        # filter healthy pulses with a time difference to previous trend data more than 2 s
        # only define healthy pulses with a time difference to the previous trend data of < 2s
        time_diff = event_timestamp - trend_timestamp
        time_diff_threshold = pd.to_timedelta(2, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold
        is_healthy = file["clic_label/is_healthy"][:] & filter_timestamp_diff

        # select events with breakdown in next pulse
        selection = is_bd_in_next_pulse

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    df = hdf_to_df_selection(context_data_file_path, selection=selection)

    clm_for_training = pd.Index(["DC_Down__D1", "DC_Down__D9", "DC_Down__tsfresh__mean", "DC_Down__tsfresh__maximum",
                                 "DC_Down__tsfresh__median", "DC_Down__tsfresh__minimum",
                                 "DC_Up__D1", "DC_Up__D9", "DC_Up__tsfresh__mean", "DC_Up__tsfresh__maximum",
                                 "DC_Up__tsfresh__median", "DC_Up__tsfresh__minimum",
                                 "PEI_Amplitude__pulse_length", "PEI_Amplitude__pulse_amplitude",
                                 "PKI_Amplitude__pulse_length", "PKI_Amplitude__pulse_amplitude",
                                 "PSI_Amplitude__pulse_length", "PSI_Amplitude__pulse_amplitude"])
    X = df[clm_for_training].to_numpy(dtype=float)
    X = X[..., np.newaxis]
    X = np.nan_to_num(X)
    y = df["is_healthy"].to_numpy(dtype=bool)
    return X, y


if __name__ == '__main__':
    X, y = select_data(Path('C:\\Users\\holge\\cernbox\\CLIC_data\\Xbox2_hdf\\context.hdf'))


