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

    clm_for_training = pd.Index(["PrevTrendData__Loadside_win", "PrevTrendData__Tubeside_win",
                                 "PrevTrendData__Collector", "PrevTrendData__Gun", "PrevTrendData__IP_before_PC",
                                 "PrevTrendData__PC_IP", "PrevTrendData__WG_IP", "PrevTrendData__IP_Load",
                                 "PrevTrendData__IP_before_structure", "PrevTrendData__US_Beam_Axis_IP",
                                 "PrevTrendData__Klystron_Flange_Temp", "PrevTrendData__Load_Temp",
                                 "PrevTrendData__PC_Left_Cavity_Temp", "PrevTrendData__PC_Right_Cavity_Temp",
                                 "PrevTrendData__Bunker_WG_Temp", "PrevTrendData__Structure_Input_Temp",
                                 "PrevTrendData__Chiller_1", "PrevTrendData__Chiller_2", "PrevTrendData__Chiller_3",
                                 "PrevTrendData__PKI_FT_avg", "PrevTrendData__PSI_FT_avg", "PrevTrendData__PSR_FT_avg",
                                 "PrevTrendData__PSI_max", "PrevTrendData__PSR_max", "PrevTrendData__PEI_max",
                                 "PrevTrendData__DC_Down_min", "PrevTrendData__DC_Up_min",
                                 "PrevTrendData__PSI_Pulse_Width"])
    X = df[clm_for_training].to_numpy(dtype=float)
    X = X[..., np.newaxis]
    X = np.nan_to_num(X)
    y = df["is_healthy"].to_numpy(dtype=bool)
    return X, y


if __name__ == '__main__':
    X, y = select_data(Path('C:\\Users\\holge\\cernbox\\CLIC_data\\Xbox2_hdf\\context.hdf'))


