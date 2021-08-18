"""Selecting from context data and prepare dataset XBOX2_event_bd20ms for machine learning. """
from pathlib import Path
import typing
import h5py
import numpy as np
import pandas as pd
from src.utils.dataset_creator import DatasetCreator


class XBOX2TrendBD20msSelect(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def select_trend_data_events(event_timestamps: np.datetime64,
                                 trend_timestamps: np.datetime64,
                                 time_threshold: float) -> bool:
        """
        Selects trend data timestamps for filtering healthy pulses with time diff more than threshold.
        :param event_timestamps: array of event data timestamps
        :param trend_timestamps: array of trend data timestamps
        :param time_threshold: threshold in seconds
        :return: filter for boolean indexing
        """
        time_diff = event_timestamps - trend_timestamps
        time_diff_threshold = pd.to_timedelta(time_threshold, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold
        return filter_timestamp_diff

    def select_events(self, context_data_file_path: Path) -> typing.List[bool]:
        """
        selection of events in data
        :param context_data_file_path: path to context data file
        :return selection: boolean filter for selecting breakdown events
        """
        selection_list = ["is_bd_in_20ms"]
        with h5py.File(context_data_file_path, "r") as file:
            # load relevant data from context file
            features_read = []
            for key in selection_list:
                features_read.append(self.read_hdf_dataset(file, key))
            selection = features_read[0]
            for event_index in range(1, len(features_read)):
                selection = selection | features_read[event_index]

            event_timestamps = self.read_hdf_dataset(file, "Timestamp")
            trend_timestamp = self.read_hdf_dataset(file, "PrevTrendData/Timestamp")

            # only define healthy pulses with a time difference to the previous trend data of less than 2 s
            filter_timestamp_diff = self.select_trend_data_events(event_timestamps, trend_timestamp, 2)
            is_healthy = self.read_hdf_dataset(file, "clic_label/is_healthy") & filter_timestamp_diff

            # select 2.5% of the healthy pulses randomly
            selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])
            return selection

    def select_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        returns features of selected events for modeling
        :param df: dataframe with selected events
        :return X: label of selected events
        """
        selection_list = ["PrevTrendData__Loadside_win", "PrevTrendData__Tubeside_win",
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
                          "PrevTrendData__PSI_Pulse_Width"]
        feature_names = pd.Index(selection_list)

        X = df[feature_names].to_numpy(dtype=float)
        X = X[..., np.newaxis]
        X = np.nan_to_num(X)

        return X

    def select_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        returns labels of selected events for supervised machine learning
        :param df: dataframe with selected events
        :return y: label of selected events
        """
        y = df["is_healthy"].to_numpy(dtype=bool)
        return y
