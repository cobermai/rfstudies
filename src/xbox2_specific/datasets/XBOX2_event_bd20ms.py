"""Selecting from context data and prepare dataset XBOX2_event_bd20ms for machine learning. """
from pathlib import Path
import typing
import h5py
import numpy as np
import pandas as pd
from pandas import Timestamp
from src.utils.dataset_creator import DatasetCreator


class XBOX2EventBD20msSelect(DatasetCreator):
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
        selection_list = ["DC_Down__D1", "DC_Down__D9", "DC_Down__tsfresh__mean", "DC_Down__tsfresh__maximum",
                          "DC_Down__tsfresh__median", "DC_Down__tsfresh__minimum",
                          "DC_Up__D1", "DC_Up__D9", "DC_Up__tsfresh__mean", "DC_Up__tsfresh__maximum",
                          "DC_Up__tsfresh__median", "DC_Up__tsfresh__minimum",
                          "PEI_Amplitude__pulse_length", "PEI_Amplitude__pulse_amplitude",
                          "PKI_Amplitude__pulse_length", "PKI_Amplitude__pulse_amplitude",
                          "PSI_Amplitude__pulse_length", "PSI_Amplitude__pulse_amplitude"]
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

    @staticmethod
    def select_run(df: pd.DataFrame, run_no: int):
        if run_no == 0 or run_no < -9 or run_no > 9:
            raise ValueError("Run number does not exist")

        # Hardcoded timestamps for classifying runs. Format is [start, stop, run_no]. Negative run_no is commissioning.
        timestamp_list_run = np.array([
            [Timestamp('2018-05-15 21:16:59.626459'), Timestamp('2018-05-23 14:41:58.024856'), -1],
            [Timestamp('2018-05-23 14:42:58.036909'), Timestamp('2018-06-01 11:41:23.304752'), 1],
            [Timestamp('2018-06-01 11:42:23.326588'), Timestamp('2018-06-10 09:57:46.167949'), -2],
            [Timestamp('2018-06-10 09:58:46.176918'), Timestamp('2018-06-20 10:49:26.504749'), 2],
            [Timestamp('2018-06-20 10:50:26.517873'), Timestamp('2018-06-27 01:56:14.100424'), -3],
            [Timestamp('2018-06-27 01:57:14.092206'), Timestamp('2018-07-17 08:49:18.581234'), 3],
            [Timestamp('2018-07-17 08:50:18.644042'), Timestamp('2018-07-17 14:24:22.808270'), -4],
            [Timestamp('2018-07-17 14:25:22.808983'), Timestamp('2018-07-26 15:56:55.238768'), 4],
            [Timestamp('2018-07-26 15:57:55.254684'), Timestamp('2018-07-30 12:58:28.800693'), -5],
            [Timestamp('2018-07-30 12:59:28.850502'), Timestamp('2018-08-09 07:18:19.717621'), 5],
            [Timestamp('2018-08-09 07:19:19.717776'), Timestamp('2018-08-16 07:48:45.260491'), -6],
            [Timestamp('2018-08-16 07:49:45.217265'), Timestamp('2018-08-22 19:07:06.581874'), 6],
            [Timestamp('2018-08-24 22:53:03.560161'), Timestamp('2018-08-27 20:21:22.319445'), -7],
            [Timestamp('2018-08-27 20:22:22.331644'), Timestamp('2018-09-03 09:53:18.547360'), 7],
            [Timestamp('2018-09-03 09:54:18.540067'), Timestamp('2018-09-05 16:48:36.589576'), -8],
            [Timestamp('2018-09-05 16:49:36.595947'), Timestamp('2018-09-17 06:27:33.398326'), 8],
            [Timestamp('2018-09-17 06:28:33.412608'), Timestamp('2018-09-19 00:05:14.894480'), -9],
            [Timestamp('2018-09-19 00:06:14.912150'), Timestamp('2018-09-25 09:51:59.222968'), 9]
        ])

        # Select specified run number
        for run in timestamp_list_run:
            if run[2] == run_no:
                run_start = run[0]
                run_end = run[1]

        selection = (df['Timestamp'] >= run_start) & (df['Timestamp'] <= run_end)
        return selection


if __name__ == '__main__':
    selector = XBOX2EventBD20msSelect()
    with h5py.File(Path('C:\\Users\\holge\\cernbox\\CLIC_data\\Xbox2_hdf\\context.hdf'), "r") as file:
        timestamps_raw = selector.read_hdf_dataset(file, "Timestamp")
    df = pd.DataFrame({'Timestamp': timestamps_raw})

    run_selection = selector.select_run(df, 9)

    print(run_selection.sum())
