"""Selecting from context data and prepare dataset XBOX2_trend_bd20ms for machine learning. """
from collections import namedtuple
from pathlib import Path
import typing
from typing import Optional
import h5py
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn.preprocessing import StandardScaler
from src.utils.dataset_creator import DatasetCreator
from src.utils.hdf_tools import hdf_to_df_selection


class XBOX2TrendBD20msSelect(DatasetCreator):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    def __init__(self, runs_list: typing.List):
        """
        :param runs_list: list of runs to be included in dataset
        """
        super().__init__()
        self.runs = runs_list

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

    @staticmethod
    def select_run(df: pd.DataFrame, run_no: int):
        """
        Function which generates data selection filter based on run number
        :param df: pandas dataframe with xbox2 data
        :param run_no: xbox2 run number. Negative run_no is commissioning.
        :return selection: boolean selection filter for selecting data from specific runs
        """
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
        run_index = []
        timestamp = df["Timestamp"]
        for run in timestamp_list_run:
            if run[2] == run_no:
                run_start = run[0]
                run_end = run[1]
                run_index = df.index[(timestamp >= run_start) & (timestamp <= run_end)]
                break

        return run_index

    def scale_data(self, X: np.ndarray) -> np.ndarray:
        """
        Function scales data for with sklearn standard scaler.
        For trend data the data is scaled for each run.
        E.g., run 1 data is scaled by the mean and std of run 1 data etc.
        :param X: data array of shape (event, sample, feature)
        :return: X_scaled: scaled data array of shape (event, sample, feature)
        """
        X_scaled = np.zeros_like(X)
        for run in self.runs:
            run_index = self.select_run(self.df, run)
            for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
                X_scaled[run_index, :, feature_index] = StandardScaler().fit_transform(
                    X[run_index, :, feature_index].T).T
        return X_scaled

    def train_valid_test_split(self, X: np.ndarray, y: np.ndarray,
                               splits: Optional[tuple] = None) -> tuple:
        """
        Function splits data into training, testing and validation set using random sampling. Note that this function
        can be overwritten in the concrete dataset selection.
        :param X: input data array of shape (event, sample, feature)
        :param y: output data array of shape (event)
        :param splits: tuple specifying splitting fractions (training, validation, test)
        :return: train, valid, test: Tuple with data of type named tuple
        """
        if splits is None:
            splits = ([1, 2, 3, 4, 5, 6, 7], 8, 9)
            runs_train = self.select_run(self.df, 1)
            for run in splits[0]:
                runs_train = runs_train.append(self.select_run(self.df, run))
            runs_valid = self.select_run(self.df, splits[1])
            runs_test = self.select_run(self.df, splits[2])

        idx = np.arange(len(X))
        X_train, y_train, idx_train = X[runs_train], y[runs_train], idx[runs_train]
        X_valid, y_valid, idx_valid = X[runs_valid], y[runs_valid], idx[runs_valid]
        X_test, y_test, idx_test = X[runs_test], y[runs_test], idx[runs_test]

        data = namedtuple("data", ["X", "y", "idx"])
        train = data(X_train, y_train, idx_train)
        valid = data(X_valid, y_valid, idx_valid)
        test = data(X_test, y_test, idx_test)

        return train, valid, test

if __name__ == '__main__':
    selector = XBOX2TrendBD20msSelect(runs_list=list(range(10)))
    hdf_dir = Path('C:\\Users\\holge\\cernbox\\CLIC_data\\Xbox2_hdf')
    selector.event_selection = selector.select_events(context_data_file_path=hdf_dir / "context.hdf")
    selector.df = hdf_to_df_selection(hdf_dir / "context.hdf", selection=selector.event_selection)

    run_selection = selector.select_run(selector.df, 1)
    for i in range(2, 10):
        run_selection = run_selection.append(selector.select_run(selector.df, i))

    selector.X = selector.select_features(df=selector.df)
    selector.y = selector.select_labels(df=selector.df)

    print("Elements in event selection:")
    print(sum(selector.event_selection))
    print("Elements in runs")
    print(sum(run_selection))
    print("Elements from events in runs")
    print(sum(selector.event_selection))
    print("Breakdowns in event selection")
    print(len(selector.y) - sum(selector.y))
    print("Breakdowns in from events in run selection")
    print(len(selector.y[run_selection]) - sum(selector.y[run_selection]))