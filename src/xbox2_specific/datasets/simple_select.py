"""example code how to select from context data and prepare data for machine learning. """
from pathlib import Path
import typing
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.hdf_tools import hdf_to_df_selection
from src.utils.handler_tools.dataset_creator import DatasetCreator

class SimpleSelect(DatasetCreator):

    def scale_data(self, X):
        """
        function scales data for prediction with standard scaler
        :param X: data array of shape (event, sample, feature)
        :return: X_scaled: scaled data array of shape (event, sample, feature)
        """
        X_scaled = np.zeros_like(X)
        for feature_index in range(len(X[0, 0, :])):  # Iterate through feature
            X_scaled[:, :, feature_index] = StandardScaler().fit_transform(X[:, :, feature_index].T).T
        return X_scaled


    def read_hdf_dataset(self, file: h5py.File, key: str):
        """
        Read dataset from hdf file
        :param file: h5py File object with read access
        :param key: string specifying key for dataset to read
        :return: array containing contents of h5py dataset
        """
        dataset = file[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("Specified key does not yield a hdf dataset")
        return dataset[:]


    def create_filter(self, event_timestamps: np.datetime64, trend_timestamps: np.datetime64, time_threshold: float) -> bool:
        """
        Creates filter from event data and trend data timestamps for filtering healthy pulses
        with time diff more than threshold.
        :param event_timestamps: array of event data timestamps
        :param trend_timestamps: array of trend data timestamps
        :param time_threshold: threshold in seconds
        :return: filter for boolean indexing
        """
        time_diff = event_timestamps - trend_timestamps
        time_diff_threshold = pd.to_timedelta(time_threshold, "s")
        filter_timestamp_diff = time_diff < time_diff_threshold
        return filter_timestamp_diff


    def load_X_data(self, df: pd.DataFrame, index: pd.Index) -> np.ndarray:
        """
        Loads data to numpy array from dataframe
        :param df: Pandas dataframe containing data
        :param index: Pandas index variable with features for use in machine learning
        :return X: Numpy array with data loaded from dataframe
        """
        X = df[index].to_numpy(dtype=float)
        X = X[..., np.newaxis]
        X = np.nan_to_num(X)
        return X


    def create_breakdown_selection_filter(self, context_data_file_path: Path,
                                          selection_list: typing.List[str]) -> typing.List[bool]:
        """
        Creates a filter for selecting breakdown events
        :param context_data_file_path: path to context data file
        :param selection_list: List of features used for selecting breakdowns
        :return selection: boolean filter for selecting breakdown events
        """
        with h5py.File(context_data_file_path, "r") as file:
            # load relevant data from context file
            features_read = []
            for key in selection_list:
                features_read.append(self.read_hdf_dataset(file, key))
            selection = features_read[0]
            for i in range(1, len(features_read)):
                selection = selection | features_read[i]

            event_timestamps = self.read_hdf_dataset(file, "Timestamp")
            trend_timestamp = self.read_hdf_dataset(file, "PrevTrendData/Timestamp")

            # only define healthy pulses with a time difference to the previous trend data of less than 2 s
            filter_timestamp_diff = self.create_filter(event_timestamps, trend_timestamp, 2)
            is_healthy = self.read_hdf_dataset(file, "clic_label/is_healthy") & filter_timestamp_diff

            # select 2.5% of the healthy pulses randomly
            selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])
            return selection

    def select_data(self, context_data_file_path: Path) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Returns all breakdown events (no distinction between runs) and 2.5% of the healthy events (chosen randomly).
        filters out healthy events where the prev trend data is further away than two seconds.
        :param context_data_file_path: Path to context.hdf file
        :return X_scaled, y_hot: X and y prepared for machine learning
        """
        # select all breakdown events and the two directly preceding pulses
        breakdown_selection = ["is_bd_in_40ms", "is_bd_in_20ms", "is_bd"]
        selection = self.create_breakdown_selection_filter(context_data_file_path, breakdown_selection)
        df = hdf_to_df_selection(context_data_file_path, selection=selection)
        # Use all features apart from
        feature_names = ["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_healthy", "is_bd_in_20ms", "is_bd_in_40ms"]
        clm_for_training = df.columns.difference(pd.Index(feature_names))
        # Load features and labels
        X = self.load_X_data(df, clm_for_training)
        y = df["is_healthy"].to_numpy(dtype=bool)

        X_scaled = self.scale_data(X)
        y_hot = self.one_hot_encode(y)
        return X_scaled, y_hot
