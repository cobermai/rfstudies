from pathlib import Path
import typing
import h5py
import numpy as np
import pandas as pd


def read_hdf_dataset(file: h5py.File, key: str):
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


def select_trend_data_events(event_timestamps: np.ndarray,
                             trend_timestamps: np.ndarray,
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


def select_events_from_list(context_data_file_path: Path, selection_list: typing.List) -> pd.DataFrame:
    """
    Function for creating selection filter of events in xbox2 data set
    :param context_data_file_path: path to context data file
    :param selection_list: events to include in selection
    :return selection: boolean filter for selecting breakdown events
    """
    with h5py.File(context_data_file_path, "r") as file:
        # load relevant data from context file
        features_read = []
        for key in selection_list:
            features_read.append(read_hdf_dataset(file, key))
        selection = features_read[0]
        for event_index in range(1, len(features_read)):
            selection = selection | features_read[event_index]

        event_timestamps = read_hdf_dataset(file, "Timestamp")
        trend_timestamp = read_hdf_dataset(file, "PrevTrendData/Timestamp")

        # only define healthy pulses with a time difference to the previous trend data of less than 2 s
        filter_timestamp_diff = select_trend_data_events(event_timestamps, trend_timestamp, 2)
        is_healthy = read_hdf_dataset(file, "clic_label/is_healthy") & filter_timestamp_diff

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    return selection


def select_features_from_list(df: pd.DataFrame, selection_list) -> np.ndarray:
    """
    returns features of selected events for modeling
    :param df: dataframe with selected events
    :param selection_list: list of features to include in selection
    :return X: label of selected events
    """
    feature_names = pd.Index(selection_list)

    X = df[feature_names].to_numpy(dtype=float)
    X = X[..., np.newaxis]
    X = np.nan_to_num(X)

    return X