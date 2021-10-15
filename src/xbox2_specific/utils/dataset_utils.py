from pathlib import Path
import typing
from itertools import compress
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d


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


def read_hdf_dataset_selection(file: h5py.File, key: str, selection: bool):
    """
    Read dataset from hdf file
    :param file: h5py File object with read access
    :param key: string specifying key for dataset to read
    :param selection: boolean array for specifying which elements of hdf dataset to read
    :return: array containing contents of h5py dataset
    """
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError("Specified key does not yield a hdf dataset")
    return dataset[selection]


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


def select_events_from_list(context_data_file_path: Path, selection_list: typing.List) -> list:
    """
    Function for creating selection filter of events in xbox2 data set
    :param context_data_file_path: path to context data file
    :param selection_list: events to include in selection
    :return selection: boolean filter for selecting breakdown events
    """
    with h5py.File(context_data_file_path, "r") as file:
        # define relevant breakdown events from context file
        bds_read = []
        for key in selection_list:
            bds_read.append(read_hdf_dataset(file, key))
        bd_selection = bds_read[0]
        for event_index in range(1, len(bds_read)):
            bd_selection = bd_selection | bds_read[event_index]

        run_no = read_hdf_dataset(file, "run_no")

        # define stable runs
        stable_run = run_no > 0

        # TODO: implement selection of followup and primal bds in context file?

        selection = bd_selection & stable_run  # selected breakdowns

        # load timestamps for filtering healthy events
        event_timestamps = read_hdf_dataset(file, "Timestamp")
        trend_timestamp = read_hdf_dataset(file, "PrevTrendData/Timestamp")

        # only define healthy pulses with a time difference to the previous trend data of less than 2 s
        healthy_filter_timestamp_diff = select_trend_data_events(event_timestamps, trend_timestamp, 2)
        is_healthy = read_hdf_dataset(file, "clic_label/is_healthy") & healthy_filter_timestamp_diff & stable_run

        # also select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    return selection


def event_ext_link_hdf_to_da_selection(file_path, selection, feature_list) -> xr.DataArray:
    """
    Function that reads features from external link hdf file and returns data as xarray DataArray
    :param file_path: path to data files
    :param selection: boolean array for selecting groups in external link file
    :param feature_list: list of feature names to be included in data
    """
    with h5py.File(file_path, "r") as file:
        # find name of groups to be read
        groups_list = list(file.keys())
        list_of_events = list(compress(groups_list, selection))

        # buffer for data
        data = np.empty(shape=(len(list_of_events), 1600, len(feature_list)))
        for event_ind, event in enumerate(list_of_events):
            # read features
            for feature_ind, feature in enumerate(feature_list):
                data_feature = file[event][feature][:]
                ts_length = len(data_feature)
                # Interpolate if time series is not 3200 points
                if ts_length < 3200:
                    x_low = np.linspace(0, 1, num=ts_length, endpoint=True)
                    x_high = np.linspace(0, 1, num=3200, endpoint=True)
                    interpolate = interp1d(x_low, data_feature, kind='linear')
                    data_feature = interpolate(x_high)
                # Downsample by taking every 2nd sample
                # TODO: better way to do this?
                data_feature = data_feature[::2]
                data[event_ind, :, feature_ind] = data_feature

    # Create xarray DataArray
    dim_names = ["event", "sample", "feature"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    data_array = xr.DataArray(data=data,
                              dims=dim_names,
                              coords={"feature": feature_names}
                              )
    return data_array


def da_to_numpy_for_ml(data_array: xr.DataArray) -> np.ndarray:
    """
    Function that takes raw values of xarray, replaces NaN with zero and infinity with large finite numbers
    :param data_array: xarray DataArray
    :return: numpy array ready for machine learning algorithms
    """
    out = data_array.values
    out = np.nan_to_num(out)
    return out


def shift_values(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
