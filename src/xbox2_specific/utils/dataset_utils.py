from pathlib import Path
import typing
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from typing import List


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


def read_hdf_dataset_selection(file: h5py.File, key: str, selection: typing.List[bool]):
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

        # filter data on PSI Amplitude
        scaling_coeff1 = -1317300
        scaling_coeff2 = 7.8582e7
        PSI_amplitude_event = read_hdf_dataset(file, "PSI Amplitude/pulse_amplitude")
        PSI_amplitude_event_recomputed = (scaling_coeff1 * PSI_amplitude_event) + \
                                         (scaling_coeff2 * (PSI_amplitude_event ** 2))
        PSI_filter_threshold = 650000  # defined by XBOX2 expert William Lee Millar 08.06.2020
        PSI_amplitude_filter = PSI_amplitude_event_recomputed > PSI_filter_threshold
        selection = selection & PSI_amplitude_filter

    return selection


def event_ext_link_hdf_to_da_timestamp(file_path: Path, ext_link_index: np.ndarray, feature_list: list) -> xr.DataArray:
    """
    Function that reads features from external link hdf file and returns data as xarray DataArray
    :param file_path: path to data files
    :param ext_link_index: array with index of events to select in external link file
    :param feature_list: list of feature names to be included in data
    """
    file_path = r'{}'.format(file_path)
    with h5py.File(file_path, "r") as file:
        # find name of groups to be read
        groups_list = list(file.keys())

        # buffer for data
        data = np.empty(shape=(len(ext_link_index), 1600, len(feature_list)))
        timestamps_found = []
        events = np.array(groups_list)[ext_link_index]
        for ind, event in enumerate(events):
            timestamp = np.datetime64(file[event].attrs.__getitem__("Timestamp").decode('utf8'))
            # read features
            timestamps_found.append(timestamp)
            for feature_ind, feature in enumerate(feature_list):
                data_feature = file[event][feature][:]
                data_feature = scale_signal(data_feature, feature)
                ts_length = len(data_feature)
                # Interpolate if time series is not 3200 points
                standard_signal_length = 3200
                if ts_length < standard_signal_length:
                    x_low = np.linspace(0, 1, num=ts_length, endpoint=True)
                    x_high = np.linspace(0, 1, num=3200, endpoint=True)
                    interpolate = interp1d(x_low, data_feature, kind='linear')
                    data_feature = interpolate(x_high)
                # Downsample by taking every 2nd sample
                # TODO: better way to do this?
                data_feature = data_feature[::2]
                data[ind, :, feature_ind] = data_feature

    # Create xarray DataArray
    dim_names = ["event", "sample", "feature"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    data_array = xr.DataArray(data=data,
                              dims=dim_names,
                              coords={"feature": feature_names}
                              )
    data_array = data_array.assign_coords(timestamp_event=("event", np.array(timestamps_found)))
    return data_array


def shift_values(array: np.ndarray, num: int, fill_value=np.nan):
    """
    Function for shifting values of a 1D array
    :param array: The array which elements should be shifted
    :param num: The number of times the array should be shifted (positive -> right, negative -> left).
    :param fill_value: The value to fill in the spots shifted into the array.
    """
    result = np.empty_like(array)
    if num > 0:
        result[:num] = fill_value
        result[num:] = array[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = array[-num:]
    else:
        result[:] = array
    return result


def determine_followup(bd_label: np.ndarray, timestamp: np.ndarray, threshold: typing.Any) -> np.ndarray:
    """
    Function that takes breakdown labels and timestamps as input and
    returns a boolean array with 1's representing followup breakdowns.
    :param bd_label: boolean array specifying whether a breakdown happened.
    :param timestamp: an array of timestamps for bd_label.
    :param threshold: threshold in seconds for followup bds.
    :return: boolean array which specifies whether bd_label indexes are followup bds.
    """
    is_followup = np.zeros_like(bd_label)
    ind_last_bd_in_20ms = 0
    for index in range(1, len(bd_label)):
        if bd_label[index]:
            if (ind_last_bd_in_20ms != 0) \
                    and ((timestamp[index] - timestamp[ind_last_bd_in_20ms]) < np.timedelta64(threshold, 's')):
                is_followup[index] = True
            ind_last_bd_in_20ms = index
    return is_followup


def da_to_numpy_for_ml(data_array: xr.DataArray) -> np.ndarray:
    """
    Function that takes raw values of xarray, replaces NaN with zero and infinity with large finite numbers
    :param data_array: xarray DataArray
    :return: numpy array ready for machine learning algorithms
    """
    out = data_array.values
    out = np.nan_to_num(out)
    return out


def scale_signal(signal, feature_name):
    if feature_name == "PEI Amplitude":
        coeff0 = 0
        coeff1 = -378810
        coeff2 = 4.4043e7
        return coeff0 + (coeff1 * signal) + (coeff2 * (signal ** 2))
    elif feature_name == "PSI Amplitude":
        coeff0 = 0
        coeff1 = -1317300
        coeff2 = 7.8582e7
        return coeff0 + (coeff1 * signal) + (coeff2 * (signal ** 2))
    elif feature_name == "PSR Amplitude":
        return signal  # no scaling
    elif feature_name == "PKI Amplitude":
        coeff0 = 0
        coeff1 = -1240300
        coeff2 = 5.2222e7
        return coeff0 + (coeff1 * signal) + (coeff2 * (signal ** 2))
    elif feature_name == "DC Up":
        return signal  # no scaling
    elif feature_name == "DC Down":
        return signal  # no scaling


def read_features_from_selection(data_path: Path, feature_list: List[str], selection: List[bool]) -> xr.DataArray:
    with h5py.File(data_path / "context.hdf", 'r') as file:
        ext_link_index = read_hdf_dataset(file, "event_ext_link_index")[selection]
    data_array = event_ext_link_hdf_to_da_timestamp(file_path=data_path / "EventDataExtLinks.hdf",
                                                    ext_link_index=ext_link_index,
                                                    feature_list=feature_list)
    return data_array


def read_label_and_meta_data_from_selection(data_path: Path, label_name: str, selection: List[bool]):
    with h5py.File(data_path / "context.hdf", 'r') as file:
        is_bd_in_20ms = read_hdf_dataset(file, label_name)[selection]
        timestamp = read_hdf_dataset(file, "Timestamp")[selection]
        run_no = read_hdf_dataset(file, "run_no")[selection]
    return is_bd_in_20ms, timestamp, run_no
