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

        # select 2.5% of the healthy pulses randomly
        selection[is_healthy] = np.random.choice(a=[True, False], size=(sum(is_healthy),), p=[0.025, 0.975])

    return selection


def hdf_ext_link_to_da_selection(file_path, selection, feature_list) -> xr.DataArray:
    with h5py.File(file_path, "r") as file:
        # find name of groups to be read
        groups_list = list(file.keys())
        list_of_events = list(compress(groups_list, selection))

        # buffer for data
        data_array = np.empty(shape=(len(list_of_events), 1600, len(feature_list)))
        event_timestamps = []
        for event_ind, event in enumerate(list_of_events):
            # get timestamp from group attributes
            timestamp = file[event].attrs.get("Timestamp")
            event_timestamps.append(np.datetime64(timestamp))

            # read features
            for feature_ind, feature in enumerate(feature_list):
                data = file[event][feature][:]
                ts_length = len(data)

                # Interpolate if time series is not 3200 points
                if ts_length < 3200:
                    x_low = np.linspace(0, 1, num=ts_length, endpoint=True)
                    x_high = np.linspace(0, 1, num=3200, endpoint=True)
                    interpolate = interp1d(x_low, data, kind='linear')
                    data = interpolate(x_high)

                # Downsample by taking every 2nd sample
                # TODO: better way to do this?
                data = data[::2]
                data_array[event_ind, :, feature_ind] = data

    # Create xarray DataArray
    dim_names = ["event", "time", "feature"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    sample_time = 1.25e-9
    time_steps = np.multiply(np.array(list(range(1600))), sample_time)
    da = xr.DataArray(data=data_array,
                      dims=dim_names,
                      coords={"event": event_timestamps,
                              "time": time_steps,
                              "feature": feature_names
                              }
                      )
    return da


def da_to_numpy_for_ml(da: xr.DataArray) -> xr.DataArray:
    out = da.values
    out = np.nan_to_num(out)
    return out
