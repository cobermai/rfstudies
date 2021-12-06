from pathlib import Path
import numpy as np
import h5py


def assign_run_no(timestamps: np.datetime64):
    """
    :param timestamps: array of available timestamps in data
    :return runs_assigned: array with assigned run numbers for each input timestamp
    """
    # Hardcoded timestamps for classifying runs. Format is [start, stop, run_no]. Negative run_no is commissioning.
    timestamp_list_run = np.array([
        [np.datetime64('2018-05-15 21:16:59.626459'), np.datetime64('2018-05-23 14:41:58.024856'), -1],
        [np.datetime64('2018-05-23 14:42:58.036909'), np.datetime64('2018-06-01 11:41:23.304752'), 1],
        [np.datetime64('2018-06-01 11:42:23.326588'), np.datetime64('2018-06-10 09:57:46.167949'), -2],
        [np.datetime64('2018-06-10 09:58:46.176918'), np.datetime64('2018-06-20 10:49:26.504749'), 2],
        [np.datetime64('2018-06-20 10:50:26.517873'), np.datetime64('2018-06-27 01:56:14.100424'), -3],
        [np.datetime64('2018-06-27 01:57:14.092206'), np.datetime64('2018-07-17 08:49:18.581234'), 3],
        [np.datetime64('2018-07-17 08:50:18.644042'), np.datetime64('2018-07-17 14:24:22.808270'), -4],
        [np.datetime64('2018-07-17 14:25:22.808983'), np.datetime64('2018-07-26 15:56:55.238768'), 4],
        [np.datetime64('2018-07-26 15:57:55.254684'), np.datetime64('2018-07-30 12:58:28.800693'), -5],
        [np.datetime64('2018-07-30 12:59:28.850502'), np.datetime64('2018-08-09 07:18:19.717621'), 5],
        [np.datetime64('2018-08-09 07:19:19.717776'), np.datetime64('2018-08-16 07:48:45.260491'), -6],
        [np.datetime64('2018-08-16 07:49:45.217265'), np.datetime64('2018-08-22 19:07:06.581874'), 6],
        [np.datetime64('2018-08-24 22:53:03.560161'), np.datetime64('2018-08-27 20:21:22.319445'), -7],
        [np.datetime64('2018-08-27 20:22:22.331644'), np.datetime64('2018-09-03 09:53:18.547360'), 7],
        [np.datetime64('2018-09-03 09:54:18.540067'), np.datetime64('2018-09-05 16:48:36.589576'), -8],
        [np.datetime64('2018-09-05 16:49:36.595947'), np.datetime64('2018-09-17 06:27:33.398326'), 8],
        [np.datetime64('2018-09-17 06:28:33.412608'), np.datetime64('2018-09-19 00:05:14.894480'), -9],
        [np.datetime64('2018-09-19 00:06:14.912150'), np.datetime64('2018-09-25 09:51:59.222968'), 9]
    ])

    runs_assigned = np.zeros(shape=timestamps.shape, dtype=int)
    for run in timestamp_list_run:
        run_start = run[0]
        run_end = run[1]
        runs_assigned[(timestamps >= run_start) & (timestamps <= run_end)] = run[2]
    return runs_assigned


def get_run_no(file: h5py.File):
    """
    Returns the run_number of timestamps in hdf file
    :param file: an h5py.File
    :return: array with assigned run numbers
    """
    timestamps = file["Timestamp"]
    run_no = assign_run_no(timestamps)
    return run_no


def get_event_timestamp_ext_link_index(ext_link_file: h5py.File, timestamps: np.ndarray):
    """
    Function which generates an index map from timestamps to index in external link file
    :param ext_link_file: external link file as h5py.File object
    :param timestamps: array of timestamps
    :return: map of indexes in timestamps to indexes in ext_link_file
    """
    # find name of groups to be read
    groups_list = list(ext_link_file.keys())
    timestamps_ext_link = []
    for event_ind, event in enumerate(groups_list):
        timestamps_ext_link.append(np.datetime64(ext_link_file[event].attrs.__getitem__("Timestamp").decode('utf8')))
    timestamps_ext_link = np.array(timestamps_ext_link)
    ext_link_index = np.empty_like(timestamps, dtype=int)
    for index, timestamp in enumerate(timestamps):
        ext_link_index[index] = np.where(timestamp == timestamps_ext_link)[0]
    return ext_link_index


