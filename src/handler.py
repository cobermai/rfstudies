"""future data handler. UNFINISHED"""
import os.path
import logging
from datetime import datetime
import itertools
import numpy as np
from pandas import to_datetime
import h5py
from pathlib import Path
from src.utils.transf_tools.gather import hdf_path_combine
from setup_logging import setup_logging
setup_logging()
LOG = logging.getLogger("test_handler")
import numpy.typing as npt
import dateutil.parser



def get_hdf_paths(file_path: Path, depth: int, hdf_path: str="/") -> list:
    """
    Returns a list of all hdf-paths of an hdf file at the given depth.
    :param file_path: file path of the hdf file
    :param depth: depth of the returning hdf path (i.e. the number of additional "/" in the output paths)
    :param hdf_path: root path to start path search
    :return: list of hdf-paths
    """
    if depth==0:
        return [hdf_path]
    elif depth>0:
        with h5py.File(file_path, "r") as hdf_file:
            return [path for key in hdf_file[hdf_path].keys()
                        for path in get_hdf_paths(file_path, depth-1, hdf_path_combine(hdf_path, key))]
    else:
        raise ValueError(f"depth has to be a non negative int, but got {depth}")


def is_iso8601(maybe_iso_str: str) -> bool:
    ret = True
    try:
        datetime.fromisoformat(maybe_iso_str)
    except ValueError:
        ret = False
    return ret


def union(source_file_path: Path, hdf_path_list: list, data_set_name: str, destination_file_path: Path):
    """
    for each dataset_name it unites (concatenates) all hdf-dataset at given depth to one large hdf-dataset in the
    destination file
    :param source_file_path:
    :param destination_file_path:
    :param depth:
    :return:
    """
    with h5py.File(source_file_path, mode="r") as source_file, \
         h5py.File(destination_file_path, mode="a") as dest_file:

        if source_file[hdf_path_list[0]][data_set_name].dtype==float:
            data = np.concatenate([source_file[path][data_set_name][:] for path in hdf_path_list])
            dest_file.create_dataset(name=data_set_name, dtype=float, data=data, chunks=True)
        elif source_file[hdf_path_list[0]][data_set_name].dtype=='S27':
            example_str = source_file[hdf_path_list[0]][data_set_name][0]
            if is_iso8601(example_str):  # is the string an iso date-time format
                if example_str[-1]=="Z" or example_str[-6:]=="+00:00":  # the timezone "ZULU" time can be parsed much faster than other timezones
                    data = [ts.replace("Z", "") for path in hdf_path_list for ts in
                            source_file[path][data_set_name][:].astype(str)]
                    data = np.array(data, dtype=np.datetime64)
                else:  # if a timezone is given, we use pandas for parsing
                    data = to_datetime([ts for path in hdf_path_list
                                           for ts in source_file[path][data_set_name][:].astype(dtype=str)])\
                            .to_numpy(dtype=np.datetime64)
                dest_file.create_dataset(name = data_set_name, data=data.astype(dtype=h5py.opaque_dtype(data.dtype)),
                                         chunks=True)
            else:
                raise NotImplementedError("Unknown data format, only know numeric, string and string in iso8601 format")
        else:
            raise NotImplementedError("Unknown data format, only know numeric, string and string in iso8601 format")

def clean(file_path: Path):
    with h5py.File(file_path, "r+") as file:
        key_list = [key for key in file.keys() if np.issubdtype(file[key].dtype, np.number)]
        length = file[key_list[0]][:].shape[0]
        is_corrupt = np.zeros((length,), dtype=bool)
        for key in [key for key in file.keys() if np.issubdtype(file[key].dtype, np.number)]:
            is_corrupt |= np.isnan(file[key][:]) | np.isinf(file[key][:])
        size = sum(~is_corrupt)
        for key in file.keys():
            data = file[key][~is_corrupt]
            file[key].resize(size=(size,))
            file[key][...] = data

def sort_by_timestamp(file_path: Path):
    with h5py.File(file_path, "r+") as file:
        indices_order = file["Timestamp"][:].argsort()
        for key in file.keys():
            file[key][...] = file[key][indices_order]

def create_context_data(source_file_path: Path, dest_file_path: Path):
    with h5py.File(source_file_path, mode="r") as source_file, \
         h5py.File(destination_file_path, mode="w") as dest_file:
        key_list = list(source_file.keys())
        dest_file["event_key"] = key_list  # has length of ~300_000
        length = len(key_list)

        label_to_name = {0:'is_log', 1:'is_in40ms', 2:'is_in20ms', 3:'is_bd'}
        for name in label_to_name.values():
            dest_file.create_dataset(name=name, data=np.zeros(shape=(length,), dtype=bool), chunks=True)

        for key, index in zip(source_file.keys(), itertools.count(0)):
            label = source_file[key].attrs["Log Type"]
            dest_file[label_to_name[label]][index] = True

        dataset_settings_list = [('Timestamp', h5py.opaque_dtype(np.datetime64)),
                                 ('prev_td_Timestamp', h5py.opaque_dtype(np.datetime64))]
        for name, dt in dataset_settings_list:
            dest_file.create_dataset(name=name, shape=(length,), dtype=dt, chunks=True)

        data = to_datetime([source_file[key].attrs["Timestamp"].astype(str) for key in key_list]).to_numpy(dtype=np.datetime64)
        dest_file["Timestamp"][:] = data.astype(h5py.opaque_dtype(np.datetime64))

def add_context_data():
    dataset_settings_list = [('Timestamp',         h5py.opaque_dtype(np.datetime64)),
                             ('prev_td_Timestamp', h5py.opaque_dtype(np.datetime64))]

    def pulse_length(arr: npt.ArrayLike):
        acquisation_window: float = 2e-6
        number_of_signals = arr.size[0]
        is_high_val = arr > arr.max()/2
        # calculate longest continous high point
        # indices_low_vals = np.flatnonzero(~is_high_val)
        # dist_between_low_vals = np.diff(np.r_[-1, indices_low_vals, number_of_signals])
        # dist_between_low_vals.max() - 1
        return acquisation_window*is_high_val.sum()/number_of_signals

    #for key in ["PKI Amplitude", "PKI Phase"]:
    #    dest_file[f"pulse_length_{key}"] = pulse_length(source_file[key][:])


if __name__ == "__main__":
    source_file_path = Path("~/output_files/TrendDataExtLinks.hdf").expanduser()
    destination_file_path = Path("~/output_files/combined_td.hdf").expanduser()
    h5py.File(destination_file_path, mode="w").close()

    hdf_path_list = get_hdf_paths(file_path=source_file_path, depth=1)
    with h5py.File(source_file_path, "r") as source_file:
        data_set_list = source_file[hdf_path_list[0]].keys()

    for data_set_name in  data_set_list:
        print(data_set_name)
        union(source_file_path=source_file_path,
              hdf_path_list=hdf_path_list,
              data_set_name=data_set_name,
              destination_file_path=destination_file_path)
    LOG.debug("## Union done")
    clean(destination_file_path)
    LOG.debug("## Clean done")
    sort_by_timestamp(destination_file_path)
    LOG.debug("##sort_by_timestmap done")

    ed_links_file_path = Path("~/output_files/EventDataExtLinks.hdf").expanduser()
    destination_file_path= Path("~/output_files/context_data.hdf").expanduser()
    create_context_data(ed_links_file_path, destination_file_path)