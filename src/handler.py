"""future data handler. UNFINISHED"""
import os.path
import logging
from datetime import datetime
import math
import itertools
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from src.utils.transf_tools.gather import hdf_path_combine
from setup_logging import setup_logging
import numpy.typing as npt
import dateutil.parser
import time
from utils.handler_tools.union import union, clean, sort_by

setup_logging()
LOG = logging.getLogger("test_handler")


def create_context_data(source_file_path: Path, dest_file_path: Path):
    with h5py.File(source_file_path, mode="r") as source_file, \
         h5py.File(destination_file_path, mode="w") as dest_file:

        key_list = list(source_file.keys())
        dest_file.create_dataset(name="event_key", data =key_list, chunks=True)  # has length of ~300_000
        length = len(key_list)
        del key_list

        for chn in ['BLM', 'BLM TIA', 'Col.', 'DC Down', 'DC Up', 'PEI Amplitude', 'PEI Phase', 'PER log',
                    'PKI Amplitude', 'PKI Phase', 'PKR log', 'PSI Amplitude', 'PSI Phase', 'PSR Amplitude',
                    'PSR Phase', 'PSR log']:
            dest_file.create_group(name=chn)

        ds_timestamp = dest_file.create_dataset("Timestamp", shape=(length,), dtype=h5py.opaque_dtype("<M8[us]"), chunks=True)

        label_name_dict = {0:'is_log', 1:'is_in40ms', 2:'is_in20ms', 3:'is_bd'}
        for name in label_name_dict.values():
            dest_file.create_dataset(name=name, data=np.zeros(shape=(length,), dtype=bool), chunks=True)


        def pulse_length(df: pd.DataFrame):
            threshold = df.max(axis=0) / 2
            acquisation_window: float = 2e-6
            return (df > threshold).sum() / df.shape[0] * acquisation_window

        for index, key in zip(range(10), source_file.keys()):  # zip(itertools.count(0), source_file.keys()):
            if math.fmod(index, 1000) == 0: print(index)
            ret = {}
            grp = source_file[key]
            ret["Timestamp"] = grp.attrs["Timestamp"][:-1]
            label = source_file[key].attrs["Log Type"]
            ret[label_name_dict[label]] = True
            df500 = pd.DataFrame([grp[key][:] for key in grp.keys() if grp[key].shape==(500,)], index=[key for key in grp.keys() if grp[key].shape==(500,)]).T
            df3200 = pd.DataFrame([grp[key][:] for key in grp.keys() if grp[key].shape==(3200,)], index=[key for key in grp.keys() if grp[key].shape==(3200,)]).T
            amplitude_channels = ['PEI Amplitude', 'PKI Amplitude', 'PSI Amplitude', 'PSR Amplitude']
            ret.update({chn: {"pulse_length": pls_len} for chn, pls_len in zip(amplitude_channels, pulse_length(df3200[amplitude_channels])) })
            print(ret)

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
    dest_file_path = Path("~/output_files/combined_td.hdf").expanduser()

    if True:
        h5py.File(dest_file_path, mode="w").close()
        union(source_file_path=source_file_path, dest_file_path=dest_file_path)
        clean(file_path=dest_file_path)
        sort_by(file_path=dest_file_path, sort_by_name="Timestamp")
    else:
        ed_links_file_path = Path("~/output_files/EventDataExtLinks.hdf").expanduser()
        destination_file_path= Path("~/output_files/context_data.hdf").expanduser()
        df = create_context_data(ed_links_file_path, destination_file_path)