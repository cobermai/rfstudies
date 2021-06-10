"""future data handler. UNFINISHED"""
import os.path
import logging
import typing
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
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as Lock_dtype
import functools
import feature_funcs

setup_logging()
log = logging.getLogger("test_handler")
log.setLevel("DEBUG")

class Feature:
    def __init__(self, name: str,  func: typing.Callable[[Path, str], typing.Any], output_dtype: type, hdf_path: str = "/"):
        self.name: str = name
        self.dataset_name: str = hdf_path  # hdf_grp can also be None
        self.func: typing.Callable = func
        self.dtype = output_dtype

    def initalize(self, file_path: Path, length: int):
        with h5py.File(file_path, "a") as file:
            grp = file.require_group(self.dataset_name)
            grp.create_dataset(name = self.name, shape=(length,), dtype=self.dtype, chunks=True)
            if not self.func.__doc__ is None:
                grp[self.name].attrs.create(name="funcion_doc", data=self.func.__doc__)
            else:
                raise RuntimeWarning(f"there is no function doc string for function {self.func.__name__} {self.name}. This is added as a feature description.")

    def apply(self, file_path: Path, hdf_path: str):

        path = os.path.join(hdf_path, self.dataset_name)
        log.debug("%s and %s-> %s",hdf_path, self.dataset_name, path)
        return self.func(file_path, path)

def task(grp_name: str, index: int, src_file_path: Path, dest_file_path: Path, features: typing.Iterable[Feature], write_lock: Lock_dtype):
    # CALC
    calculated_features = {}
    for feature in features:
        calculated_features[feature] = feature.apply(src_file_path, grp_name)
    # WRITE
    with write_lock:
        with h5py.File(dest_file_path, "r+") as dest_file:
            for feature, value in calculated_features.items():
                hdf_path = os.path.join(grp_name, feature.dataset_name)
                log.debug("hdf dest path: %s",hdf_path)
                dest_file[hdf_path][index] = value

class ContextDataCreator:
    def __init__(self, src_file_path: Path, dest_file_path: Path, features: typing.Iterable[Feature]):
        self.src_file_path: Path = src_file_path
        self.dest_file_path: Path = dest_file_path
        self.features: typing.Iterable[Feature] = features
        with h5py.File(self.src_file_path, "r") as file:
            self.len: int = len(file)  # = number of keys

    def init_features(self):
        h5py.File(self.dest_file_path, "w").close()
        for feature in self.features:
            feature.initalize(self.dest_file_path, self.len)

    def calc_features(self, num_iter):
        write_lock = Lock()
        #with h5py.File(self.src_file_path, "r") as file:
        #    event_keys = list(file.keys())
        with ThreadPool(1) as pool, h5py.File(self.src_file_path, "r") as file:
            partial_task = functools.partial(task, src_file_path=self.src_file_path,
                                             dest_file_path=self.dest_file_path,
                                             features=self.features,
                                             write_lock=write_lock)
            pool.starmap(partial_task, zip(file.keys(), range(num_iter)))  # itertools.count(0)


def create_context_data(source_file_path: Path, dest_file_path: Path):
    features: typing.Iterable[Feature] = []
    features += [Feature("pulse_length", feature_funcs.pulse_length,                 float,                     hdf_path=chn) for chn in ['PEI Amplitude', 'PKI Amplitude', 'PSI Amplitude', 'PSR Amplitude']]
    features += [Feature(is_type,        feature_funcs.log_type_translator(is_type), bool                                   ) for is_type in ["is_log", "is_bdin40ms", "is_bdin20ms", "is_bd"]]
    features += [Feature("timestamp",    feature_funcs.get_timestamp,                h5py.opaque_dtype('M8[us]')            )]
    cd_creator = ContextDataCreator(source_file_path, dest_file_path, features)
    t0 = time.time()
    cd_creator.init_features()
    t1 = time.time()
    num_iter=10
    cd_creator.calc_features(num_iter=num_iter)
    log.info("init took %ssek and clac took %ssek (so %s per iter)", t1 - t0, time.time() - t1, (time.time() - t1) / num_iter)


if __name__=='__main__':
    ed_links_file_path = Path("~/output_files/EventDataExtLinks.hdf").expanduser()
    destination_file_path = Path("~/output_files/context_data.hdf").expanduser()
    create_context_data(ed_links_file_path, destination_file_path)

"""
def create_context_data_old(source_file_path: Path, dest_file_path: Path):
    with h5py.File(source_file_path, mode="r") as source_file, \
         h5py.File(destination_file_path, mode="w") as dest_file:

        length = source_file.__len__()

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
"""