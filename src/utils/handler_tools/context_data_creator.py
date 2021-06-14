"""This module contains a class structure for creating a context data file. There is a Feature class where the features
can be applied and written. Additionally there is a ContextDataCreator class that organizes the creation of the context
data file."""
import typing
import logging
from pathlib import Path
from functools import partial
import numpy as np
import numpy.typing as npt
import h5py
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as Lock_dtype
import features
from setup_logging import setup_logging
from src.utils.hdf_tools import hdf_path_combine

setup_logging()
log = logging.getLogger("test_handler")
log.setLevel("DEBUG")


class Feature:
    """A feature is a statistical property of a time series (min, max, mean, pulse_amplitude, pulse_length, etc).
    One object represents one Feature and can be applied on an hdf5 dataset via the apply function"""

    def __init__(self,
                 name: str,
                 func: typing.Callable[[Path, str], typing.Any],
                 output_dtype: type,
                 hdf_path: str = "/",
                 info: str = None):
        self.name: str = name
        self.func: typing.Callable = func
        self.dtype = output_dtype
        self.dataset_name: str = hdf_path  # hdf_grp can also be None
        if info is None:
            raise RuntimeWarning("""It is recommended to add an info text to describe the feature function.
                                    Maybe you can use func.__doc__?""")
        self.info_text = info

    def write(self, dest_file_path: Path, data: npt.ArrayLike) -> None:
        """creates an hdf-dataset and fill the data into it.
        If the group does not exist yet, it also creates an hdf-group of the channel name where the feature belongs to.
        :param dest_file_path: the file path of the destination file
        :param data: the length the hdf-dataset should have
        """
        with h5py.File(dest_file_path, "a") as file:
            grp = file.require_group(self.dataset_name)
            grp.create_dataset(name=self.name, data=data, dtype=self.dtype, chunks=True)
            grp[self.name].attrs.create(name="info", data=self.info_text)

    def apply(self, src_file_path: Path) -> npt.ArrayLike:
        """
        applies the function of the feature to every hdf-object / self.dataset_name. and returns a vector with the
        calculated features
        :param src_file_path: file path of the source file (hdf file)
        :return: numpy array of datatype self.dtype
        """
        with h5py.File(src_file_path, "r") as file:
            ret_vec = np.empty(shape=(file.__len__(),), dtype=self.dtype)
            for key, index in zip(file.keys(), range(1000)):
                ret_vec[index] = self.func(src_file_path, hdf_path_combine(key, self.dataset_name))
        return ret_vec


def task(feature: Feature, src_file_path: Path, dest_file_path: Path, write_lock: Lock_dtype) -> None:
    """
    One task is calculating one feature for the whole dataset.
    :param feature: the Feature object to calculate
    :param src_file_path: the file path of the source file
    :param dest_file_path: the file path of the destination file, where the features will be stored
    :param write_lock: the lock for writing into the destination file path
    """
    vec = feature.apply(src_file_path)
    log.info(f"finished calculations for {feature.name}")
    with write_lock:
        feature.write(dest_file_path, vec)


class ContextDataCreator:
    """operates the creation of the context data file (a file filled with calculated features for each group in the
     input file."""
    def __init__(self, src_file_path: Path,
                 dest_file_path: Path,
                 get_features: typing.Callable[[], typing.Iterable[Feature]]):
        self.src_file_path: Path = src_file_path
        self.dest_file_path: Path = dest_file_path
        self.get_features: typing.Callable[[], typing.Iterable[Feature]] = get_features
        with h5py.File(self.src_file_path, "r") as file:
            self.len: int = len(file)  # = number of keys

    def calc_features(self, num_threads: int = 4):
        """assigns the tasks (calculating each feature) toa bunch of threads.
        :param num_threads: the number of threads for multiprocessing"""
        write_lock = Lock()
        with ThreadPool(num_threads) as pool:
            partial_task = partial(task, src_file_path=self.src_file_path,
                                   dest_file_path=self.dest_file_path,
                                   write_lock=write_lock)
            pool.map(partial_task, self.get_features())


if __name__ == '__main__':
    destination_file_path = Path("~/output_files/context_data.hdf").expanduser()
    h5py.File(destination_file_path, "w").close()  # overwrite destination file
    source_file_path = Path("~/output_files/EventDataExtLinks.hdf").expanduser()
    cd_creator = ContextDataCreator(src_file_path=source_file_path,
                                    dest_file_path=destination_file_path,
                                    get_features=features.get_features)
    cd_creator.calc_features()
