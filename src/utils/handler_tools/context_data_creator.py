"""This module contains a class structure for creating a context data file. The ContextDataCreator class organizes the
creation of the context data file."""
import typing
import logging
from pathlib import Path
from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as Lock_dtype
import h5py
from src.utils.handler_tools.feature import EventDataFeature

logger = logging.getLogger(__name__)


def task(feature: EventDataFeature,
         ed_file_path: Path,
         dest_file_path: Path,
         write_lock: Lock_dtype) -> None:
    """
    One task is calculating one feature for the whole dataset.
    :param feature: the Feature object to calculate
    :param src_file_path: the file path of the source file
    :param dest_file_path: the file path of the destination file, where the features will be stored
    :param write_lock: the lock for writing into the destination file path
    """
    logger.debug("calculate feature %s for %s", feature.name, feature.dest_hdf_path)
    vec = feature.apply(ed_file_path)
    logger.debug("start writing feature %s for %s", feature.name, feature.dest_hdf_path)
    with write_lock:
        feature.write(dest_file_path, vec)


class ContextDataCreator:
    """operates the creation of the context data file (a file filled with calculated features for each group in the
     input file."""

    def __init__(self, ed_file_path: Path,
                 td_file_path: Path,
                 dest_file_path: Path,
                 get_features: typing.Callable[[], typing.Iterable[EventDataFeature]]):
        self.ed_file_path: Path = ed_file_path
        self.td_file_path: Path = td_file_path
        self.dest_file_path: Path = dest_file_path
        self.get_event_data_features: typing.Callable[[], typing.Iterable[EventDataFeature]] = get_features
        with h5py.File(self.ed_file_path, "r") as file:
            self.len: int = len(file)  # = number of keys

    def calc_features(self, num_threads: int = 4):
        """assigns the tasks (calculating each feature) toa bunch of threads.
        :param num_threads: the number of threads for multiprocessing"""
        write_lock = Lock()
        with ThreadPool(num_threads) as pool:
            partial_task = partial(task,
                                   ed_file_path=self.ed_file_path,
                                   dest_file_path=self.dest_file_path,
                                   write_lock=write_lock)
            pool.map(partial_task, self.get_event_data_features())
