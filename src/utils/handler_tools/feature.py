"""This module contains a class class that defines machine learning features. The Feature class contains the functions
apply and write."""
import typing
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
import h5py
from src.utils.hdf_tools import hdf_path_combine

log = logging.getLogger(__name__)


class Feature:
    """A feature is a statistical property of a time series (min, max, mean, pulse_amplitude, pulse_length, etc).
    One object represents one Feature and can be applied on an hdf5 dataset via the apply function and written via write
    This is the base class"""
    def __init__(self, name: str,
                 func: typing.Callable,
                 dest_hdf_path: str = "/",
                 info: str = None):
        self.name = name
        self.func = func
        self.dest_hdf_path = dest_hdf_path
        self.info = info

    def write(self, dest_file_path: Path, data):
        """
        creates an hdf-dataset at the self.dest_hdf_path and writes calculated feature data into it.
        If the destination hdf-path does not exist yet, it creates it. If the input data is of np.datetime64 format it
        will be stored as opaque dtype (https://docs.h5py.org/en/stable/special.html#storing-other-types-as-opaque-data)
        :param dest_file_path: the file path of the destination file
        :param data: the length the hdf-dataset should have
        """
        with h5py.File(dest_file_path, "a") as file:
            grp = file.require_group(self.dest_hdf_path)
            if np.issubdtype(data.dtype, np.datetime64):
                grp.create_dataset(name=self.name, data=data, dtype=h5py.opaque_dtype(data.dtype), chunks=True)
            else:
                grp.create_dataset(name=self.name, data=data, dtype=data.dtype, chunks=True)
            grp[self.name].attrs.create(name="info", data=self.info)


class EventDataFeature(Feature):
    """calculates features from the event data."""

    def __init__(self, name: str,
                 func: typing.Callable[[Path, str], typing.Any],
                 info: str = None,
                 working_on_signal: str = "/"):
        if info is None:
            raise RuntimeWarning("It is recommended to add an info text to describe the feature function."
                                 "Maybe you can use func.__doc__?")
        super().__init__(name=name, func=func, dest_hdf_path=working_on_signal, info=info)
        self.working_on_signal = working_on_signal

    def apply(self, ed_file_path: Path) -> npt.ArrayLike:
        """applies the function of the feature to every hdf-object / self.dataset_name. and returns a vector with the
        calculated features
        :param ed_file_path: file path of the source file (hdf file)
        :return: numpy array of datatype self.dtype
        """
        with h5py.File(ed_file_path, "r") as file:
            example_key = file.keys().__iter__().__next__()
            example_val = self.func(ed_file_path, hdf_path_combine(example_key, self.working_on_signal))
            dtype = type(example_val)
            ret_vec = np.empty(shape=(file.__len__(),), dtype=dtype)
            for key, index in zip(file.keys(), range(100)):
                ret_vec[index] = self.func(ed_file_path, hdf_path_combine(key, self.working_on_signal))
        return ret_vec


class CustomFeature(Feature):
    """This feature class creates features out of trend data and existing context_data calculated by the
    EventDataFeatures."""

    def __init__(self, name: str,
                 func: typing.Callable,
                 info: str = None,
                 dest_hdf_path: str = "/"):
        if info is None:
            raise RuntimeWarning("It is recommended to add an info text to describe the feature function."
                                 "Maybe you can use func.__doc__?")
        super().__init__(name=name, func=func, dest_hdf_path=dest_hdf_path, info=info)

    def apply(self, td_file_path) -> npt.ArrayLike:
        """
        applies the feature function on the trend data and the pre calculated context_data
        :param td_file_path: file path of the source file (an hdf file)
        :return: numpy array of datatype self.dtype
        """
        return self.func(td_file_path)
