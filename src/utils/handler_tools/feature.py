"""This module contains a class class that defines machine learning features. The Feature class contains the functions
apply and write."""
import typing
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
import h5py
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
                 working_dataset: str = "/",
                 info: str = None):
        self.name: str = name
        self.func: typing.Callable = func
        self.dtype = output_dtype
        self.working_dataset: str = working_dataset  # hdf_grp can also be None
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
            grp = file.require_group(self.working_dataset)
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
            for key, index in zip(file.keys(), range(300)):
                ret_vec[index] = self.func(src_file_path, hdf_path_combine(key, self.working_dataset))
        return ret_vec
