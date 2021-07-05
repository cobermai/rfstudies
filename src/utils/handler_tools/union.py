"""This module contains three straight forward functions for hdf-files: unite, clean, sort_by. (
Unite: unites all hdf-datasets of the source file with the same name into one large dataset. The datasets are expected
to be separated in different groups.
Clean: The clean_by_row cleans "corrupt" values row by row after all datasets have been united.
Sort: sort_by sorts all datasets with respect to one of them"""
import typing
from pathlib import Path
from dateutil.parser import parse
import numpy as np
from numpy import typing as npt
import h5py
from src.utils.hdf_tools import get_datasets


def is_datetime(val: typing.Any):
    """returns True if val can be parsed by dateutil.parser.parse() and False if not."""
    try:
        parse(val)
    except (TypeError, ValueError):
        ret = False
    else:
        ret = True
    return ret


def get_first_value(file: h5py.File):
    """returns the first hdf-object in the list of values. Instead of 'for x in file.values()[:1]:' this is:
    :param file: a file object of the h5py module
    :return: the first hdf object of the value iterable"""
    return file.values().__iter__().__next__()


def get_datetime_array_converter_for(example_data: typing.Union[str, bytes]) \
        -> typing.Callable:
    """Creates the fastest date-time converter known for other date-times given in the the same format.
    :param example_data: an example value of datetime format.
    """
    zulu_time_ending = b"Z"  # timezone +00:00
    if isinstance(example_data, bytes) and example_data.endswith(zulu_time_ending):
        def convert(data) -> npt.ArrayLike[np.datetime64]:
            return np.array([value.replace(zulu_time_ending, b"") for value in data], np.datetime64)
    else:
        def convert(data) -> npt.ArrayLike[np.datetime64]:
            return np.array([parse(value) for value in data], np.datetime64)
    return convert


def union(source_file_path: Path, dest_file_path: Path) -> None:
    """
    In the first layer of the hdf-directory structure there have to be only groups. In each group is required to have
    the same hdf-datasets, they are referred to as dataset-type.
    for each dataset-type: unites (concatenates) all datasets of the same dataset-type from all groups.
    :param source_file_path: file of the un-united groups with same data set names
    :param dest_file_path: file where the united datasets will be stored
    """
    with h5py.File(source_file_path, mode="r") as source_file, \
            h5py.File(dest_file_path, mode="a") as dest_file:
        first_grp = get_first_value(source_file)
        for chn in first_grp.keys():  # the channel names are always the same
            example_val = first_grp[chn][0]
            if is_datetime(val=example_val):
                convert = get_datetime_array_converter_for(example_val)
                data = convert([ts for path in source_file.keys() for ts in source_file[path][chn][:]])
                dest_file.create_dataset(name=chn, data=data.astype(dtype=h5py.opaque_dtype(data.dtype)), chunks=True)
            else:
                data = np.concatenate([source_file[path][chn][:] for path in source_file.keys()])
                dest_file.create_dataset(name=chn, data=data, chunks=True)


def check_corruptness(arr) -> npt.ArrayLike[bool]:  # npt.ArrayLike[typing.Union[np.number, np.datetime64]]
    """checks if the input array is healthy of corrupt. In this case corrupt means infinite value or nan value.
    :param arr: input array
    :return: ndarray with boolean values. True if the value in the input cell was healthy, False if it was corrupt."""
    if np.issubdtype(arr.dtype, np.number):
        ret = np.isnan(arr) | np.isinf(arr)
    elif np.issubdtype(arr.dtype, np.datetime64):
        ret = np.isnat(arr)
    else:
        raise NotImplementedError("Corrupt data is only known for numeric and datetime values.")
    return ret


def clean_by_row(file_path: Path) -> None:
    """remove "rows" where the check_corruptness function returns at least one True value in the the row.
    A column is referred to as a dataset of the hdf file.
    A row is referred to as the values from all columns with the same index.
    :param file_path: the path of the hdf  file with the datasets. (already united with unite())"""
    with h5py.File(file_path, "r+") as file:
        shape = get_first_value(file).shape  # shape of the first dataset
        is_corrupt = np.zeros(shape, dtype=bool)
        for chn in file.keys():
            is_corrupt |= check_corruptness(file[chn][:])
        new_shape = (sum(~is_corrupt),)
        for key in file.keys():
            data = file[key][~is_corrupt]
            file[key].resize(size=new_shape)
            file[key][...] = data


def sort_by(file_path: Path, sort_by_name: str) -> None:
    """
    sorts all datasets with respect to one specific dataset (sort_by_name), inplace.
    :param file_path: the path of the hdf  file with the datasets. (already united with unite())
    :param sort_by_name: name of the dataset to be sorted
    """
    with h5py.File(file_path, "r+") as file:
        indices_order = file[sort_by_name][:].argsort()
        for key in get_datasets(file_path):
            file[key][...] = file[key][indices_order]
