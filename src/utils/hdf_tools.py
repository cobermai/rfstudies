"""
This module contains straight forward functions for hdf-files. To explain some of them in further detail:
merge: merges all hdf-datasets of the source file with the same name into one large dataset. The datasets are expected
to be separated in different groups.
clean: The clean_by_row cleans "corrupt" values row by row after all datasets have been merged.
sort: sort_by sorts all datasets with respect to one of them
conversion to df: convert_to_df converts an hdf file into a pandas dataframe for quick manipulation
"""
from pathlib import Path
import logging
import re
import typing
import numpy as np
import h5py
import pandas as pd

logger = logging.getLogger(__name__)


def merge(source_file_path: Path, dest_file_path: Path) -> None:
    """
    In the first layer of the hdf-directory structure there have to be only groups. In each group it is required to have
    the same hdf-datasets, they are referred to as dataset-type.
    For each dataset-type: merges (concatenates) all datasets of the same dataset-type from all groups.
    :param source_file_path: file of the un-merged groups with same data set names
    :param dest_file_path: file where the merged datasets will be stored
    """
    with h5py.File(source_file_path, mode="r") as source_file, \
            h5py.File(dest_file_path, mode="a") as dest_file:
        first_grp = source_file.values().__iter__().__next__()
        for channel_name in first_grp.keys():  # the channel names are always the same
            logger.debug("currently merging: %s", channel_name)
            data = np.concatenate([grp[channel_name][:] for grp in source_file.values()])
            dest_file.create_dataset(name=channel_name, data=data, chunks=True)


def convert_iso8601_to_datetime(file_path: Path, also_convert_attrs: bool = True) -> None:
    """converts datasets and attributes of strings of iso8601 format to numpy datetime format.
    :param file_path: Path of the hdf file to convert.
    :param also_convert_attrs: boolean value to define if attrs datetime should be converted too."""
    def convert_attrs(_: str, hdf_obj):
        """This visitor function (hdf.File.visititems()) converts all the attributes of the given hdf_obj."""
        for attrs_key, val in hdf_obj.attrs.items():
            try:
                val = pd.to_datetime(val.astype(str), format="%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                pass
            else:
                val = val.to_numpy(np.datetime64)
                del hdf_obj.attrs[attrs_key]
                hdf_obj.attrs.create(name=attrs_key, data=np.array(val).astype(h5py.opaque_dtype(val.dtype)))

    with h5py.File(file_path, mode="r+") as file:
        if also_convert_attrs:
            convert_attrs("/", file)
            file.visititems(convert_attrs)

        for key, channel in list(get_all_dataset_items(file)):
            try:
                data = pd.to_datetime(channel[:].astype(str), format="%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                pass
            else:
                data = data.to_numpy(np.datetime64)
                del file[key]
                file.create_dataset(name=key, data=data.astype(h5py.opaque_dtype(data.dtype)))


def _check_corruptness(arr: np.ndarray):
    """checks if the input array is healthy of corrupt. In this case corrupt means infinite value or nan value.
    :param arr: input array
    :return: array with boolean values. True if the value in the input cell was healthy, False if it was corrupt."""
    if np.issubdtype(arr.dtype, np.number):
        is_corrupt = np.isnan(arr) | np.isinf(arr)
    elif np.issubdtype(arr.dtype, np.datetime64):
        is_corrupt = np.isnat(arr)
    else:
        raise NotImplementedError("Corrupt data is only known for numeric and datetime values.")
    return is_corrupt


def clean_by_row(file_path: Path) -> None:
    """remove "rows" where the check_corruptness function returns at least one True value in the the row.
    A column is referred to as a dataset of the hdf file.
    A row is referred to as the values from all columns with the same index.
    :param file_path: the path of the hdf  file with the datasets. (already merged with merge())"""
    with h5py.File(file_path, "r+") as file:
        shape = file.values().__iter__().__next__().shape  # shape of the first dataset
        is_corrupt = np.zeros(shape, dtype=bool)
        for channel in file.values():
            is_corrupt |= _check_corruptness(channel[:])
        new_shape = (sum(~is_corrupt),)
        for channel in file.values():
            logger.debug("cleaning channel: %s", channel)
            data = channel[~is_corrupt]
            channel.resize(size=new_shape)
            channel[...] = data


def sort_by(file_path: Path, sort_by_name: str) -> None:
    """
    sorts all datasets with respect to one specific dataset (given by the key), done inplace.
    :param file_path: the path of the hdf  file with the datasets. (already merged with merge())
    :param sort_by_name: name of the dataset to be sorted
    """
    with h5py.File(file_path, "r+") as file:
        indices_order = file[sort_by_name][:].argsort()
        for channel in get_all_dataset_values(file):
            data = channel[:]
            channel[...] = data[indices_order]


def hdf_path_combine(*argv: str) -> str:
    """
    Concatenates hdf path with "/" in between. Works similar to Path(str, str, str) or the / operator for Path objects
    but for hdf paths (as strings)
    :param argv: the group names/to concatenate
    :return: the concatenated path string
    """
    path = "/" + "/".join(argv)
    path = re.sub('//+', '/', path)
    return path


def _get_all_dataset_items_rec(hdf_path: str,
                               hdf_obj: typing.Union[h5py.File, h5py.Dataset, h5py.Group]) -> typing.Generator:
    if isinstance(hdf_obj, h5py.Dataset):
        yield hdf_path, hdf_obj
    else:
        for key, val in hdf_obj.items():
            yield from ((hdf_path_combine(hdf_path, key), val) for key, val in _get_all_dataset_items_rec(key, val))


def get_all_dataset_items(hdf_obj, path: str = "/") -> typing.Generator:
    """a generator that returns all items that are children of the value hdf object
    :param hdf_obj: starting hdf_obj (file, group, dataset)
    :param path: the path of the starting hdf_obj
    """
    yield from _get_all_dataset_items_rec(path, hdf_obj)


def get_all_dataset_values(value: typing.Union[h5py.File, h5py.Dataset, h5py.Group]) -> typing.Generator:
    """a generator that returns all values that are children of the value hdf object
    :param value: the value to recursively go through all values
    """
    if isinstance(value, h5py.Dataset):
        yield value
    else:
        for val in value.values():
            yield from get_all_dataset_values(val)


def hdf_to_df(file_path: Path, hdf_path: str = "/"):
    """Converts hdf files with the write format into hdf files. This will be extended to further functionality."""
    with h5py.File(file_path, "r") as file:
        return pd.DataFrame(data={path[1:].replace("/", "__").replace(" ", "_"): val[:]
                                  for path, val in get_all_dataset_items(file[hdf_path], hdf_path)})


def hdf_to_df_selection(file_path: Path, selection, hdf_path: str = "/"):
    """Converts hdf files with the write format into hdf files. This will be extended to further functionality."""
    with h5py.File(file_path, "r") as file:
        return pd.DataFrame(data={path[1:].replace("/", "__").replace(" ", "_"): val[selection]
                                  for path, val in get_all_dataset_items(file[hdf_path], hdf_path)})


if __name__ == "__main__":
    df = hdf_to_df(Path("~/output_files/context_data.h5").expanduser())
    print(df)
