"""This module contains three straight forward functions for hdf-files: unite, clean, sort_by. (
Unite: unites all hdf-datasets of the source file with the same name into one large dataset. The datasets are expected
to be separated in different groups.
Clean: The clean_by_row cleans "corrupt" values row by row after all datasets have been united.
Sort: sort_by sorts all datasets with respect to one of them"""
import logging
from pathlib import Path
import argparse
import numpy as np
import h5py
import coloredlogs
import pandas as pd
from src.utils.hdf_tools import get_all_dataset_items, get_all_dataset_values

logger = logging.getLogger(__name__)


def merge(source_file_path: Path, dest_file_path: Path) -> None:
    """
    In the first layer of the hdf-directory structure there have to be only groups. In each group is required to have
    the same hdf-datasets, they are referred to as dataset-type.
    for each dataset-type: unites (concatenates) all datasets of the same dataset-type from all groups.
    :param source_file_path: file of the un-united groups with same data set names
    :param dest_file_path: file where the united datasets will be stored
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


def _check_corruptness(arr):  # npt.ArrayLike[typing.Union[np.number, np.datetime64]]
    """checks if the input array is healthy of corrupt. In this case corrupt means infinite value or nan value.
    :param arr: input array
    :return: array with boolean values. True if the value in the input cell was healthy, False if it was corrupt."""
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
    :param file_path: the path of the hdf  file with the datasets. (already united with unite())
    :param sort_by_name: name of the dataset to be sorted
    """
    with h5py.File(file_path, "r+") as file:
        indices_order = file[sort_by_name][:].argsort()
        for channel in get_all_dataset_values(file):
            data = channel[:]
            channel[...] = data[indices_order]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
            Merges all channels(datasets) scattered on multiple groups into large
            channels(datasets) located at the root.
            [tdms syntax: root->group->channel, (hdf syntax: root->group->dataset)]""")
    parser.add_argument("source", type=Path,
                        help="file path of the source hdf file where the channels(datasets) are scattered on multiple "
                             "groups.")
    parser.add_argument("--dest", type=Path, default=Path("./combined.hdf"),
                        help="file path of the destination file where the merged channels(datasets) will be located.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print debug log messages")
    parser.add_argument("-c", "--clean", action="store_true", help="remove smelly values")
    parser.add_argument("--convert_datetime", action="store_true", help="convert iso8601 datetime strings to datetime"
                                                                        "format")
    parser.add_argument("--sort_by", type=str, action="store", default="Timestamp",
                        help="the channel(dataset) name the data will be sorted on.")
    args = parser.parse_args()
    coloredlogs.install(level="DEBUG" if args.verbose else "INFO")
    logger = logging.getLogger(__name__)

    logger.debug("start merge")
    merge(source_file_path=args.source.resolve(), dest_file_path=args.dest.resolve())
    if args.convert_datetime:
        logger.debug("start convert_iso8601_to_datetime")
        convert_iso8601_to_datetime(file_path=args.dest.resolve())
    if args.clean:
        logger.debug("start clean")
        clean_by_row(file_path=args.dest.resolve())
    logger.debug("start sort_by")
    sort_by(file_path=args.dest.resolve(), sort_by_name=args.sort_by)
