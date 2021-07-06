"""This module contains three straight forward functions for hdf-files: unite, clean, sort_by. (
Unite: unites all hdf-datasets of the source file with the same name into one large dataset. The datasets are expected
to be separated in different groups.
Clean: The clean_by_row cleans "corrupt" values row by row after all datasets have been united.
Sort: sort_by sorts all datasets with respect to one of them"""
import typing
import logging
from pathlib import Path
import argparse
import dateutil.parser
import numpy as np
import h5py
import coloredlogs
from src.utils.hdf_tools import get_datasets

LOG = logging.getLogger(__name__)


def is_datetime(val: typing.Any):
    """returns True if val can be parsed by dateutil.parser.parse() and False if not."""
    try:
        dateutil.parser.isoparse(val)
    except (TypeError, ValueError):
        ret = False
    else:
        ret = True
    return ret


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
        for channel_name, first_channel in first_grp.items():  # the channel names are always the same
            logger.debug("currently merging: %s", channel_name)
            example_val = first_channel[0]
            if is_datetime(val=example_val):
                data = np.array([dateutil.parser.isoparse(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")
                                 for grp in source_file.values() for ts in grp[channel_name][:]],
                                dtype=np.datetime64)
                dest_file.create_dataset(name=channel_name,
                                         data=data.astype(dtype=h5py.opaque_dtype(data.dtype)),
                                         chunks=True)
            else:
                data = np.concatenate([grp[channel_name][:] for grp in source_file.values()])
                dest_file.create_dataset(name=channel_name,
                                         data=data,
                                         chunks=True)


def check_corruptness(arr):  # npt.ArrayLike[typing.Union[np.number, np.datetime64]]
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
            is_corrupt |= check_corruptness(channel[:])
        new_shape = (sum(~is_corrupt),)
        for channel in file.values():
            logger.debug("cleaning channel: %s", channel)
            data = channel[~is_corrupt]
            channel.resize(size=new_shape)
            channel[...] = data


def sort_by(file_path: Path, sort_by_name: str) -> None:
    """
    sorts all datasets with respect to one specific dataset (sort_by_name), inplace.
    :param file_path: the path of the hdf  file with the datasets. (already united with unite())
    :param sort_by_name: name of the dataset to be sorted
    """
    with h5py.File(file_path, "r+") as file:
        indices_order = file[sort_by_name][:].argsort()
        for channel_name in get_datasets(file_path):
            file[channel_name][...] = file[channel_name][indices_order]


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
    parser.add_argument("--sort_by", type=str, action="store", default="Timestamp",
                        help="the channel(dataset) name the data will be sorted on.")
    args = parser.parse_args()
    coloredlogs.install(level="DEBUG" if args.verbose else "INFO")
    logger = logging.getLogger(__name__)

    logger.debug("start merge")
    merge(source_file_path=args.source.resolve(), dest_file_path=args.dest.resolve())
    if args.clean:
        logger.debug("start clean")
        clean_by_row(file_path=args.dest.resolve())
    logger.debug("start sort_by")
    sort_by(file_path=args.dest.resolve(), sort_by_name=args.sort_by)
