"""This module contains helping functions for handling hdf-files."""
from pathlib import Path
import re
import typing
import h5py
import pandas as pd


def hdf_path_combine(*argv: str) -> str:
    """
    Concatenates hdf path with "/" in between. Works similar to Path(str, str, str) or the / operator for Path objects
    but for hdf paths (as strings)
    :param argv: the group names/to concatenate
    :return: the concatenated path string
    """
    path = "/".join(argv)
    path = re.sub('//+', '/', path)
    return path


def get_datasets(file_path: Path, hdf_path: str = "/", mode="r+") -> typing.Generator:
    """a generator that returns all dataset names of an hdf-file with hdf-path root hdf_path
    :param file_path: the path of the hdf  file with the datasets.
    :param hdf_path: root path where searching for datasets should be started
    :param mode: the mode with which the hdf file should be read. This matters because of how hdf files work internally
    """
    with h5py.File(file_path, mode=mode) as file:
        if isinstance(file[hdf_path], h5py.Dataset):
            yield hdf_path
        else:
            for key in file[hdf_path].keys():
                yield from get_datasets(file_path, hdf_path=hdf_path_combine(hdf_path, key), mode=mode)


def hdf_to_df(file_path: Path, hdf_path: str = "/"):
    """Converts hdf files with the write format into hdf files. This will be extended to further functionality."""
    with h5py.File(file_path, "r") as file:
        return pd.DataFrame(data=(file[path][:100] for path in get_datasets(file_path, hdf_path=hdf_path, mode="r")),
                            index=get_datasets(file_path, hdf_path=hdf_path, mode="r")).T


if __name__ == "__main__":
    print(hdf_to_df(Path("~/output_files/context_data.hdf").expanduser()))
