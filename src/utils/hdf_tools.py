"""toolbox for quick read of hdf files to pandas dataframes and manipulation of hdf paths."""
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
    path = "/" + "/".join(argv)
    path = re.sub('//+', '/', path)
    return path


def get_all_dataset_items(file) -> typing.Generator:
    """a generator that returns all items that are children of the value hdf object
    :param value: the value to recursively go through all values
    """
    def return_items(path, val):
        if isinstance(val, h5py.Dataset):
            yield path, val
        else:
            for key, val in val.items():
                yield from ((hdf_path_combine(path, key), val) for key, val in return_items(key, val))
    yield from return_items("/", file)

def get_all_dataset_values(value) -> typing.Generator:
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
                                  for path, val in get_all_dataset_items(file)})


def hdf_to_df_selection(file_path: Path, selection, hdf_path: str = "/"):
    """Converts hdf files with the write format into hdf files. This will be extended to further functionality."""
    with h5py.File(file_path, "r") as file:
        return pd.DataFrame(data={path[1:].replace("/", "__").replace(" ", "_"): val[selection]
                                  for path, val in get_all_dataset_items(file)})


if __name__ == "__main__":
    df = hdf_to_df(Path("~/output_files/context_data.h5").expanduser())
    print(df)
