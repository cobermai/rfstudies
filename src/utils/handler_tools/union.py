import os.path
from pathlib import Path
from typing import Union
import logging
from datetime import datetime
import math
import itertools
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from src.utils.transf_tools.gather import hdf_path_combine
from setup_logging import setup_logging
setup_logging()
LOG = logging.getLogger("test_handler")
import numpy.typing as npt
from dateutil.parser import parse
from functools import partial
import time



def is_datetime(val):
    try:
        parse(val)
    except (TypeError, ValueError):
        ret = False
    else:
        ret = True
    return ret

def get_datetime_converter_for(val):
    convert = parse
    if isinstance(val, bytes):
        def cut_end(ts: bytes, end: bytes) -> bytes:
            return ts.replace(end, b"")
        for useless_timezone_ending in [b"Z", b"+00:00"]:
            if val.endswith(useless_timezone_ending):
                convert = partial(cut_end, end=useless_timezone_ending)
    return convert


def union(source_file_path: Path, dest_file_path: Path):
    """
    for each dataset_name it unites (concatenates) all hdf-dataset at given depth to one large hdf-dataset in the
    destination file
    :param source_file_path:
    :param destination_file_path:
    :param depth:
    :return:
    """
    with h5py.File(source_file_path, mode="r") as source_file, \
         h5py.File(dest_file_path, mode="a") as dest_file:
        first_grp = source_file.values().__iter__().__next__()
        for chn in first_grp.keys():
            example_val = first_grp[chn][0]
            if is_datetime(val=example_val):
                convert = get_datetime_converter_for(example_val)
                data = [convert(ts) for path in source_file.keys() for ts in source_file[path][chn][:]] # .astype(str)
                data = np.array(data, dtype=np.datetime64)
                dest_file.create_dataset(name=chn, data=data.astype(dtype=h5py.opaque_dtype(data.dtype)), chunks=True)
            else:
                data = np.concatenate([source_file[path][chn][:] for path in source_file.keys()])
                dest_file.create_dataset(name=chn, data=data, chunks=True)

def clean(file_path: Path):
    with h5py.File(file_path, "r+") as file:
        chn_list = [chn for chn in file.keys() if np.issubdtype(file[chn].dtype, np.number)]
        shape = file.values().__iter__().__next__().shape
        is_corrupt = np.zeros(shape, dtype=bool)
        for chn in chn_list:
            is_corrupt |= np.isnan(file[chn][:]) | np.isinf(file[chn][:])
        new_shape = (sum(~is_corrupt),)
        for key in file.keys():
            data = file[key][~is_corrupt]
            file[key].resize(size=new_shape)
            file[key][...] = data

def sort_by(file_path: Path, sort_by_name: str):
    with h5py.File(file_path, "r+") as file:
        indices_order = file[sort_by_name][:].argsort()
        for key in file.keys():
            file[key][...] = file[key][indices_order]
