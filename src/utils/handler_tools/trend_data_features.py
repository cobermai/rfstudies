"""This module contains the definition and gathering of TrendDataFatures for the XBox2 data set."""
from pathlib import Path
import typing
from functools import partial
import numpy as np
import numpy.typing as npt
import h5py
import tsfresh
import pandas as pd
from src.utils.handler_tools.customfeature import TrendDataFeature


def get_trend_data_features(length: int, trend_data_file_path: Path) -> typing.Generator:
    with h5py.File(trend_data_file_path, "r") as file:
        for key in file.keys():
            yield TrendDataFeature(name=key,
                                   func=select(trend_data_file_path, key),
                                   output_dtype=h5py.opaque_dtype("M8[us]") if key=="Timestamp" else float,
                                   length=length,
                                   hdf_path="PrevTrendData",
                                   info=f"Previous Trend Data of {key}")


def select(trend_data_file_path: Path, key: str):
    def selection(selection):
        """returns a selection of one signal of trend data."""
        with h5py.File(trend_data_file_path, "r") as file:
            return np.array(file[key])[selection]
    return selection