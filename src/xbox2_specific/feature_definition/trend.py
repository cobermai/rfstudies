"""This module contains the definition and gathering of TrendDataFeatures for the XBox2 data set."""
from pathlib import Path
import typing
import h5py
import numpy as np
from src.utils.handler_tools.feature_class import TrendDataFeature


def get_trend_data_features(length: int, trend_data_file_path: Path) -> typing.Generator:
    """This function generates all TrendDataFeatures for the xbox2 data set.
    :param length: number of values that will be calculated by each feature.
    :param trend_data_file_path: file path of the trend_data_file
    :return: generator of features"""
    with h5py.File(trend_data_file_path, "r") as file:
        for key in file.keys():
            yield TrendDataFeature(name=key,
                                   func=_select(trend_data_file_path, key),
                                   output_dtype=h5py.opaque_dtype("M8[us]") if key == "Timestamp" else float,
                                   length=length,
                                   hdf_path="PrevTrendData",
                                   info=f"Previous Trend Data of {key}")


def _select(trend_data_file_path: Path, key: str):
    """creates and returns a function that selects a given selection from the dataset at the predefined key."""
    def selection(indices_selection: np.ndarray):
        """returns a selection of one signal of the trend data.
        :param indices_selection: selection of indices to return (array of boolean values)"""
        with h5py.File(trend_data_file_path, "r") as file:
            return np.array(file[key])[indices_selection]
    return selection
