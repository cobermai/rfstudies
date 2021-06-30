"""This module contains the definition and gathering of EvenAttributeFatures for the XBox2 data set."""
from pathlib import Path
import typing
from functools import partial
import numpy as np
import h5py
import tsfresh
import pandas as pd
from src.utils.handler_tools.customfeature import EventAttributeFeature


def get_event_attribute_features(length: int) -> typing.Generator:
    """This function generates all Features for the xb2 data set.
    :return: generator of features"""
    yield EventAttributeFeature(name="Timestamp", func=get_timestamp, length=length, hdf_path="/", output_dtype=h5py.opaque_dtype('M8[us]'),
                                info=get_timestamp.__doc__)
    for is_type in ["is_log", "is_bd_in_40ms", "is_bd_in_20ms", "is_bd"]:
        func = log_type_translator(is_type)
        yield EventAttributeFeature(name=is_type, func=func, length=length, hdf_path="/", output_dtype=bool, info=func.__doc__)


def log_type_translator(is_type: str) -> typing.Callable:
    """function to create functions that return True if the input value matches the translation of the is_type label and
    False in the other cases.
    :param is_type: string of the type that were interested in(in {"is_log", "is_bd_in_40ms", "is_bd_in_20ms", "is_bd"})
    """
    log_type_dict = {"is_log": 0, "is_bd_in_40ms": 1, "is_bd_in_20ms": 2, "is_bd": 3}

    def test_is_type(attrs) -> bool:
        """
        This function translates the 'Log Type' group properties of the event data into a boolean value.
        :param file_path: file path of the data source
        :param hdf_path: hdf-path of the source group
        :return: True if (is_log -> 0, is_bd_in40ms -> 1, is_bd_in20ms -> 2, is_bd -> 3) in other cases return False
        """
        label = attrs["Log Type"]
        if label in log_type_dict.values():
            ret = label == log_type_dict[is_type]
        else:
            raise ValueError(f"'Log Type' label not valid no translation for {label} in {log_type_dict}!")
        return ret

    return test_is_type


def get_timestamp(attrs) -> np.datetime64:
    """
    returns the Timestamp from group properties/attribute in numpy datetime format
    :param file_path: file path of the data source
    :param hdf_path: hdf-path of the source group
    :return: numpy datetime format of the timestamp
    """
    datetime_str = attrs["Timestamp"][:-1]
    return np.datetime64(datetime_str).astype(h5py.opaque_dtype('M8[us]'))

