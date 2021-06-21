"""This module contains the definition and gathering of Features for the XBox2 data set."""
from pathlib import Path
import typing
from functools import partial
import numpy as np
import numpy.typing as npt
import h5py
import tsfresh
import pandas as pd
from src.utils.hdf_tools import hdf_path_combine
from src.utils.handler_tools.feature import EventDataFeature


def get_event_data_features() -> typing.Generator:
    """This function generates all Features for the xb2 data set.
    :return: generator of features"""
    for is_type in ["is_log", "is_bd_in_40ms", "is_bd_in_20ms", "is_bd"]:
        func = log_type_translator(is_type)
        yield EventDataFeature(name=is_type, func=func, info=func.__doc__)

    func_len = pulse_length
    func_amp = pulse_amplitude
    for chn in ['PEI Amplitude', 'PKI Amplitude', 'PSI Amplitude', 'PSR Amplitude']:
        yield EventDataFeature(name="pulse_length", func=func_len, working_on_signal=chn,
                               info=func_len.__doc__)
        yield EventDataFeature(name="pulse_amplitude", func=func_amp, working_on_signal=chn,
                               info=func_amp.__doc__)

    for chn in ['DC Up', 'DC Down']:
        yield EventDataFeature(name="min", func=apply_func_creator(np.min), working_on_signal=chn,
                               info="calculates the minimum of the data")
        yield EventDataFeature(name="D1", func=apply_func_creator(partial(np.quantile, q=.1)),
                               working_on_signal=chn, info="calculates the first deciles of the data")
        yield EventDataFeature(name="mean", func=apply_func_creator(np.mean), working_on_signal=chn,
                               info="calculates the mean of the data")
        yield EventDataFeature(name="D9", func=apply_func_creator(partial(np.quantile, q=.9)),
                               working_on_signal=chn, info="calculates the 9th deciles of the data")
        yield EventDataFeature(name="max", func=apply_func_creator(np.max), working_on_signal=chn,
                               info="calculates the maximum of the data")


def log_type_translator(is_type: str) -> typing.Callable[[Path, str], bool]:
    """function to create functions that return True if the input value matches the translation of the is_type label and
    False in the other cases.
    :param is_type: string of the type that were interested in(in {"is_log", "is_bd_in_40ms", "is_bd_in_20ms", "is_bd"})
    """
    log_type_dict = {"is_log": 0, "is_bd_in_40ms": 1, "is_bd_in_20ms": 2, "is_bd": 3}

    def test_is_type(file_path: Path, hdf_path: str) -> bool:
        """
        This function translates the 'Log Type' group properties of the event data into a boolean value.
        :param file_path: file path of the data source
        :param hdf_path: hdf-path of the source group
        :return: True if (is_log -> 0, is_bd_in40ms -> 1, is_bd_in20ms -> 2, is_bd -> 3) in other cases return False
        """
        with h5py.File(file_path, "r") as file:
            label = file[hdf_path].attrs["Log Type"]
            if label in log_type_dict.values():
                ret = label == log_type_dict[is_type]
            else:
                raise ValueError(f"'Log Type' label not valid no translation for {label} in {log_type_dict}!")
        return ret

    return test_is_type


def get_timestamp(file_path: Path, hdf_path: str) -> np.datetime64:
    """
    returns the Timestamp from group properties/attribute in numpy datetime format
    :param file_path: file path of the data source
    :param hdf_path: hdf-path of the source group
    :return: numpy datetime format of the timestamp
    """
    with h5py.File(file_path, "r") as file:
        datetime_str = file[hdf_path].attrs["Timestamp"][:-1]
        return np.datetime64(datetime_str).astype(h5py.opaque_dtype('M8[us]'))


def pulse_length(file_path: Path, hdf_path: str):
    """calculates the duration in sec where the amplitude is higher than the threshold (=half of the maximal value)."""
    acquisition_window: float = 2e-6
    with h5py.File(file_path, "r") as file:
        data = file[hdf_path][:]
        threshold: float = data.max() / 2
        return (data > threshold).sum() / data.shape[0] * acquisition_window


def pulse_amplitude(file_path: Path, hdf_path: str) -> float:
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value)."""
    with h5py.File(file_path, "r") as file:
        data = file[hdf_path][:]
        threshold: float = data.max() / 2
        return data[data > threshold].mean()


def apply_func_creator(func: typing.Callable) -> typing.Callable[[Path, str], typing.Any]:
    """This function creates feature functions. It applies the input function "func" on the input dataset of the input
    data set of apply_func.
    :param func: the function to apply on the input data of the apply_func
    :return: a function that applies func """
    def apply_func(file_path: Path, hdf_path: str) -> float:
        """calculates the median of the data"""
        with h5py.File(file_path, "r") as file:
            data: npt.ArrayLike = file[hdf_path][:]
            return func(data)
    return apply_func

def ts_fresh(file_path: Path, hdf_path: str):
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value)."""
    with h5py.File(file_path, "r") as file:
        grp = file[hdf_path]
        example_key = grp.keys().__iter__().__next__()
        ts = grp[grp[example_key].attrs.get("Timestamp")]
        acquisition_window = 2e-6
        ts_df = ts + acquisition_window/3200
        df = pd.DataFrame(data={key: grp[key][:] for key in grp.keys() if len(grp[key][:])==3200})
        return df  # df.melt(value_vars=df.columns)


df = ts_fresh(Path("~/output_files/data/EventData_20180401.hdf").expanduser(), "/Log_2018.04.01-23:39:54.227")
print(df)
