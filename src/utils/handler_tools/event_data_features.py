"""This module contains the definition and gathering of EventDataFeatures for the XBox2 data set."""
from pathlib import Path
import typing
from functools import partial
import numpy as np
import numpy.typing as npt
import h5py
import tsfresh
import pandas as pd

from src.utils.hdf_tools import hdf_path_combine
from src.utils.handler_tools.customfeature import EventDataFeature


def get_event_data_features() -> typing.Generator:
    """This function generates all Features for the xb2 data set.
    :return: generator of features"""
    func_len = pulse_length
    func_amp = pulse_amplitude
    for chn in ['PEI Amplitude', 'PKI Amplitude', 'PSI Amplitude', 'PSR Amplitude']:
        yield EventDataFeature(name="pulse_length", func=func_len, output_dtype=float, hdf_path=chn,
                               info=func_len.__doc__)
        yield EventDataFeature(name="pulse_amplitude", func=func_amp, output_dtype=float, hdf_path=chn,
                               info=func_amp.__doc__)

    for chn in ['DC Up', 'DC Down']:
        yield EventDataFeature(name="min", func=apply_func_creator(np.min), output_dtype=float, hdf_path=chn,
                               info="calculates the minimum of the data")
        yield EventDataFeature(name="D1", func=apply_func_creator(partial(np.quantile, q=.1)),
                               output_dtype=float, hdf_path=chn, info="calculates the first deciles of the data")
        yield EventDataFeature(name="mean", func=apply_func_creator(np.mean), output_dtype=float, hdf_path=chn,
                               info="calculates the mean of the data")
        yield EventDataFeature(name="D9", func=apply_func_creator(partial(np.quantile, q=.9)),
                               output_dtype=float, hdf_path=chn, info="calculates the 9th deciles of the data")
        yield EventDataFeature(name="max", func=apply_func_creator(np.max), output_dtype=float, hdf_path=chn,
                               info="calculates the maximum of the data")


def pulse_length(data) -> float:
    """calculates the duration in sec where the amplitude is higher than the threshold (=half of the maximal value)."""
    acquisition_window: float = 2e-6
    threshold: float = data.max() / 2
    return (data > threshold).sum() / data.shape[0] * acquisition_window


def pulse_amplitude(data) -> float:
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value)."""

    threshold: float = data.max() / 2
    return data[data > threshold].mean()


def apply_func_creator(func: typing.Callable) -> typing.Callable[[Path, str], typing.Any]:
    """This function creates feature functions. It applies the input function "func" on the input dataset of the input
    data set of apply_func.
    :param func: the function to apply on the input data of the apply_func
    :return: a function that applies func """
    def apply_func(data) -> float:
        """calculates the median of the data"""
        return func(data)
    return apply_func

def ts_fresh_features(file_path: Path, hdf_path: str):
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value)."""
    with h5py.File(file_path, "r") as file:
        grp = file[hdf_path]
        df = pd.DataFrame(data={key: grp[key][:] for key in grp.keys() if len(grp[key][:])==3200 and "Amplitude" in key})
        df['column_sort'] = df.index
        df_molten = df.melt(id_vars='column_sort')
        settings = tsfresh.feature_extraction.settings.EfficientFCParameters()
        return tsfresh.extract_features(timeseries_container=df_molten,
                                        column_id="variable",
                                        column_sort="column_sort",
                                        column_value="value",
                                        default_fc_parameters=settings,n_jobs=0)


#df = ts_fresh_features(Path("~/output_files/data/EventData_20180401.hdf").expanduser(), "/Log_2018.04.01-23:39:54.227")
#print(df)
