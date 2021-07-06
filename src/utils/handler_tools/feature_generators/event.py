"""This module contains the definition and gathering of EventDataFeatures for the XBox2 data set."""
import typing
from functools import partial
import numpy as np
from src.utils.handler_tools.feature_class import EventDataFeature


def get_event_data_features() -> typing.Generator:
    """This function generates all custom EventDataFeatures for the xbox2 data set.
    :return: generator of features"""
    func_len = pulse_length
    func_amp = pulse_amplitude
    for chn in ['PEI Amplitude', 'PKI Amplitude', 'PSI Amplitude', 'PSR Amplitude']:
        yield EventDataFeature(name="pulse_length", func=func_len, hdf_path=chn,
                               info=func_len.__doc__)
        yield EventDataFeature(name="pulse_amplitude", func=func_amp, hdf_path=chn,
                               info=func_amp.__doc__)

    for chn in ['DC Up', 'DC Down']:
        yield EventDataFeature(name="D1", func=apply_func_creator(partial(np.quantile, q=.1)),
                               hdf_path=chn, info="calculates the first deciles of the data")
        yield EventDataFeature(name="D9", func=apply_func_creator(partial(np.quantile, q=.9)),
                               hdf_path=chn, info="calculates the 9th deciles of the data")


def pulse_length(data) -> float:
    """calculates the duration in micro seconds where the amplitude is higher than the threshold
    (=half of the maximal value)."""
    acquisition_window = 2  # in micro seconds
    threshold = data.max() / 2
    num_total_values = data.shape[0]
    num_of_high_values = (data > threshold).sum()
    return acquisition_window * (num_of_high_values / num_total_values)


def pulse_amplitude(data) -> float:
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value)."""
    threshold = data.max() / 2
    return data[data > threshold].mean()


def apply_func_creator(func: typing.Callable) -> typing.Callable:
    """creates a feature-function that applies func to the input data of the feature-function.
    :param func: the function to apply on the input data of the apply_func
    :return: a function that applies func"""
    def apply_func(data) -> float:
        """applies simple function func on data"""
        return func(data)
    return apply_func
