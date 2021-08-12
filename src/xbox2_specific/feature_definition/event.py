"""This module contains the definition and gathering of EventDataFeatures for the XBox2 data set."""
import typing
from functools import partial
import numpy as np
from src.utils.handler_tools.feature_class import EventDataFeature


def get_event_data_features(length) -> typing.Generator:
    """This function generates all custom EventDataFeatures for the xbox2 data set.
    :return: generator of features"""
    for chn in ['PEI Amplitude', 'PKI Amplitude', 'PSI Amplitude', 'PSR Amplitude']:
        yield EventDataFeature(name="pulse_length",
                               func=_pulse_length,
                               length=length,
                               output_dtype=float,
                               hdf_path=chn,
                               working_on_dataset=chn,
                               info="The pulse length is the pulse duration in mirco seconds. The pulse is defined as "
                                    "the region where the amplitude is higher than the threshold "
                                    "(=half of maximal value)")
        yield EventDataFeature(name="pulse_amplitude",
                               func=_pulse_amplitude,
                               working_on_dataset=chn,
                               length=length,
                               output_dtype=float,
                               hdf_path=chn,
                               info="The Pulse Amplitude is the mean value of the pulse. The pulse is defined as the "
                                    "region where the amplitude is higher than the threshold (=half of maximal value)")

    for chn in ['DC Up', 'DC Down']:
        yield EventDataFeature(name="D1",
                               func=_apply_func_creator(partial(np.quantile, q=.1)),
                               working_on_dataset=chn,
                               length=length,
                               output_dtype=float,
                               hdf_path=chn,
                               info="calculates the first deciles of the data")
        yield EventDataFeature(name="D9",
                               func=_apply_func_creator(partial(np.quantile, q=.9)),
                               working_on_dataset=chn,
                               length=length,
                               output_dtype=float,
                               hdf_path=chn,
                               info="calculates the 9th deciles of the data")

    yield EventDataFeature(name="dc_up_threshold_reached",
                           func=_dc_up_threshold_func,
                           working_on_dataset="DC Up",
                           length=length,
                           output_dtype=bool,
                           hdf_path="/",
                           info="Decides if event is a breakdown with a threshold of -0.01 on the DC Up signal."
                                "So if the min of DC Up is < (threshold assigned by experts) it is labeled as a "
                                "breakdown.")


def _get_data_above_half_max(data: np.ndarray):
    """returns data that has a value higher than half of the maximal value."""
    threshold = data.max(initial=-np.inf)/2
    return data[data > threshold]


def _dc_up_threshold_func(data: np.ndarray) -> bool:
    """checks if any of the signals is below the threshold.
    :param data: a vector of values of the group working_on_dataset (see EventDataFeature.working_on_dataset)"""
    threshold = -0.05  # Threshold defined by RF Cavity Experts
    return bool(np.any(data < threshold))


def _pulse_length(data: np.ndarray) -> float:
    """calculates the pulse duration in micro seconds where the amplitude is higher than the threshold
    (=half of the maximal value).
    :param data: a vector of values of the group working_on_dataset
    (see :func:`~src.utils.handler_tools.feature_class.EventDataFeature.working_on_dataset` )
    """
    acquisition_window = 2  # in micro seconds
    num_total_values = len(data)
    num_relatively_large_values = len(_get_data_above_half_max(data))
    if num_total_values == 0:
        pulse_length = 0.
    else:
        pulse_length = acquisition_window * (num_relatively_large_values / num_total_values)
    return pulse_length

def _pulse_amplitude(data: np.ndarray) -> float:
    """calculates the mean value where the amplitude is higher than the threshold (=half of the maximal value).
    :param data: a vector of values of the group working_on_dataset (see EventDataFeature.working_on_dataset)
    """
    return _get_data_above_half_max(data).mean()


def _apply_func_creator(func: typing.Callable) -> typing.Callable:
    """creates a feature-function that applies func to the input data of the feature-function.
    :param func: the function to apply on the input data of the apply_func
    :return: a function that applies func"""
    def apply_func(data: np.ndarray) -> float:
        """applies simple function func on input data
        :param data: a vector of values of the group working_on_dataset (see EventDataFeature.working_on_dataset)
        """
        return func(data)
    return apply_func
