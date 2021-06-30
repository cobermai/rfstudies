"""This module contains a class class that defines machine learning features. The Feature class contains the functions
apply and write."""
from dataclasses import dataclass
import typing
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
import h5py
from src.utils.hdf_tools import hdf_path_combine
from src.utils.hdf_tools import hdf_path_combine

log = logging.getLogger(__name__)


@dataclass
class CustomFeature:
    """A feature is a statistical property of a time series (min, max, mean, pulse_amplitude, pulse_length, etc).
    One object represents one Feature and can be applied on an hdf5 dataset via the apply function and written via write
    This is the base class"""
    name: str
    func: typing.Callable
    output_dtype: typing.Any  # some h5py compatible data type
    hdf_path: str  # hdf_path of in the context data file
    info: str


    @property
    def full_hdf_path(self):
        return hdf_path_combine(self.hdf_path, self.name)


init_later = object()
@dataclass
class ColumnWiseFeature(CustomFeature):
    length: int
    vec: typing.Any = init_later
    def __post_init__(self):
        self.vec = np.empty(shape=(self.length,), dtype=self.output_dtype)


class EventAttributeFeature(ColumnWiseFeature):
    """calculates features from the event data attributes."""

class TrendDataFeature(ColumnWiseFeature):
    """This feature class creates features out of trend data and existing context_data calculated by the
    EventDataFeatures."""
    def calc(self, selection):
        """
        applies the feature function on the trend data and the pre calculated context_data
        :param td_file_path: file path of the source file (an hdf file)
        :return: numpy array of datatype self.dtype
        """
        return self.func(selection)


class RowWiseFeature(CustomFeature):
    """"""

class EventDataFeature(RowWiseFeature):
    """calculates features from the event data."""
    def apply(self, data):
        """applies the function of the feature to the given data and returns the feature value."""
        return self.func(data[self.hdf_path])
