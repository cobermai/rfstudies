"""This module contains a class class that defines machine learning features. The Feature class contains the functions
apply and write."""
from dataclasses import dataclass, field
import typing
import logging
import numpy as np
from src.utils.hdf_tools import hdf_path_combine

log = logging.getLogger(__name__)


@dataclass
class CustomFeature:
    """A feature is a statistical property of a time series (min, max, mean, pulse_amplitude, pulse_length, etc) or a
    measurement that is also related to the time series.
    One object represents one Feature and can be applied on an hdf5 dataset or attribute."""
    name: str  # name of the feature, also the name of the dataset in the full_hdf_path
    func: typing.Callable  # the feature function
    hdf_path: str  # hdf_path is the path of the hdf-group where the feature with self.name will be placed in the
    # context data file (= destination file)
    info: str  # will be written into the

    @property
    def full_hdf_path(self) -> str:
        """returns the full hdf path, of the feature. starting from root, ending with the hdf-dataset name."""
        return hdf_path_combine(self.hdf_path, self.name)


@dataclass
class ColumnWiseFeature(CustomFeature):
    """a parent class of all features calculated and written all at once (=column wise)."""
    length: int
    output_dtype: typing.Union[type, np.dtype]
    vec: typing.Any = field(init=False)

    def __post_init__(self):
        self.vec = np.empty(shape=(self.length,), dtype=self.output_dtype)


class EventAttributeFeature(ColumnWiseFeature):
    """represents features read from the event attributes"""

    def calc(self, index: int, attrs):
        """
        calculates the event attribute feature by applying self.func and writes it to the self.vec at the given index.
        :param index: index of the event and thus location where the calculated feature will be written.
        :param attrs: attribute of the event data """
        self.vec[index] = self.func(attrs)


class TrendDataFeature(ColumnWiseFeature):
    """Features for time series from the TrendData.
    The feature.func selects the wanted values from the time series."""
    def calc_all(self, selection):
        """
        applies the feature function on the trend data and the pre calculated context_data
        :param selection: selection of interest from the trend data
        :return: numpy array of datatype self.dtype
        """
        return self.func(selection)


@dataclass
class RowWiseFeature(CustomFeature):
    """A parent class of all features calculated row by row (=event by event)."""
    working_on_dataset: str

    def apply(self, data):
        """applies the feature.func to the given time series and returns the feature value."""
        return self.func(data[self.working_on_dataset])


class EventDataFeature(RowWiseFeature):
    """Features for time series from the EventData.
    The feature.func processes the time series."""
