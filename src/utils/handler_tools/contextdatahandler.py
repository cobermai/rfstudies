"""writes to the context data file row by row or column by column. If the destination dataset does not exist, it is
created."""
from pathlib import Path
import typing
from dataclasses import dataclass
import h5py
from src.utils.handler_tools.customfeature import ColumnWiseFeature


@dataclass
class ContextDataHandler:
    """An abstract context data handler class. Different writing methods are inherited."""
    context_data_file_path: Path
    length: int


class ColumnWiseContextDataHandler(ContextDataHandler):  # pylint: disable=too-few-public-methods
    """makes creating the context data easier for column wise writing. Column wise means writing for single feature
    applied on all all events.
    """
    def write_clm(self, feature: ColumnWiseFeature) -> None:
        """writes a column placed in feature.vec into a specific path given by feature.full_hdf_path. If that path does
        not exist, it will be created.
        :param feature: a ColumnWiseFeature with a filled feature.vec
        """
        with h5py.File(self.context_data_file_path, "a") as file:
            dataset = file.require_dataset(name=feature.full_hdf_path,
                                           shape=(self.length,),
                                           dtype=feature.vec.dtype,
                                           chunks=True)
            dataset[:] = feature.vec
            dataset.attrs.create(name="info", data=feature.info)


class RowWiseContextDataHandler(ContextDataHandler):
    """makes creating the context data easier for row wise writing. Row wise means writing all row wise features
    applied on a single event. """
    def write_row_custom_features(self, index: int, data_iter: typing.Iterable):
        """writes a single value into all specified hdf paths. if the destination hdf path does not exist, it will be
        created.
        :param index: the location of the event.
        :param data_iter: an iterable (usually a generator) that produces tuples of (hdf_path, value). The hdf path is
        the dataset at which where the value "value" will be placed into the position given by the index.
        """
        with h5py.File(self.context_data_file_path, "a") as file:
            for feature, value in data_iter:
                dataset = file.get(feature.full_hdf_path, None)
                if dataset is None:
                    dataset = file.require_dataset(feature.full_hdf_path,
                                              shape=(self.length,),
                                              dtype=type(value),
                                              chunks=True)
                    dataset.attrs.create(name="info", data=feature.info)
                dataset[index] = value

    def write_row_from_external(self, index: int, data_iter: typing.Iterable):
        """writes a single value into all specified hdf paths. if the destination hdf path does not exist, it will be
        created.
        :param index: the location of the event.
        :param data_iter: an iterable (usually a generator) that produces tuples of (hdf_path, value). The hdf path is
        the dataset at which where the value "value" will be placed into the position given by the index.
        """
        with h5py.File(self.context_data_file_path, "a") as file:
            for hdf_path, value in data_iter:
                dataset = file.get(hdf_path, None)
                if dataset is None:
                    dataset = file.require_dataset(hdf_path,
                                              shape=(self.length,),
                                              dtype=type(value),
                                              chunks=True)
                    # no info string will be added
                dataset[index] = value
