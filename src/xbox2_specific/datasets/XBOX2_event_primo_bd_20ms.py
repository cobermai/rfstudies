"""Selecting from context data and prepare dataset XBOX2_event_bd20ms for machine learning. """
from collections import namedtuple
from pathlib import Path
import xarray as xr
import h5py
from src.xbox2_specific.utils import dataset_utils
from src.xbox2_specific.datasets.XBOX2_abstract import XBOX2Abstract

data = namedtuple("data", ["X", "y", "idx"])


class XBOX2EventPrimoBD20msSelect(XBOX2Abstract):
    """
    Subclass of XBOX2Abstract to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    @staticmethod
    def select_events(data_path: Path) -> xr.DataArray:
        """
        selection of events in data
        :param data_path: path to context data file
        :return: data array with data of selected events
        """
        # select events with breakdown in 20 ms
        bd_selection_list = ["is_bd_in_20ms"]
        selection = dataset_utils.select_events_from_list(context_data_file_path=data_path / "context.hdf",
                                                          selection_list=bd_selection_list)

        # read features into data_array
        feature_list = ["PEI Amplitude",
                        "PSI Amplitude",
                        "PSR Amplitude",
                        "PKI Amplitude",
                        "DC Up",
                        "DC Down"]
        with h5py.File(data_path / "context.hdf", 'r') as file:
            ext_link_index = dataset_utils.read_hdf_dataset(file, "event_ext_link_index")[selection]
        data_array = dataset_utils.event_ext_link_hdf_to_da_timestamp(file_path=data_path / "EventDataExtLinks.hdf",
                                                                      ext_link_index=ext_link_index,
                                                                      feature_list=feature_list)

        # read label and metadata
        label_name = "is_bd_in_20ms"

        with h5py.File(data_path / "context.hdf", 'r') as file:
            is_bd_in_20ms = dataset_utils.read_hdf_dataset(file, label_name)[selection]
            timestamp = dataset_utils.read_hdf_dataset(file, "Timestamp")[selection]
            run_no = dataset_utils.read_hdf_dataset(file, "run_no")[selection]
            is_followup = dataset_utils.determine_followup(is_bd_in_20ms, timestamp, threshold=60)

        # add label to data_array
        data_array = data_array.assign_coords(is_bd_in_20ms=("event", is_bd_in_20ms))

        # add meta data
        data_array = data_array.assign_coords(run_no=("event", run_no))
        data_array = data_array.assign_coords(timestamp=("event", timestamp))
        data_array = data_array.assign_coords(is_followup=("event", is_followup))

        return data_array

    @staticmethod
    def select_features(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns features of selected events for modeling
        :param data_array: xarray DataArray with data
        :return: xarray DataArray with features of selected events
        """
        X_data_array = data_array[data_array["is_followup"] == False]
        X_data_array = X_data_array.drop_vars('is_bd_in_20ms')
        return X_data_array

    @staticmethod
    def select_labels(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns labels of selected events for supervised machine learning
        :param data_array: xarray data array of data from selected events
        :return: labels of selected events
        """
        label_name = "is_bd_in_20ms"
        y_data_array = data_array[label_name]
        y_data_array = y_data_array[y_data_array["is_followup"] == False]
        return y_data_array
