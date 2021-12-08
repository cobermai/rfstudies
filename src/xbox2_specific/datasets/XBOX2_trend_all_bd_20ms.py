"""Selecting from context data and prepare dataset XBOX2_trend_bd20ms for machine learning. """
from collections import namedtuple
from pathlib import Path
import numpy as np
import xarray as xr
import h5py
from src.xbox2_specific.utils import dataset_utils
from src.xbox2_specific.datasets.XBOX2_abstract import XBOX2Abstract

data = namedtuple("data", ["X", "y", "idx"])


class XBOX2TrendAllBD20msSelect(XBOX2Abstract):
    """
    Subclass of XBOX2Abstract to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """
    @staticmethod
    def select_events(data_path: Path) -> list:
        """
        selection of events in data
        :param data_path: path to context data file
        :return: boolean filter for selecting breakdown events
        """
        # select only events with breakdown in 20 ms + some healthy events
        bd_selection_list = ["is_bd_in_20ms"]
        selection = dataset_utils.select_events_from_list(context_data_file_path=data_path / "context.hdf",
                                                          selection_list=bd_selection_list)

        # read features into data_array
        feature_list = ["Loadside win", "Tubeside win",
                        "Collector", "Gun", "IP before PC",
                        "PC IP", "WG IP", "IP Load",
                        "IP before structure", "US Beam Axis IP",
                        "Klystron Flange Temp", "Load Temp",
                        "PC Left Cavity Temp", "PC Right Cavity Temp",
                        "Bunker WG Temp", "Structure Input Temp",
                        "Chiller 1", "Chiller 2", "Chiller 3",
                        "PKI FT avg", "PSI FT avg", "PSR FT avg",
                        "PSI max", "PSR max", "PEI max",
                        "DC Down min", "DC Up min",
                        "PSI Pulse Width"]
        label_name = "is_bd_in_20ms"

        with h5py.File(data_path / "context.hdf") as file:
            # Get real timestamp
            timestamp_trend_selection = dataset_utils.read_hdf_dataset(file, "PrevTrendData/Timestamp")[selection]
            # remove duplicate timestamps
            timestamp_trend_selection, unique_selection = np.unique(timestamp_trend_selection, return_index=True)
            # Get label and meta data
            is_bd_in_20ms = dataset_utils.read_hdf_dataset(file, label_name)[selection]
            is_bd_in_20ms = is_bd_in_20ms[unique_selection]
            timestamp = dataset_utils.read_hdf_dataset(file, "Timestamp")[selection]
            timestamp = timestamp[unique_selection]
            run_no = dataset_utils.read_hdf_dataset(file, "run_no")[selection]
            run_no = run_no[unique_selection]

        # Get selected features
        with h5py.File(data_path / "TrendDataFull.hdf") as file:
            # Read trend data timestamps and compare to selected
            trend_timestamp = file["Timestamp"][:]
            trend_selection = np.in1d(trend_timestamp, timestamp_trend_selection)

            # Create filter for selecting two previous trend data
            trend_selection_one_before = dataset_utils.shift_values(np.array(trend_selection), -1, fill_value=False)
            trend_selection_two_before = dataset_utils.shift_values(np.array(trend_selection), -2, fill_value=False)

            # Read selected features
            data_read = np.empty(shape=(np.sum(trend_selection), 3, len(feature_list)))
            for feature_ind, feature in enumerate(feature_list):
                data_read[:, 0, feature_ind] = dataset_utils.read_hdf_dataset(file, feature)[trend_selection_two_before]
                data_read[:, 1, feature_ind] = dataset_utils.read_hdf_dataset(file, feature)[trend_selection_one_before]
                data_read[:, 2, feature_ind] = dataset_utils.read_hdf_dataset(file, feature)[trend_selection]

        # Create xarray DataArray
        dim_names = ["event", "sample", "feature"]
        feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
        data_array = xr.DataArray(data=data_read,
                                  dims=dim_names,
                                  coords={"feature": feature_names})
        # add label to data_array
        data_array = data_array.assign_coords(is_bd_in_20ms=("event", is_bd_in_20ms))
        # add meta data
        data_array = data_array.assign_coords(run_no=("event", run_no))
        data_array = data_array.assign_coords(timestamp=("event", timestamp))
        return data_array

    @staticmethod
    def select_features(data_array: xr.DataArray) -> xr.DataArray:
        """
        returns features of selected events for modeling
        :param data_array: xarray DataArray with data
        :return: xarray DataArray with features of selected events
        """
        X_data_array = data_array.drop_vars('is_bd_in_20ms')
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
        return y_data_array
