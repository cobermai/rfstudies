"""
class for creating tdms and hdf files for testing simultaneously. The hdf files are generated as expected output of the
transformation.
"""
from pathlib import Path

import h5py
import numpy as np

from tests.utils.data_creator.tdms_file_creator import CreatorTdmsFile


def _hdf_array(data):
    """ Convert data array into a format suitable for initialising HDF data. (See nptdms/export/hdf_export.py)"""
    if isinstance(data[0], np.datetime64):
        string_data = np.datetime_as_string(data, unit='us', timezone='UTC')
        return [s.encode('ascii') for s in string_data]
    return data


def _hdf_attr_value(value):
    """ Convert a value into a format suitable for an HDF attribute. (See nptdms/export/hdf_export.py)"""
    if isinstance(value, np.datetime64):
        return np.string_(np.datetime_as_string(value, unit='us', timezone='UTC'))
    return value


class CreatorTestFiles(CreatorTdmsFile):
    """Creator class of tdms and hdf files simultaneously for testing."""

    def __init__(self, hdf_file_path: Path, tdms_file_path: Path, root_prop_dict: dict):
        super().__init__(tdms_file_path=tdms_file_path,
                         tdms_root_properties=root_prop_dict)
        with h5py.File(hdf_file_path, "w") as file:
            for key in root_prop_dict:
                file.attrs.create(key, _hdf_attr_value(root_prop_dict[key]))
        self.hdf_file_path = hdf_file_path
        self.grp_prop_dict: dict = {}
        self.ch_prop_dict: dict = {}
        self.ch_data_dict: dict = {}

    def test_ch_prop_and_data(self) -> set:
        """
        checks if the channel property dict keys are similar to the channel data dict keys
        :return: intersection of channel property dict keys and channel data dict keys as a set
        """
        inters = set(self.ch_prop_dict.keys()).intersection(set(self.ch_data_dict))
        union = set(self.ch_prop_dict.keys()).union(set(self.ch_data_dict))
        if inters != union:
            raise RuntimeWarning(f"""ch_prop_dict and ch_data_dict have different keys (=channel names).
                Got ch_prop_dict.keys() = {self.ch_prop_dict.keys()} and
                ch_data_dict.keys() = {self.ch_data_dict.keys()}.
                Some channels might be discarded for the creation of the tdms file.""")
        return inters

    def add_artificial_group(self, grp_name) -> None:
        """
        adds a group ("artificial" because its "man made") to the tdms creator and the hdf file
        :param grp_name: the group name to add (the rest is already in the class)
        """
        ch_set = self.test_ch_prop_and_data()
        # create tdms group
        self.add_tdms_grp(grp_name=grp_name, grp_properties=self.grp_prop_dict.copy())
        for chn in ch_set:
            self.add_tdms_ch(grp_name=grp_name,
                             ch_name=chn,
                             ch_properties=self.ch_prop_dict[chn].copy(),
                             data=self.ch_data_dict[chn])
        # create hdf group
        with h5py.File(self.hdf_file_path, "a") as file:
            file.create_group(grp_name)
            for key, val in self.grp_prop_dict.items():
                file[grp_name].attrs.create(key, _hdf_attr_value(val))
            for chn in ch_set:
                file[grp_name].create_dataset(name=chn, data=_hdf_array(self.ch_data_dict[chn]))
                for key in self.ch_prop_dict[chn].keys():
                    file[grp_name][chn].attrs.create(key, _hdf_attr_value(self.ch_prop_dict[chn][key]))
