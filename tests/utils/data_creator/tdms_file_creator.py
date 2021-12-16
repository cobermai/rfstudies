"""Module makes creating tdms files easier. The original creation method is very specific to nptdms."""
from pathlib import Path
from typing import Set, Union

import nptdms


class CreatorTdmsFile:
    """A tool to make creating tdms files easier."""

    def __init__(self, tdms_file_path: Path, tdms_root_properties: dict):
        root_object = nptdms.RootObject(properties=tdms_root_properties.copy())
        self.tdms_file_path: Path = tdms_file_path.absolute()
        self.tdms_object_set: Set[Union[nptdms.RootObject, nptdms.GroupObject, nptdms.ChannelObject]] = {root_object}
        # tdms_object_set will be additionally filled with groups and channels

    def add_tdms_grp(self, grp_name: str, grp_properties: dict) -> None:
        """
        add a group object to the segment set that will eventually be added to the tdms object
        :param grp_name: name of the group to add
        :param grp_properties: a dictionary of the group properties
        """
        grp = nptdms.GroupObject(grp_name, properties=grp_properties.copy())
        self.tdms_object_set.update({grp})

    def add_tdms_ch(self, grp_name, ch_name: str, ch_properties: dict, data: list) -> None:
        """
        add a channel object to the segment set that will eventually be added to the tdms object
        :param grp_name: name of the parent group
        :param ch_name: name of the channel
        :param ch_properties: dictionary of the channel properties
        :param data: dictionary of the channel data
        """
        channel = nptdms.ChannelObject(grp_name, ch_name, data, properties=ch_properties.copy())
        self.tdms_object_set.update({channel})

    def write(self):
        """write the tdms segment set with root object, group objects and channel objects into one tdms file"""
        with nptdms.TdmsWriter(self.tdms_file_path) as writer:
            writer.write_segment(self.tdms_object_set)
