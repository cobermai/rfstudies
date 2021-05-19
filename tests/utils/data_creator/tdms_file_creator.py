from typing import Union, Set
from pathlib import Path
import nptdms  #type: ignore

class CreatorTdmsFile():
    """A tool to make creating tdms files easier."""
    tdms_file_path: Path
    tdms_object_set: Set[Union[nptdms.RootObject, nptdms.GroupObject, nptdms.ChannelObject]]

    def __init__(self, tdms_file_path: Path, tdms_root_properties: dict):
        root_object = nptdms.RootObject(properties=tdms_root_properties)
        self.tdms_file_path = tdms_file_path.absolute()
        self.tdms_object_set = {root_object}  # this set will be additionally filled with groups and channels

    def add_tdms_grp(self, grp_name: str, grp_properties: dict) -> None:
        """
        add a group object to the segment set that will eventually be added to the tdms object
        :param grp_name: name of the group to add
        :param grp_properties: a dictonary of the gorup properties
        """
        grp = nptdms.GroupObject(grp_name, properties=grp_properties)
        self.tdms_object_set.update({grp})

    def add_tdms_ch(self, grp_name, ch_name: str, ch_properties: dict, data: list) -> None:
        """
        add a channel object to the segment set that will eventually be written out
        :param grp_name: name of the group the channel is in
        :param ch_name: name of the channel
        :param ch_properties: dictonary of the channel properties
        :param data: dictornary of the channel data
        """
        ch = nptdms.ChannelObject(grp_name, ch_name, data, properties=ch_properties)
        self.tdms_object_set.update({ch})

    def write(self):
        """write the tdms segment set with root object, group objects and channel objects into one tmds file"""
        with nptdms.TdmsWriter(self.tdms_file_path) as writer:
            writer.write_segment(self.tdms_object_set)

    def __del__(self):
        """automatically writes tdms files when object is deleted"""
        self.write()