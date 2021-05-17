from tests.utils.data_creator.tdms_file_creator import MakeTdmsFile
from pathlib import Path
import h5py  #type: ignore


class MakeTestFiles(MakeTdmsFile):
    """Used to create a XBox2 like tdms file. Can be set up to either be similar to event or trend data."""
    def __init__(self, hdf_file_path: Path, tdms_file_path: Path, root_prop_dict: dict):
        super().__init__(tdms_file_path = tdms_file_path,
                         tdms_root_properties = root_prop_dict)
        with h5py.File(hdf_file_path, "w") as file:
            for key in root_prop_dict.keys():
                file.attrs.create(key, root_prop_dict[key])
        self.hdf_file_path = hdf_file_path
        self.ch_prop_dict: dict = {}
        self.ch_data_dict: dict = {}
        self.grp_prop_dict: dict = {}

    def test_ch_prop_and_data(self) -> set:
        inters = set(self.ch_prop_dict.keys()).intersection(set(self.ch_data_dict))
        union = set(self.ch_prop_dict.keys()).union(set(self.ch_data_dict))
        if inters != union:
            raise RuntimeWarning(f"ch_prop_dict and ch_data_dict have differnt keys (=channel names).\n" +
                f"Got ch_prop_dict.keys() = {self.ch_prop_dict.keys()} and " +
                f"ch_data_dict.keys() = {self.ch_data_dict.keys()}.\n" +
                "Some channels might be discarted for the creation of the tdms file.")
        return inters

    def add_artificial_group(self, grp_name) -> None:
        ch_set = self.test_ch_prop_and_data()
        # create tdms group
        self.add_tdms_grp(grp_name=grp_name, grp_properties=self.grp_prop_dict)
        for chn in ch_set:
            self.add_tdms_ch(grp_name=grp_name,
                             ch_name=chn,
                             ch_properties=self.ch_prop_dict[chn],
                             data=self.ch_data_dict[chn])
        # create hdf group
        with h5py.File(self.hdf_file_path, "a") as file:
            file.create_group(grp_name)
            for chn in ch_set:
                file[grp_name].create_dataset(name=chn,
                                    data=self.ch_data_dict[chn])
                for key in self.ch_prop_dict[chn].keys():
                    if key=="wf_start_time" or key=="Timestamp":
                        data = f"{self.ch_prop_dict[chn][key]}"  #h5py does not like datetime
                    else:
                        data = self.ch_prop_dict[chn][key]
                    file[grp_name].attrs.create(key, data)
