"""future data handler. UNFINISHED"""
import os.path
import numpy as np
import h5py
from src.utils.system.progress import working_on

class Handler():
    """handles data UNFINISHED"""
    def __init__(self, td_file_path: str, ed_file_path: str):
        self.td_file_path = td_file_path
        self.ed_file_path = ed_file_path

    def td_combine(self, combine_file_path):
        """combines trend data into one large data set
        file path of the already combined trend data"""
        with h5py.File(self.td_file_path, mode = "r") as trend_data,\
                h5py.File(combine_file_path, mode = "w") as td_ds:
            key_list_in_order = list(trend_data.keys())
            vec_is_corrupt = np.array(False)
            for chn in trend_data[list(trend_data.keys())[0]].keys():
                data = np.concatenate([trend_data[key][chn][:] for key in key_list_in_order])
                try:
                    vec_is_corrupt += np.isnan(data) + np.isinf(data)
                    working_on("is nan: " + str(np.isnan(data).sum()) + "   " +
                               "is inf: " + str(np.isinf(data).sum()) + "   " +
                                "is corrupt total: " + str(vec_is_corrupt.sum()))
                except ValueError:
                    print("is nan does not work for " + chn)

            data = np.concatenate([trend_data[key]["Timestamp"][:] for key in key_list_in_order])
            indices_order = np.array(data[np.logical_not(vec_is_corrupt)], np.datetime64).argsort()
            for chn in trend_data[list(trend_data.keys())[0]].keys():
                print("wokring on " + chn)
                data = np.concatenate([trend_data[key][chn][:] for key in key_list_in_order])
                data = data[np.logical_not(vec_is_corrupt)]
                data = data[indices_order]
                td_ds.create_dataset(chn, data=data)

    def other_public_methods(self) -> None:
        """UNIFINISHED"""
        raise NotImplementedError("no more public methods implemented yet")


if __name__ == "__main__":
    handler = Handler(os.path.expanduser("~/output_files/TrendDataVirtual.vhdf"),
                    os.path.expanduser("~/output_files/EventDataVirtual.vhdf"))
    handler.td_combine(os.path.expanduser("~/output_files/TrendDataVDS.vhdf"))
    tdds = h5py.File(os.path.expanduser("~/output_files/TrendDataVDS.vhdf"), mode="r")
