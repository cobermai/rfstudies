import os.path
import numpy as np
import h5py
from utils.system.progress import working_on
class Handler():
    def __init__(self, td_file_path: str, ed_file_path: str):
        self.td_file_path = td_file_path
        self.ed_file_path = ed_file_path

    def td_combine(self, combine_file_path):
        with h5py.File(self.td_file_path, mode = "r") as td,\
                h5py.File(combine_file_path, mode = "w") as td_ds:
            key_list_in_order = list(td.keys())
            vec_is_corrupt = False
            for chn in [key for key in td[list(td.keys())[0]].keys()]:
                data = np.concatenate([td[key][chn][:] for key in key_list_in_order])
                try:
                    vec_is_corrupt += np.isnan(data) + np.isinf(data)
                    working_on("is nan: " + str(np.isnan(data).sum()) + "   " +
                               "is inf: " + str(np.isinf(data).sum()) + "   " +
                                "is corrupt total: " + str(vec_is_corrupt.sum()))
                except:
                    print("is nan does not work for " + chn)

            data = np.concatenate([td[key]["Timestamp"][:] for key in key_list_in_order])
            indices_order = np.array(data[np.logical_not(vec_is_corrupt)], np.datetime64).argsort()
            for chn in [key for key in td[list(td.keys())[0]].keys()]:
                print("wokring on " + chn)
                data = np.concatenate([td[key][chn][:] for key in key_list_in_order])
                data = data[np.logical_not(vec_is_corrupt)]
                data = data[indices_order]
                td_ds.create_dataset(chn, data=data)

if __name__=="__main__":
    handler = Handler(os.path.expanduser("~/output_files/TrendDataVirtual.vhdf"),
                    os.path.expanduser("~/output_files/EventDataVirtual.vhdf"))
    handler.td_combine(os.path.expanduser("~/output_files/TrendDataVDS.vhdf"))
    tdds = h5py.File(os.path.expanduser("~/output_files/TrendDataVDS.vhdf"), mode="r")
from pathlib import Path