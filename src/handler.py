import os.path
import numpy as np
import h5py

class Handler():
    def __init__(self, td_file_path: str, ed_file_path: str):
        self.td_file_path = td_file_path
        self.ed_file_path = ed_file_path

    def td_combine(self, combine_file_path):
        with h5py.File(self.td_file_path, mode = "r") as td, h5py.File(combine_file_path, mode = "w") as td_ds:
            key_list_in_order = list(td.keys())
            for chn in [key for key in td[list(td.keys())[0]].keys()]:
                td_ds.create_dataset(chn, data= np.concatenate([td[key][chn][:] for key in key_list_in_order]))

if __name__=="__main__":
    handler = Handler(os.path.expanduser("~/output_files/TrendDataVirtual.vhdf"),
                    os.path.expanduser("~/output_files/EventDataVirtual.vhdf"))
    handler.td_combine(os.path.expanduser("~/output_files/TrendDataVDS.vhdf"))