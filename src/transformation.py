import os
import h5py
from src.utils.hdf_utils.tdms_read import Convert
from src.utils.hdf_utils.filter import Filter
from src.utils.hdf_utils.combine import ExternalLinks
import glob
from src.utils.system.logger import logger
log = logger("DEBUG")
import numpy as np


def transform(tdms_dir: str, hdf_dir: str):
    ## read tdms files and convert them to hdf5
    Convert(check_already_converted=True, num_processes=4)\
        .from_tdms(tdms_dir)\
        .to_hdf(hdf_dir + "data/").run()

    ## Combining all Events and TrendData sets into one hdf5 file with external links
    def td_func_to_fulfill(file_path: str, hdf_path: str) -> bool:
        ret = True
        with h5py.File(file_path, "r") as f:
            ch_shapes = [f[hdf_path][key].shape[0] for key in f[hdf_path].keys()]
            len_equal = all([ch_shape == ch_shapes[0] for ch_shape in ch_shapes])
            return len_equal and len(f[hdf_path].keys()) == 35

    ExternalLinks(to_file=hdf_dir + "TrendDataExtLinks.hdf")\
        .from_files(glob.glob(hdf_dir +  "data/Trend*.hdf"))\
        .if_fulfills(td_func_to_fulfill, on_error=False)


    def ed_func_to_fulfill(file_path: str, hdf_path: str)->bool:
        with h5py.File(file_path, "r") as f:
            ch_len = [f[hdf_path][key].shape[0] for key in f[hdf_path].keys()]
            return hdf_path[-1]==str(1) and\
                f[hdf_path].attrs.get("Timestamp", None)!=None and \
                ch_len.count(500)==8 and \
                ch_len.count(3200)==8 and \
                not any([any(np.isnan(f[hdf_path][key][:])) for key in f[hdf_path].keys()])

    ExternalLinks(to_file=hdf_dir + "EventDataExtLinks.hdf")\
        .from_files(glob.glob(hdf_dir + "data/Event*.hdf"))\
        .if_fulfills(ed_func_to_fulfill, on_error=False)

    exit()

if __name__=="__main__":
    transform(tdms_dir = os.path.expanduser("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/"),
                   hdf_dir = os.path.expanduser("~/output_files/"))
    # transformation(tdms_dir = "/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/CLIC_DATA_Xbox2_T24PSI_2/",
    #               hdf_dir = "/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf/")