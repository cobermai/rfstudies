from pathlib import Path
import glob
import h5py
import numpy as np
from utils.hdf_utils.tdms_read import Convert
from utils.hdf_utils.gather import Gather
from utils.system.logger import logger
from utils.system.logger import logger_add_tg

log = logger("DEBUG")
logger_add_tg(log, "INFO")


def transform(tdms_dir: str, hdf_dir: str) -> None:
    """
    transforms all tdms files to hdf files, filters faulty data and gathers everything with external links.
    :param tdms_dir: input directory with tdms files
    :param hdf_dir: output directory with hdf files
    """
    Path(hdf_dir + "/data/").mkdir(parents=False, exist_ok=True)

    ## read tdms files and convert them to hdf5
    Convert(check_already_converted=True, num_processes=4)\
        .from_tdms(tdms_dir)\
        .to_hdf(hdf_dir + "/data/").run()

    ## Combining all Events and TrendData sets into one hdf5 file with external links
    def td_func_to_fulfill(file_path: str, hdf_path: str) -> bool:
        with h5py.File(file_path, "r") as file:
            ch_shapes = [file[hdf_path][key].shape[0] for key in file[hdf_path].keys()]
            len_equal = all(ch_shape == ch_shapes[0] for ch_shape in ch_shapes)
            return len_equal and len(file[hdf_path].keys()) == 35

    Gather().from_files(glob.glob(hdf_dir +  "/data/Trend*.hdf"))\
        .to_hdf_file(hdf_dir + "TrendDataExtLinks.hdf")\
        .if_fulfills(td_func_to_fulfill, on_error=False)\
        .with_external_links()


    def ed_func_to_fulfill(file_path: str, hdf_path: str)->bool:
        with h5py.File(file_path, "r") as file:
            ch_len = [file[hdf_path][key].shape[0] for key in file[hdf_path].keys()]
            print(file_path + hdf_path)
            return file[hdf_path].attrs.get("Timestamp", None) is not None and \
                ch_len.count(500)==8 and \
                ch_len.count(3200)==8 and \
                not any((any(np.isnan(file[hdf_path][key][:])) for key in file[hdf_path].keys()))

    Gather().from_files(glob.glob(hdf_dir +  "/data/Event*.hdf"))\
        .to_hdf_file(hdf_dir + "EventDataExtLinks.hdf")\
        .if_fulfills(ed_func_to_fulfill, on_error=True)\
        .with_external_links()

if __name__=="__main__":
    transform(tdms_dir = Path.expanduser("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/"),
                   hdf_dir = Path.expanduser("~/output_files/"))
