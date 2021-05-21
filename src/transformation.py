"""
This module provides tools to transform data to a capable format so that further analyzing can be done easily.
"""
from pathlib import Path
from typing import Union
import h5py
import numpy as np
from src.utils.hdf_utils.tdms_read import Convert
from src.utils.hdf_utils.gather import Gather
from src.utils.system.logger import logger
from src.utils.system.logger import logger_add_tg

log = logger("DEBUG")
try:
    logger_add_tg(log, "DEBUG")
except FileNotFoundError:
    pass

def transform(tdms_dir: Union[Path, str], hdf_dir: Union[Path, str]) -> None:
    """
    transforms all tdms files to hdf files, filters faulty data and gathers hdf groups with depth 1 of the hdf files
    into one hdf file with external links.
    :param tdms_dir: input directory with tdms files
    :param hdf_dir: output directory with hdf files
    """
    tdms_dir = Path(tdms_dir)
    hdf_dir = Path(hdf_dir)

    Path(hdf_dir,"data").mkdir(parents=False, exist_ok=True)

    ## read tdms files and convert them to hdf5 and writing them into hdf_dir/data/
    Convert(check_already_converted=True, num_processes=4)\
        .from_tdms(tdms_dir)\
        .to_hdf(hdf_dir / "data").run()

    ## Combining all Events and TrendData sets into one hdf5 file with external links if they are not faulty
    def td_func_to_fulfill(file_path: Path, hdf_path: str) -> bool:
        with h5py.File(file_path, "r") as file:
            ch_shapes = [file[hdf_path][key].shape[0] for key in file[hdf_path].keys()]
            len_equal = all(ch_shape == ch_shapes[0] for ch_shape in ch_shapes)
            return len_equal and len(file[hdf_path].keys()) == 35

    Gather(num_processes=4)\
        .from_files(hdf_dir.glob("data/Trend*.hdf"))\
        .to_hdf_file(hdf_dir / "TrendDataExtLinks.hdf")\
        .if_fulfills(td_func_to_fulfill, on_error=False)\
        .run_with_external_links()


    def ed_func_to_fulfill(file_path: Path, hdf_path: str)->bool:
        with h5py.File(file_path, "r") as file:
            ch_len = [file[hdf_path][key].shape[0] for key in file[hdf_path].keys()]
            return file[hdf_path].attrs.get("Timestamp", None) is not None and \
                ch_len.count(500)==8 and \
                ch_len.count(3200)==8 and \
                not any((any(np.isnan(file[hdf_path][key][:])) for key in file[hdf_path].keys()))

    Gather().from_files(hdf_dir.glob("data/Event*.hdf"))\
        .to_hdf_file(hdf_dir / "EventDataExtLinks.hdf")\
        .if_fulfills(ed_func_to_fulfill, on_error=True)\
        .run_with_external_links()


if __name__=="__main__":
    transform(tdms_dir = Path("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/").expanduser(),
                   hdf_dir = Path("~/output_files/").expanduser())
