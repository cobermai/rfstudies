"""
This module provides tools to transform data to a capable format so that further analyzing can be done easily.
"""
from pathlib import Path
from typing import Union
import logging
import h5py
import numpy as np
from src.utils.transf_tools.convert import Convert
from src.utils.transf_tools.gather import Gather
from src.utils.system.setup_logging import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)

def transform(tdms_dir: Union[Path, str], hdf_dir: Union[Path, str]) -> None:
    """
    transforms all tdms files to hdf files, filters faulty data and gathers hdf groups with depth 1 of the hdf files
    into one hdf file with external links.
    :param tdms_dir: input directory with tdms files
    :param hdf_dir: output directory with hdf files
    """
    tdms_dir = Path(tdms_dir)
    hdf_dir = Path(hdf_dir)

    Path(hdf_dir, "data").mkdir(parents=False, exist_ok=True)

    # read tdms files and convert them to hdf5 and writing them into hdf_dir/data/
    Convert(check_already_converted=True, num_processes=4)\
        .from_tdms(tdms_dir)\
        .to_hdf(hdf_dir / "data").run()

    # Combining all Events and TrendData sets into one hdf5 file with external links if they are not faulty
    def td_func_to_fulfill(file_path: Path, hdf_path: str) -> bool:
        with h5py.File(file_path, "r") as file:
            ch_shapes: list = [file[hdf_path][key].shape[0] for key in file[hdf_path].keys()]
            len_equal: bool = all(ch_shape == ch_shapes[0] for ch_shape in ch_shapes)
            num_of_samples: int = 35
            return len_equal and len(file[hdf_path].keys()) == num_of_samples

    Gather(num_processes=4)\
        .from_files(hdf_dir.glob("data/Trend*.hdf"))\
        .to_hdf_file(hdf_dir / "TrendDataExtLinks.hdf")\
        .if_fulfills(td_func_to_fulfill, on_error=False)\
        .run_with_external_links()

    def ed_func_to_fulfill(file_path: Path, hdf_path: str) -> bool:
        with h5py.File(file_path, "r") as file:
            ch_len: list = [file[hdf_path][key].shape[0] for key in file[hdf_path].keys()]

            acquisation_window: float = 2e-6  # time period of one event is 2 microseconds

            # acquisition card NI-5772 see https://www.ni.com/en-us/support/model.ni-5772.html
            sampling_frequency_ni5772: int = int(1.6e8)  # sampling frequency of the acquisition card
            num_of_values_ni5772: int = int(acquisation_window * sampling_frequency_ni5772)
            number_of_signals_monitored_with_ni5772: int = 8

            # acquisition card NI-5761 see https://www.ni.com/en-us/support/model.ni-5761.html
            sampling_frequency_ni5761: int = int(1.6e9)  # sampling frequency of the acquisition card
            num_of_values_ni5761: int = int(acquisation_window * sampling_frequency_ni5761)
            number_of_signals_monitored_with_ni5761: int = 8

            def has_smelly_vals(data) -> bool:
                return any(np.isnan(data) | np.isinf(data))

            return file[hdf_path].attrs.get("Timestamp", None) is not None \
                and ch_len.count(num_of_values_ni5772) == number_of_signals_monitored_with_ni5772 \
                and ch_len.count(num_of_values_ni5761) == number_of_signals_monitored_with_ni5761 \
                and not any(has_smelly_vals(file[hdf_path][key][:]) for key in file[hdf_path].keys())

    Gather(num_processes=1).from_files(hdf_dir.glob("data/Event*.hdf"))\
        .to_hdf_file(hdf_dir / "EventDataExtLinks.hdf")\
        .if_fulfills(ed_func_to_fulfill, on_error=True)\
        .run_with_external_links()


if __name__ == "__main__":
    transform(tdms_dir=Path("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/").expanduser(),
              hdf_dir=Path("~/output_files").expanduser())
