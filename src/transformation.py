"""
This module provides tools to transform data to a capable format so that further analyzing can be done easily.
"""
from pathlib import Path
import logging
import h5py
import numpy as np
from src.utils.transf_tools.convert import Convert
from src.utils.transf_tools.gather import gather
from src.utils.system.setup_logging import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)


def transform(tdms_dir: Path, hdf_dir: Path) -> None:
    """
    transforms all tdms files to hdf files, filters faulty data and gathers hdf groups with depth 1 of the hdf files
    into one hdf file with external links.
    :param tdms_dir: input directory with tdms files
    :param hdf_dir: output directory with hdf files
    """
    Path(hdf_dir, "data").mkdir(parents=False, exist_ok=True)

    # read tdms files, convert them to hdf5 and write them into hdf_dir/data/
    Convert(check_already_converted=True, num_processes=4)\
        .from_tdms(tdms_dir)\
        .to_hdf(hdf_dir / "data").run()

    # Combining all Events and TrendData sets into one hdf5 file with external links if they are not faulty
    def td_func_to_fulfill(file_path: Path, hdf_path: str) -> bool:
        with h5py.File(file_path, "r") as file:
            grp = file[hdf_path]
            ch_shapes = [grp[key].shape[0] for key in grp.keys()]
            len_equal = all(ch_shape == ch_shapes[0] for ch_shape in ch_shapes)
            num_of_samples = 35
            return len_equal and grp.len == num_of_samples

    gather(src_file_paths=hdf_dir.glob("data/Trend*.hdf"),
           dest_file_path=hdf_dir / "TrendDataExtLinks.hdf",
           if_fulfills=td_func_to_fulfill,
           on_error=False,
           num_processes=4)

    def ed_func_to_fulfill(file_path: Path, hdf_path: str) -> bool:
        with h5py.File(file_path, "r") as file:
            grp = file[hdf_path]
            ch_len = [ch.shape[0] for ch in grp.items()]

            acquisition_window = 2e-6  # time period of one event is 2 microseconds

            # acquisition card NI-5772 see https://www.ni.com/en-us/support/model.ni-5772.html
            sampling_frequency_ni5772 = 1.6e8  # sampling frequency of the acquisition card
            num_of_values_ni5772 = acquisition_window * sampling_frequency_ni5772
            number_of_signals_monitored_with_ni5772 = 8

            # acquisition card NI-5761 see https://www.ni.com/en-us/support/model.ni-5761.html
            sampling_frequency_ni5761 = 1.6e9  # sampling frequency of the acquisition card
            num_of_values_ni5761 = acquisition_window * sampling_frequency_ni5761
            number_of_signals_monitored_with_ni5761 = 8

            def has_smelly_values(data) -> bool:
                return any(np.isnan(data) | np.isinf(data))

            return grp.attrs.get("Timestamp", None) is not None \
                and ch_len.count(num_of_values_ni5772) == number_of_signals_monitored_with_ni5772 \
                and ch_len.count(num_of_values_ni5761) == number_of_signals_monitored_with_ni5761 \
                and not any(has_smelly_values(ch[:]) for ch in grp.items())

    gather(src_file_paths=hdf_dir.glob("data/Event*.hdf"),
           dest_file_path=hdf_dir / "EventDataExtLinks.hdf",
           if_fulfills=ed_func_to_fulfill,
           on_error=False,
           num_processes=1)


if __name__ == "__main__":
    transform(tdms_dir=Path("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/").expanduser(),
              hdf_dir=Path("~/output_files").expanduser())
