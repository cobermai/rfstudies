from pathlib import Path
import numpy as np
import numpy.typing as npt
import h5py
import time


def log_type_translator(is_type: str) : #-> typing.Callable[[Path, str], bool]:
    log_type_dict = {"is_log": 0, "is_bdin40ms": 1, "is_bdin20ms": 2, "is_bd": 3}
    def test_is_type(file_path: Path, hdf_path: str) -> bool:
        """
        This function translates the 'Log Type' group properties of the event data into a boolean value.
        :param data: data.attrs the log label assigned in the group properties is in  {0,1,2,3}
        :return: True if (is_log -> 0, is_bdin40ms -> 1, is_bdin20ms -> 2, is_bd -> 3) in other cases return False
        """
        with h5py.File(file_path, "r") as file:
            print(hdf_path)
            print({key: val for key, val in file[hdf_path].attrs.items()})
            label = file[hdf_path].attrs["Log Type"]
            if label in log_type_dict.values():
                return label == log_type_dict[is_type]
            else:
                raise ValueError(f"'Log Type' label not valid no translation for {label} in {log_type_dict}!")

    return test_is_type

def get_timestamp(file_path: Path, hdf_path: str) -> np.datetime64:
    """
    returns the Timestamp from group propperties in numpy datetime format
    :param data: input data
    :return:
    """
    with h5py.File(file_path, "r", ) as file:
        datetime_str = file[hdf_path].attrs["Timestamp"][:-1]
        return np.datetime64(datetime_str).astype(h5py.opaque_dtype('M8[us]'))


def pulse_length(file_path: Path, hdf_path: str):
    """calculates the duration where the amplitude is higher than half of the maximal value."""
    with h5py.File(file_path, "r") as file:
        data: npt.ArrayLike = file[hdf_path][:]
        threshold = data.max() / 2
        acquisition_window: float = 2e-6
        return (data > threshold).sum() / data.shape[0] * acquisition_window