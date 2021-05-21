"""creating XBox2 like trend data for testing"""
from pathlib import Path
import numpy as np
from tests.utils.data_creator.file_creator_for_testing import CreatorTestFiles


def _trend_data_creator(tdms_file_path: Path, hdf_file_path: Path) -> CreatorTestFiles:
    tdms_creator = CreatorTestFiles(hdf_file_path=hdf_file_path,
                                  tdms_file_path=tdms_file_path,
                                  root_prop_dict={"name": tdms_file_path.stem, "Version": 2})
    chn_list = ['Timestamp', 'Loadside win', 'Tubeside win', 'Collector', 'Gun', 'IP before PC', 'PC IP',
        'WG IP', 'IP Load', 'IP before structure', 'US Beam Axis IP', 'Klystron Flange Temp',
        'Load Temp', 'PC Left Cavity Temp', 'PC Right Cavity Temp', 'Bunker WG Temp',
        'Structure Input Temp', 'Chiller 1', 'Chiller 2', 'Chiller 3', 'PKI FT avg', 'PSI FT avg',
        'PSR FT avg', 'PEI FT avg', 'PKI max', 'PSI max', 'PSR max', 'PEI max', 'BLM TIA min',
        'BLM min', 'DC Down min', 'DC Up min', 'BLM TIA Q', 'PSI Pulse Width', 'Pulse Count']
    tdms_creator.grp_prop_dict = {}
    tdms_creator.ch_prop_dict = {chn: {} for chn in chn_list}
    tdms_creator.ch_data_dict = {chn: np.array(range(10), dtype=np.float64) for chn in chn_list}
    return tdms_creator

def _create_empty(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    file_name = "TrendData_20210101_empty"
    _trend_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

def _create_ok_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    file_name = "TrendData_20210101_ok"
    tdms_creator = _trend_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

    tdms_creator.add_artificial_group("2021.01.01-00:00:00.000_ok_normal")


def _create_semi_corrupt_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    """creates a tmds file with the specified path that is similar to trend data. And tests the functionality of
    the transformation part of the continuous integration."""
    file_name = "TrendData_20210101_semicorrupt"
    tdms_creator = _trend_data_creator((created_tdms_files_dir / file_name).with_suffix(".tdms"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

    chn_to_alter = list(tdms_creator.ch_prop_dict.keys())[6]
    tdms_creator.ch_data_dict[chn_to_alter][2] = np.NaN
    tdms_creator.add_artificial_group("2021.01.01-00:00:00.000_semicorrupt_NaNval")

    tdms_creator.ch_data_dict.update({chn_to_alter: np.array(range(2), dtype=np.float64)})
    tdms_creator.add_artificial_group("2021.01.01-00:00:00.000_corrupt_len")


def _create_corrupt_data(created_tdms_files_dir: Path, created_hdf_files_dir: Path) -> None:
    """creates a tmds file with the specified path that is similar to trend data. And tests the functionality of
    the transformation part of the continuous integration."""
    file_name = "TrendData_20210101_corrupt"
    tdms_creator = _trend_data_creator((created_tdms_files_dir / file_name).with_suffix(".hdf"),
                                     (created_hdf_files_dir / file_name).with_suffix(".hdf"))

    chn_list_altered = ["d", "e", "f"]
    tdms_creator.ch_prop_dict = {chn:{} for chn in chn_list_altered}
    tdms_creator.ch_data_dict = {chn: np.array(range(10), dtype=np.float64) for chn in chn_list_altered}
    tdms_creator.add_artificial_group("2021.01.01-00:00:00.000_corrupt_chn")

def create_all(created_tdms_files_dir: Path, created_hdf_files_dir:Path) -> None:
    """
    runs all the trend data creators and creates tdms and hdf files for testing in the specified directories
    :param created_tdms_files_dir: the destination directory for the tdms files
    :param created_hdf_files_dir: the destination directory for the hdf files
    """
    _create_empty(created_tdms_files_dir, created_hdf_files_dir)
    _create_semi_corrupt_data(created_tdms_files_dir, created_hdf_files_dir)
    _create_ok_data(created_tdms_files_dir, created_hdf_files_dir)
    _create_corrupt_data(created_tdms_files_dir, created_hdf_files_dir)
