from tests.utils.data_creator.test_files_creator import MakeTestFiles
from pathlib import Path
import numpy as np
import nptdms  # type: ignore


def _trend_data_creator(tdms_file_path: Path):
    tdms_maker = MakeTestFiles(hdf_file_path=tdms_file_path.with_suffix(".hdf"),
                               tdms_file_path=tdms_file_path,
                               root_prop_dict={"name": tdms_file_path.stem, "Version": 2})
    chn_list = ['Timestamp', 'Loadside win', 'Tubeside win', 'Collector', 'Gun', 'IP before PC', 'PC IP',
        'WG IP', 'IP Load', 'IP before structure', 'US Beam Axis IP', 'Klystron Flange Temp',
        'Load Temp', 'PC Left Cavity Temp', 'PC Right Cavity Temp', 'Bunker WG Temp',
        'Structure Input Temp', 'Chiller 1', 'Chiller 2', 'Chiller 3', 'PKI FT avg', 'PSI FT avg',
        'PSR FT avg', 'PEI FT avg', 'PKI max', 'PSI max', 'PSR max', 'PEI max', 'BLM TIA min',
        'BLM min', 'DC Down min', 'DC Up min', 'BLM TIA Q', 'PSI Pulse Width', 'Pulse Count']
    tdms_maker.grp_prop_dict = {}
    tdms_maker.ch_prop_dict = {chn: {} for chn in chn_list}
    tdms_maker.ch_data_dict = {chn: np.array(range(10), dtype=np.float64) for chn in chn_list}
    return tdms_maker

def _create_empty(tdms_file_path: Path):
    tdms_maker = _trend_data_creator(tdms_file_path)

def _create_ok_data(tdms_file_path):
    tdms_maker = _trend_data_creator(tdms_file_path)

    tdms_maker.add_artificial_group("2021.01.01-00:00:00.000_ok_normal")


def _create_semi_corrupt_data(tdms_file_path):
    """creates a tmds file with the specified path that is similar to trend data. And tests the functionality of
    the transformation part of the continuous integration."""
    tdms_maker = _trend_data_creator(tdms_file_path)

    chn_to_alter = list(tdms_maker.ch_prop_dict.keys())[6]
    tdms_maker.ch_data_dict[chn_to_alter][2] = np.NaN
    tdms_maker.add_artificial_group("2021.01.01-00:00:00.000_semicorrupt_NaNval")

    tdms_maker.ch_data_dict.update({chn_to_alter: np.array(range(2), dtype=np.float64)})
    tdms_maker.add_artificial_group("2021.01.01-00:00:00.000_corrupt_len")

def _create_corrupt_data(tdms_file_path):
    """creates a tmds file with the specified path that is similar to trend data. And tests the functionality of
    the transformation part of the continuous integration."""
    tdms_maker = _trend_data_creator(tdms_file_path)

    chn_list_altered = ["d", "e", "f"]
    tdms_maker.ch_prop_dict = {chn:{} for chn in chn_list_altered}
    tdms_maker.ch_data_dict = {chn: np.array(range(10), dtype=np.float64) for chn in chn_list_altered}
    tdms_maker.add_artificial_group("2021.01.01-00:00:00.000_corrupt_chn")

def create_all(tdms_dir: Path):
    _create_empty(              tdms_dir / "TrendData_20210101_empty.tdms")
    _create_semi_corrupt_data(  tdms_dir / "TrendData_20210101_semicorrupt.tdms")
    _create_ok_data(            tdms_dir / "TrendData_20210101_ok.tdms")
    _create_corrupt_data(       tdms_dir / "TrendData_20210101_corrupt.tdms")