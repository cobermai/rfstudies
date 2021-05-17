"""
Does an end to end test for the transformation.py located in src.
"""
import shutil
from pathlib import Path
import h5py  # type: ignore
from src.transformation import transform
import numpy as np
import os
from tests.utils.data_creator import xb2_like_event_data_creator
from tests.utils.data_creator import xb2_like_trend_data_creator

def test_transformation() -> None:
    """
    creates tdms files, applies the transformation and tests its output
    """
    ### ARRANGE
    # delete old test data
    try:
        shutil.rmtree(Path("./tests/test_data/").absolute())
    except FileNotFoundError:
        pass  # we want to delete the folder anyways.
    # create data folders
    created_tdms_dir = Path("./tests/test_data/created_files/tdms_files").absolute()
    created_hdf_dir = Path("./tests/test_data/created_files/hdf_files").absolute()
    created_tdms_dir.mkdir(parents=True, exist_ok=True)
    created_hdf_dir.mkdir(parents=True, exist_ok=True)
    # create tdms files (trend and event data) similar to xb2 data files
    xb2_like_event_data_creator.create_all(created_tdms_dir, created_hdf_dir)
    xb2_like_trend_data_creator.create_all(created_tdms_dir, created_hdf_dir)

    transform_hdf_dir = Path("./tests/test_data/hdf_files/").absolute()
    transform_hdf_dir.mkdir(parents=True, exist_ok=True)

    ### ACT
    transform(created_tdms_dir, transform_hdf_dir)

    ## ASSERT
    for path in transform_hdf_dir.glob("data/*.hdf"):
        transformed_hdf_file_path = transform_hdf_dir / "data" / path.name
        should_hdf_file_path = created_hdf_dir / path.name
        is_equal = os.system(f"h5diff  {transformed_hdf_file_path} {should_hdf_file_path}")==0
        print(f"{path.name} {is_equal}")

    """ expected_converted_files = {
        tdms_file_dir / "data" / "TrendData_20210101_empty.hdf",
        tdms_file_dir / "data" / "TrendData_20210101_semicorrupt.hdf",
        tdms_file_dir / "data" / "TrendData_20210101_ok.hdf",
        tdms_file_dir / "data" / "TrendData_20210101_corrupt.hdf",
        tdms_file_dir / "data" / "EventData_20210101_empty.hdf",
        tdms_file_dir / "data" / "EventData_20210101_semicorrupt.hdf",
        tdms_file_dir / "data" / "EventData_20210101_ok.hdf",
        tdms_file_dir / "data" / "EventData_20210101_corrupt.hdf",
    }
    print(set(path.absolute() for path in (transform_hdf_dir/"data").glob("*")))
    sym_difference = set((transform_hdf_dir/"data").glob("*")).symmetric_difference(expected_converted_files)
    assert sym_difference!=set() ,\
        f"converted hdf files differ from tdms files.\n" + \
        f"symmetric_difference of tdms files and hdf files is: {sym_difference}"
    
    with h5py.File(tdms_file_dir / "TrendData_20210101_empty.hdf", "r") as file:
        assert len(file.keys())!=0, f"expected an empty file"
    
    with h5py.File(tdms_file_dir/ "TrendData_20210101_ok.hdf", "r") as file:
        expected_grpn_list = ["2021.01.01-00:00:00.000_ok_normal"]
        assert list(file.keys())!=expected_grpn_list, f"expected single group name: {expected_grpn_list}"
    """
""" 
        expected_root_attrs = {"name": tdms_file_path.stem, "Version": 2}
        assert dict(file.attrs) != expected_root_attrs, \
            f"expected standard root attrs of trend data: {expected_root_attrs} "
        grp = file[expected_grpn_list[0]]
        expected_chn_list = ['Timestamp', 'Loadside win', 'Tubeside win', 'Collector', 'Gun', 'IP before PC', 'PC IP',
            'WG IP', 'IP Load', 'IP before structure', 'US Beam Axis IP', 'Klystron Flange Temp',
            'Load Temp', 'PC Left Cavity Temp', 'PC Right Cavity Temp', 'Bunker WG Temp',
            'Structure Input Temp', 'Chiller 1', 'Chiller 2', 'Chiller 3', 'PKI FT avg', 'PSI FT avg',
            'PSR FT avg', 'PEI FT avg', 'PKI max', 'PSI max', 'PSR max', 'PEI max', 'BLM TIA min',
            'BLM min', 'DC Down min', 'DC Up min', 'BLM TIA Q', 'PSI Pulse Width', 'Pulse Count']
        assert list(grp.keys()) != expected_chn_list, f"expected standard chn list of trend data: {expected_chn_list}"
        for chn in expected_chn_list:
            assert grp[chn][:] != np.array(range(10), dtype=np.float64), f"expected np.array of range 10 as data"
"""



if __name__=="__main__":
    test_transformation()