"""
Does an end to end test for the transformation.py located in src.
"""
import shutil
from pathlib import Path
import os
from src.transformation import transform
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
        assert is_equal


if __name__=="__main__":
    test_transformation()
