"""
Does an end to end test for the transformation.py located in src.
"""
from pathlib import Path
import os
from src.transformation import transform
from tests.utils.data_creator import xb2_like_event_data_creator
from tests.utils.data_creator import xb2_like_trend_data_creator
from tests.utils.dir_handler import remkdir

def test_transformation() -> None:
    """
    creates tdms files and hdf5 files that we want, applies the transformation and tests its output to the created
    hdf5 files
    """
    ### ARRANGE
    data_dir_path = remkdir(Path(__file__).parent / "data")

    created_tdms_dir = remkdir(data_dir_path / "created_tdms")
    created_hdf_dir = remkdir(data_dir_path / "created_hdf")

    # create tdms files (trend and event data) similar to xb2 data files
    xb2_like_event_data_creator.create_all(created_tdms_dir, created_hdf_dir)
    xb2_like_trend_data_creator.create_all(created_tdms_dir, created_hdf_dir)

    transform_hdf_dir = remkdir(data_dir_path / "transformed_hdf")

    ### ACT
    transform(created_tdms_dir, transform_hdf_dir)

    ## ASSERT
    for path in transform_hdf_dir.glob("data/*.hdf"):
        transformed_hdf_file_path = transform_hdf_dir / "data" / path.name
        should_hdf_file_path = created_hdf_dir / path.name
        is_equal = os.system(f"h5diff  {transformed_hdf_file_path} {should_hdf_file_path}") == 0
        assert is_equal


if __name__ == "__main__":
    test_transformation()
