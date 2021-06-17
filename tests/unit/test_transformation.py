"""
tests the transformation module
"""
import os
import h5py
from src.transformation import transform
from tests.utils.data_creator.xb2_like_event_data_creator import create_event_data
from tests.utils.data_creator.xb2_like_trend_data_creator import create_trend_data


def test_transformation(tmp_path_factory) -> None:
    """
    creates tdms files and hdf5 files that we want, applies the transformation and tests its output to the created
    hdf5 files
    """
    # ARRANGE
    transform_hdf_dir = tmp_path_factory.mktemp("transformed_hdf")
    created_tdms_dir = tmp_path_factory.mktemp("created_tdms")
    created_hdf_dir = tmp_path_factory.mktemp("created_hdf")

    # create tdms files (trend and event data) similar to xb2 data files
    create_event_data(created_tdms_dir, created_hdf_dir)
    create_trend_data(created_tdms_dir, created_hdf_dir)

    # ACT
    transform(created_tdms_dir, transform_hdf_dir)

    # ASSERT
    # testing content of the transformed files
    for path in created_hdf_dir.glob("*.hdf"):
        path_of_output = transform_hdf_dir / "data" / path.name
        path_of_expected = created_hdf_dir / path.name
        is_equal = os.system(f"h5diff {path_of_output} {path_of_expected}") == 0
        assert is_equal, f"the transformed file {path_of_output.name} differs from the expected output"
    # testing the gathered files
    for file_name in [ "EventDataExtLinks.hdf", "TrendDataExtLinks.hdf"]:
        with h5py.File(transform_hdf_dir / file_name, "r") as file:
            for key in file.keys():
                assert "_semicorrupt" in key or "_ok" in key, \
                    "ExternalLinks with healthy data should contain _semicorrupt or _ok in their names"
                assert not "_corrupt" in key, \
                    "ExternalLinks with healthy data should not contain _corrupt in their names"
