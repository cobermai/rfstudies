"""this module tests the functions from hdf_tools"""
import os
import h5py
import numpy as np
import pandas as pd
from src.utils import hdf_tools


def test__hdf_path_combine() -> None:
    """tests hdf_path_combine function"""
    # ARRANGE
    arg_list = (("a", "b", "c"),
                ("/a/b", "/c"),
                ("/a///b///", "/c"))
    expected_output = "/a/b/c"
    # ACT + ASSERT
    for args in arg_list:
        output = hdf_tools.hdf_path_combine(*args)
        assert output == expected_output, f"expected {expected_output}\nbut got {output}"


def test_merge(tmp_path):
    """tests merge"""
    # ARRANGE
    src_file_path = tmp_path / "src.h5"
    dest_file_path = tmp_path / "dest.h5"
    expected_file_path = tmp_path / "expected.h5"
    with h5py.File(src_file_path, "w") as file:
        for l in [1, 2]:
            for k in [1, 2]:
                file.create_dataset(f"group{l}/dataset{k}", data=[l, k])
    with h5py.File(expected_file_path, "w") as file:
        for k in [1, 2]:
            file.create_dataset(f"dataset{k}", data=[1, k, 2, k])

    # ACT
    hdf_tools.merge(src_file_path, dest_file_path)

    # ASSERT
    is_equal = os.system(f"h5diff {dest_file_path} {expected_file_path}") == 0
    assert is_equal


def test_convert_iso8601_to_datetime__with_attrs(tmp_path):
    """tests conversion of iso datetime strings to datetime format. It tests conversion of datasets and attributes."""
    # ARRANGE
    work_file_path = tmp_path / "test.h5"
    expected_file_path = tmp_path / "expected.h5"

    attr_data = np.array([b"2021-01-01T00:00:00.123456789Z"])
    data = np.array([b"2021-01-01T00:00:00.111222333Z", b"2021-01-01T00:00:00.444555666Z"])
    with h5py.File(work_file_path, "w") as file:
        file.attrs.create("at1", data=attr_data)
        file.create_dataset("ds1", data=data)
        grp = file.create_group("test")
        grp.attrs.create("at2", data=attr_data)
        grp.create_dataset("ds2", data=data)

    attr_data_converted = pd.to_datetime(attr_data.astype(str)).to_numpy(np.datetime64)
    attr_data_converted = attr_data_converted.astype(h5py.opaque_dtype(attr_data_converted.dtype))
    data_converted = pd.to_datetime(data.astype(str)).to_numpy(np.datetime64)
    data_converted = data_converted.astype(h5py.opaque_dtype(data_converted.dtype))
    with h5py.File(expected_file_path, "w") as file:
        file.attrs.create("at1", data=attr_data_converted)
        file.create_dataset("ds1", data=data_converted)
        grp = file.create_group("test")
        grp.attrs.create("at2", data=attr_data_converted)
        grp.create_dataset("ds2", data=data_converted)

    # ACT
    hdf_tools.convert_iso8601_to_datetime(work_file_path)

    # ASSERT
    is_equal = os.system(f"h5diff {work_file_path} {expected_file_path}") == 0
    assert is_equal


def test_convert_iso8601_to_datetime__without_attrs(tmp_path):
    """tests conversion of iso strings to datetime without converting attributes, so only hdf-datsets."""
    work_file_path = tmp_path / "test.h5"
    expected_file_path = tmp_path / "expected.h5"

    attr_data = np.array([b"2021-01-01T00:00:00.123456789Z"])
    data = np.array([b"2021-01-01T00:00:00.111222333Z", b"2021-01-01T00:00:00.444555666Z"])
    with h5py.File(work_file_path, "w") as file:
        file.attrs.create("at1", data=attr_data)
        file.create_dataset("ds1", data=data)
        grp = file.create_group("test")
        grp.attrs.create("at2", data=attr_data)
        grp.create_dataset("ds2", data=data)

    data_converted = pd.to_datetime(data.astype(str)).to_numpy(np.datetime64)
    data_converted = data_converted.astype(h5py.opaque_dtype(data_converted.dtype))
    with h5py.File(expected_file_path, "w") as file:
        file.attrs.create("at1", data=attr_data)
        file.create_dataset("ds1", data=data_converted)
        grp = file.create_group("test")
        grp.attrs.create("at2", data=attr_data)
        grp.create_dataset("ds2", data=data_converted)

    # ACT
    hdf_tools.convert_iso8601_to_datetime(work_file_path, also_convert_attrs=False)

    # ASSERT
    is_equal = os.system(f"h5diff {work_file_path} {expected_file_path}") == 0
    assert is_equal


def test_clean_row_by_row(tmp_path):
    """tests clean row by row"""
    # ARRANGE
    work_file_path = tmp_path / "test.h5"
    expected_file_path = tmp_path / "expected.h5"

    with h5py.File(work_file_path, "w") as file:
        data = np.array([0, 1, 2, 3, "NaT"], dtype="datetime64[us]")
        file.create_dataset(name="d0", data=data.astype(h5py.opaque_dtype(data.dtype)), chunks=True)
        file.create_dataset(name="d1", data=[0, np.nan, 2, 3, 4], chunks=True)
        file.create_dataset(name="d2", data=[0., 1., 2., np.nan, 4.], chunks=True)
    with h5py.File(expected_file_path, "w") as file:
        data = np.array([0, 2], dtype="datetime64[us]")
        file.create_dataset(name=f"d0", data=data.astype(h5py.opaque_dtype(data.dtype)))
        file.create_dataset(name=f"d1", data=[0, 2])
        file.create_dataset(name=f"d2", data=[0., 2.])

    # ACT
    hdf_tools.clean_by_row(work_file_path)

    # ASSERT
    is_equal = os.system(f"h5diff {work_file_path} {expected_file_path}") == 0
    assert is_equal


def test_sort_by(tmp_path):
    """tests sort_by"""
    # ARRANGE
    work_file_path = tmp_path / "test.h5"

    with h5py.File(work_file_path, "w") as file:
        file.create_dataset(name="d0", data=[0, 1, 2, 3, 4], chunks=True)
        file.create_dataset(name="d1", data=[4, 3, 2, 1, 0], chunks=True)

    # ACT
    hdf_tools.sort_by(work_file_path, "d1")

    # ASSERT
    with h5py.File(work_file_path, "r") as file:
        assert np.all(np.diff(file["d1"][:]) >= 0)
