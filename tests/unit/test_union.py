from pathlib import Path
import h5py
import os
import numpy as np
import pandas as pd
from src import union


def test_merge(tmp_path):
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
    union.merge(src_file_path, dest_file_path)
    # ASSERT
    is_equal = os.system(f"h5diff {dest_file_path} {expected_file_path}")==0
    assert is_equal

def test_convert_iso8601_to_datetime__with_attrs(tmp_path):
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
    union.convert_iso8601_to_datetime(work_file_path)
    # ASSERT
    is_equal = os.system(f"h5diff {work_file_path} {expected_file_path}")==0
    assert is_equal


def test_convert_iso8601_to_datetime__without_attrs(tmp_path):
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
    union.convert_iso8601_to_datetime(work_file_path, also_convert_attrs=False)
    # ASSERT
    is_equal = os.system(f"h5diff {work_file_path} {expected_file_path}") == 0
    assert is_equal

def test_is_datetime():
    assert union.is_datetime("asdf")==False
    assert union.is_datetime(1)==False
    assert union.is_datetime("1")==False
    assert union.is_datetime(b"2021-01-01T00:00:00.000000Z")==True
    assert union.is_datetime("2021-01-01T00:00:00.000000Z")==True
