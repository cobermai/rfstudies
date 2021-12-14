from collections import namedtuple
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import xarray as xr

import src.dataset_creator


@patch.multiple(dataset_creator.DatasetCreator, __abstractmethods__=set())
def test__load_dataset(tmpdir):
    """
    Test load_dataset() function
    """
    # ARRANGE
    creator = dataset_creator.DatasetCreator()

    # ACT

    # ASSERT
    assert hasattr(creator, "select_events")
    assert hasattr(creator, "select_features")
    assert hasattr(creator, "select_labels")
    assert hasattr(creator, "train_valid_test_split")
    assert hasattr(creator, "scale_data")
    assert hasattr(creator, "one_hot_encode")


@pytest.mark.skip(reason="not finished")
@patch.multiple(dataset_creator.DatasetCreator, __abstractmethods__=set(),
                select_events=MagicMock(return_value=np.ones((10,), dtype=bool)),
                select_features=MagicMock(return_value=np.ones((10, 10, 10), dtype=bool)),
                select_labels=MagicMock(return_value=np.ones((10,), dtype=bool)))
def test__load_dataset(tmpdir):
    """
    Test load_dataset() function
    """
    # ARRANGE
    creator = dataset_creator.DatasetCreator()
    path = tmpdir.join("context.hdf")
    context_dummy = h5py.File(path, 'w')
    dummy_is_bd_in_40ms_labels = np.ones((10,), dtype=bool)
    dummy_is_bd_in_20ms_labels = np.ones((10,), dtype=bool)
    dummy_is_bd_labels = np.ones((10,), dtype=bool)
    dummy_event_timestamps = np.array([np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09'),
                                       np.datetime64('2021-08-18T17:59:09')
                                       ])
    dummy_trend_timestamps = np.array([np.datetime64('2021-08-18T17:59:03'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06'),
                                       np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06'),
                                       np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04')
                                       ])
    dummy_is_healthy_labels = np.ones((10,), dtype=bool)
    with context_dummy as f:
        f.create_dataset("is_bd_in_40ms", data=dummy_is_bd_in_40ms_labels)
        f.create_dataset("is_bd_in_20ms", data=dummy_is_bd_in_20ms_labels)
        f.create_dataset("is_bd", data=dummy_is_bd_labels)
        f.create_dataset("test_data1", data=4 * np.ones((10,)))
        f.create_dataset("test_data2", data=np.zeros((10,)))
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(
            h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)
        f.create_dataset("is_healthy", data=dummy_is_healthy_labels)

    splits_expected = (0.7, 0.2, 0.1)

    # ACT
    np.random.seed(42)

    train, valid, test = dataset_creator.load_dataset(creator=creator, data_path=tmpdir / "context.hdf")
    sum_elements = len(train.idx) + len(valid.idx) + len(test.idx)
    splits = (len(train.idx) / sum_elements, len(valid.idx) / sum_elements, len(test.idx) / sum_elements)

    # ASSERT
    assert splits == splits_expected


def test_data_array_to_numpy():
    """
    Function that tests test_da_to_numpy_for_ml
    """
    # ARRANGE
    data = namedtuple("data", ["X", "y", "idx"])
    data_expected = np.ones(4)
    X = xr.DataArray(data=data_expected)
    y = xr.DataArray(data=data_expected)
    idx = xr.DataArray(data=data_expected)

    train, valid, test = data(X, y, idx), data(X, y, idx), data(X, y, idx)
    # ACT
    train_out, valid_out, test_out = dataset_creator.data_array_to_numpy(train, valid, test)

    # ASSERT
    assert (train_out.X == data_expected).all()
    assert (train_out.y == data_expected).all()
    assert (train_out.idx == data_expected).all()
    assert (valid_out.X == data_expected).all()
    assert (valid_out.y == data_expected).all()
    assert (valid_out.idx == data_expected).all()
    assert (test_out.X == data_expected).all()
    assert (test_out.y == data_expected).all()
    assert (test_out.idx == data_expected).all()
