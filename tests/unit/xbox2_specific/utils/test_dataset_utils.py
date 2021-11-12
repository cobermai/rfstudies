from unittest.mock import MagicMock
from unittest.mock import patch
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from src.xbox2_specific.utils import dataset_utils


@pytest.mark.parametrize("dataset_expected",
                         [np.array(range(10)),
                          (True, False, True)]
                         )
def test__read_hdf_dataset(tmpdir, dataset_expected):
    """
    Test read_hdf_dataset() function
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    test_key = "test"
    with context_dummy as f:
        f.create_dataset(test_key, data=dataset_expected)

    # ACT
    context_dummy = h5py.File(path, 'r')
    with context_dummy as f:
        dataset_out = dataset_utils.read_hdf_dataset(f, test_key)

    # ASSERT
    assert (dataset_expected == dataset_out).all()


def test__read_hdf_dataset_error(tmpdir):
    """
    Test read_hdf_dataset() function errors
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    test_key = "test"
    with context_dummy as f:
        grp = f.create_group(test_key)
        grp.create_dataset(test_key, data="dummy_text")

    # ACT
    context_dummy = h5py.File(path, 'r')
    with pytest.raises(ValueError):
        with context_dummy as f:
            dataset_utils.read_hdf_dataset(f, test_key)


@pytest.mark.parametrize("dataset_in, selection",
                         [(np.array(range(10)), np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)),
                          (np.array(range(10)), np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)),
                          (np.array([True, False, True]), np.array([True, True, False]))]
                         )
def test__read_hdf_dataset_selection(tmpdir, dataset_in, selection):
    """
    Test read_hdf_dataset_selection() function
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    test_key = "test"
    with context_dummy as f:
        f.create_dataset(test_key, data=dataset_in)
    dataset_out_expected = dataset_in[selection]
    # ACT
    context_dummy = h5py.File(path, 'r')
    with context_dummy as f:
        dataset_out = dataset_utils.read_hdf_dataset_selection(f, test_key, selection)

    # ASSERT
    assert (dataset_out_expected == dataset_out).all()


def test__read_hdf_dataset_selection_error(tmpdir):
    """
    Test read_hdf_dataset_selection() function errors
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    test_key = "test"
    with context_dummy as f:
        grp = f.create_group(test_key)
        grp.create_dataset(test_key, data="dummy_text")

    # ACT
    context_dummy = h5py.File(path, 'r')
    with pytest.raises(ValueError):
        with context_dummy as f:
            dataset_utils.read_hdf_dataset_selection(f, test_key, [True])


def test__select_trend_data_events():
    """
    Test select_trend_data_events() function
    """
    # ARRANGE
    dummy_event_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06')])
    dummy_trend_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:01'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:03')])

    time_threshold = 2
    time_filter_expected = np.array([True, False, True, False])

    # ACT
    time_filter_out = dataset_utils.select_trend_data_events(dummy_event_timestamps,
                                                             dummy_trend_timestamps,
                                                             time_threshold)

    # ASSERT
    assert (time_filter_expected == time_filter_out).all()


def test__select_events_from_list(tmpdir):
    """
    Test select_events_from_list() function
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    dataset = np.ones((6,), dtype=bool)
    dummy_event_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06'),
                                       np.datetime64('2021-08-18T17:59:07'),
                                       np.datetime64('2021-08-18T17:59:08')])
    dummy_trend_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:01'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:03'),
                                       np.datetime64('2021-08-18T17:59:08'),
                                       np.datetime64('2021-08-18T17:59:09')])
    with context_dummy as f:
        f.create_dataset("Timestamp",
                         data=dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype)))
        f.create_dataset("PrevTrendData/Timestamp",
                         data=dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype)))
        f.create_dataset("clic_label/is_healthy", data=dataset)
        f.create_dataset("run_no", data=dataset)
        f.create_dataset("test1", data=dataset)
        f.create_dataset("test2", data=dataset)
        f.create_dataset("PSI Amplitude/pulse_amplitude", data=dataset)

    selection_list = ["test1", "test2"]

    selection_expected = np.array([False, True, False, True, False, False])

    # ACT
    np.random.seed(42)
    selection_out = dataset_utils.select_events_from_list(path, selection_list)

    # ASSERT
    assert (selection_out == selection_expected).all()

@pytest.mark.skip(reason="not finished")
def test_event_ext_link_hdf_to_da_timestamp(tmp_dir):
    dataset_utils.event_ext_link_hdf_to_da_timestamp = MagicMock(return_value=xr.DataArray(data=np.ones((10,)), dtype=xr.DataArray))
    print(dataset_utils.event_ext_link_hdf_to_da_timestamp(tmp_dir, np.array([1, 2, 3, 4]), ['dummy_feature']))


@pytest.mark.parametrize("input_array, shift_value, fill_value, output_expected",
                         [(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 2, 0, np.array([0, 0, 1, 2, 3, 4, 5, 6, 7])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), -2, 0, np.array([3, 4, 5, 6, 7, 8, 9, 0, 0])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, 0, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))]
                         )
def test__shift_values(input_array, shift_value, fill_value, output_expected):
    # ARRANGE

    # ACT
    output = dataset_utils.shift_values(array=input_array, num=shift_value, fill_value=fill_value)

    # ASSERT
    assert all(output == output_expected)
