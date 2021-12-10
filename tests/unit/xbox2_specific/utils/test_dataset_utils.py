from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import xarray as xr

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
    dataset_utils.event_ext_link_hdf_to_da_timestamp = MagicMock(
        return_value=xr.DataArray(data=np.ones((10,)), dtype=xr.DataArray))
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


@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("bd_array, timestamp_array, threshold, output_expected",
                         [(np.array([1, 0, 0, 1], dtype=bool),
                           np.array([np.datetime64('2021-08-18T17:57:00'),
                                     np.datetime64('2021-08-18T17:57:10'),
                                     np.datetime64('2021-08-18T17:57:20'),
                                     np.datetime64('2021-08-18T17:57:30')]),
                           1,
                           np.array([0, 0, 0, 0], dtype=bool)),
                          (np.array([0, 0, 1, 1], dtype=bool),
                           np.array([np.datetime64('2021-08-18T17:57:00'),
                                     np.datetime64('2021-08-18T17:57:10'),
                                     np.datetime64('2021-08-18T17:57:20'),
                                     np.datetime64('2021-08-18T17:57:30')]),
                           1,
                           np.array([0, 0, 0, 1], dtype=bool)),
                          ]
                         )
def test__determine_followup(bd_array, timestamp_array, threshold, output_expected):
    # ARRANGE

    # ACT
    output = dataset_utils.determine_followup(bd_label=bd_array, timestamp=timestamp_array, threshold=threshold)

    # ASSERT
    assert all(output == output_expected)


@pytest.mark.parametrize("data, output_expected",
                         [(np.array([1, 2, np.NaN, 4, 5, np.NaN, 7, 8, 9]),
                           np.array([1, 2, 0, 4, 5, 0, 7, 8, 9])),
                          (np.array(["Hello_world", "this", "is", "a", "test"]),
                           np.array(["Hello_world", "this", "is", "a", "test"]))
                          ])
def test__da_to_numpy_for_ml(data, output_expected):
    # ARRANGE
    data_array = xr.DataArray(data=data)

    # ACT
    output = dataset_utils.da_to_numpy_for_ml(data_array=data_array)

    # ASSERT
    assert all(output == output_expected)


@pytest.mark.parametrize("signal, feature_name, expected_output",
                         [(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                           "PEI Amplitude",
                           np.array([4.36641900e+07, 1.75414380e+08, 3.95250570e+08, 7.03172760e+08, 1.09918095e+09,
                                     1.58327514e+09, 2.15545533e+09, 2.81572152e+09, 3.56407371e+09]
                                    )),
                          (np.array([1, -1, 2, -2, 0, 0]),
                           "PEI Amplitude",
                           np.array([4.3664190e+07, 4.4421810e+07, 1.7541438e+08, 1.7692962e+08, 0, 0])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                           "PSI Amplitude",
                           np.array([7.7264700e+07, 3.1169340e+08, 7.0328610e+08, 1.2520428e+09, 1.9579635e+09,
                                     2.8210482e+09, 3.8412969e+09, 5.0187096e+09, 6.3532863e+09]
                                    )),
                          (np.array([1, -1, 2, -2, 0, 0]),
                           "PSI Amplitude",
                           np.array([7.726470e+07, 7.989930e+07, 3.116934e+08, 3.169626e+08, 0, 0])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                           "PSR Amplitude",
                           np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]
                                    )),
                          (np.array([1, -1, 2, -2, 0, 0]),
                           "PSR Amplitude",
                           np.array([1, -1, 2, -2, 0, 0])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                           "PKI Amplitude",
                           np.array([5.0981700e+07, 2.0640740e+08, 4.6627710e+08, 8.3059080e+08, 1.2993485e+09,
                                     1.8725502e+09, 2.5501959e+09, 3.3322856e+09, 4.2188193e+09]
                                    ,)),
                          (np.array([1, -1, 2, -2, 0, 0]),
                           "PKI Amplitude",
                           np.array([5.098170e+07, 5.346230e+07, 2.064074e+08, 2.113686e+08, 0, 0])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                           "DC Up",
                           np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]
                                    )),
                          (np.array([1, -1, 2, -2, 0, 0]),
                           "DC Up",
                           np.array([1, -1, 2, -2, 0, 0])),
                          (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                           "DC Down",
                           np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])),
                          (np.array([1, -1, 2, -2, 0, 0]),
                           "DC Down",
                           np.array([1, -1, 2, -2, 0, 0])),
                          ])
def test__scale_signal(signal, feature_name, expected_output):
    # ARRANGE

    # ACT
    output = dataset_utils.scale_signal(signal=signal, feature_name=feature_name)
    print(output)
    # ASSERT
    assert np.allclose(output, expected_output)
