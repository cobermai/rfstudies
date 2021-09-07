import h5py
import numpy as np
import pandas as pd
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

#
# @pytest.mark.parametrize("dataset, selection_expected",
#                          ([np.ones((10,), dtype=bool), np.ones((10,), dtype=bool)])
#                          )
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
        f.create_dataset("test1", data=dataset)
        f.create_dataset("test2", data=dataset)

    selection_list = ["test1", "test2"]

    selection_expected = np.array([False, True, False, True, False, False])

    # ACT
    np.random.seed(42)
    selection_out = dataset_utils.select_events_from_list(path, selection_list)
    print(selection_out)

    # ASSERT
    assert (selection_out == selection_expected).all()


def test__select_features_from_list():
    """
    Test select_features_from_list()
    """
    # ARRANGE
    X = np.ones((10, ))
    X_expected = X[..., np.newaxis]
    X_expected = np.nan_to_num(X_expected)

    df = pd.DataFrame({
        "data": X,
    })

    selection_list = ["data"]

    # ACT
    X_out = dataset_utils.select_features_from_list(df, selection_list)

    # ASSERT
    assert (X_expected == X_out).all
