import h5py
import numpy as np
import pytest
import pandas as pd
from src.xbox2_specific.datasets import simple_select


@pytest.mark.parametrize("y, \
                         y_one_hot_expected",
                         [(np.array(['good', 'bad', 'good']),
                           np.array([[0, 1], [1, 0], [0, 1]])),
                          (np.array(['bad', 'good', 'bad']),
                           np.array([[1, 0], [0, 1], [1, 0]])),
                          (np.zeros(1),
                           np.array([[1]]))
                          ])
def test__one_hot(y, y_one_hot_expected):
    """
    Test one_hot function of dataset_creator
    """
    # ARRANGE

    # ACT
    y_one_hot = simple_select.one_hot_encode(y=y)

    # ASSERT
    assert (y_one_hot == y_one_hot_expected).all()


def test__scale_data():
    """
    Test scale_data() function
    """
    # ARRANGE
    X = np.array([[[0, 0, 0], [1, 1, 1]]])
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = simple_select.scale_data(X)
    print(X_output)

    # ASSERT
    assert (X_output == X_expected).all()


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
        dataset_out = simple_select.read_hdf_dataset(f, test_key)

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
        grp.create_dataset(test_key, data="dummytext")

    # ACT
    context_dummy = h5py.File(path, 'r')
    with pytest.raises(ValueError):
        with context_dummy as f:
            simple_select.read_hdf_dataset(f, test_key)


def test__create_time_filter():
    """
    Test create_time_filter() function
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
    time_filter_out = simple_select.create_time_filter(dummy_event_timestamps,
                                                       dummy_trend_timestamps,
                                                       time_threshold)

    # ASSERT
    assert (time_filter_expected == time_filter_out).all()


@pytest.mark.parametrize("index, X_expected",
                         [(['col1', 'col2'], np.nan_to_num(np.array([[1., 3.], [2., 4.]])[..., np.newaxis])),
                          (['col2', 'col3'], np.nan_to_num(np.array([[3., 5.], [4., 6.]])[..., np.newaxis]))
                          ])
def test__load_X_data(index, X_expected):
    """
    Test load_X_data() function
    """
    # ARRANGE
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]}
    df = pd.DataFrame(data=d)

    # ACT
    X_out = simple_select.load_X_data(df, pd.Index(index))
    print(X_out)
    print(X_expected)

    # ASSERT
    assert (X_out == X_expected).all()


@pytest.mark.parametrize("dummy_features, selection_filter_expected",
                         [(np.array([True, True, True, True]), np.array([False, True, True, True])),
                          (np.array([False, False, False, False]), np.array([False, False, False, False]))
                          ])
def test__create_breakdown_selection_filter(tmpdir, dummy_features, selection_filter_expected):
    """
    Test create_breakdown_selection_filter() function
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    test_key = "test"
    dummy_event_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06')])
    dummy_trend_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:01'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:03')])
    dummy_is_healthy_labels = np.array([True, True, False, False])
    with context_dummy as f:
        f.create_dataset(test_key, data=dummy_features)
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)

    # ACT
    np.random.seed(42)
    selection_filter_out = simple_select.create_breakdown_selection_filter(path, ["test"])

    # ASSERT
    assert (selection_filter_expected == selection_filter_out).all()


def test__select_data(tmpdir):
    """
    Test select_data() function
    """
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    dummy_is_bd_in_40ms_labels = np.array([True, True, False, False])
    dummy_is_bd_in_20ms_labels = np.array([True, True, False, False])
    dummy_is_bd_labels = np.array([True, True, False, False])
    dummy_event_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06')])
    dummy_trend_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:01'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:03')])
    dummy_is_healthy_labels = np.array([True, True, False, False])
    with context_dummy as f:
        f.create_dataset("is_bd_in_40ms", data=dummy_is_bd_in_40ms_labels)
        f.create_dataset("is_bd_in_20ms", data=dummy_is_bd_in_20ms_labels)
        f.create_dataset("is_bd", data=dummy_is_bd_labels)
        f.create_dataset("test_data1", data=4 * np.ones((4,)))
        f.create_dataset("test_data2", data=np.zeros((4,)))
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)
        f.create_dataset("is_healthy", data=dummy_is_healthy_labels)

    X_expected = np.array([[[-0.39223227],
                            [1.37281295],
                            [-0.98058068]]
                           ])
    y_expected = np.array([[1.]])

    # ACT
    np.random.seed(42)
    X, y = simple_select.select_data(path)

    # ASSERT
    assert np.allclose(X, X_expected)
    assert np.allclose(y, y_expected)
