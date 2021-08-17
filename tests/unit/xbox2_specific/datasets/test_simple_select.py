from pathlib import Path
import h5py
import numpy as np
import pytest
import pandas as pd
from datetime import datetime
from src.utils.handler_tools.context_data_writer import ColumnWiseContextDataWriter
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


def test__create_time_filter():
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
    # ARRANGE
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]}
    df = pd.DataFrame(data=d)
    # ACT
    X_out = simple_select.load_X_data(df, pd.Index(index))
    print(X_out)
    print(X_expected)
    # ASSERT
    assert (X_out == X_expected).all()


@pytest.mark.skip(reason="Test not finished")
def test__create_breakdown_selection_filter(tmpdir):
    # ARRANGE
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
    test_key = "test"
    dummy_features = np.array([True, True, False, False])
    dummy_event_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:04'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:06')])
    dummy_trend_timestamps = np.array([np.datetime64('2021-08-18T17:59:00'),
                                       np.datetime64('2021-08-18T17:59:01'),
                                       np.datetime64('2021-08-18T17:59:02'),
                                       np.datetime64('2021-08-18T17:59:03')])
    dummy_healthy_labels = np.array([True, True, False, False])
    with context_dummy as f:
        f.create_dataset(test_key, data=dummy_features)
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_healthy_labels)
    selection_filter_expected = np.array([True, False, False, False])
    # ACT
    selection_filter_out = simple_select.create_breakdown_selection_filter(path, ["test"])
    # ASSERT
    print(selection_filter_out)
    print(selection_filter_expected)
    assert (selection_filter_expected == selection_filter_out).all()


@pytest.mark.skip(reason="Test not finished")
def test__select_data(tmp_path):
    # ARRANGE
    context = h5py.File('context.hdf', 'w')
    with context as f:
        f.create_dataset("is_bd_in_40ms", (1, 0, 1))
        f.create_dataset("is_bd_in_20ms", (0, 1, 0))
        f.create_dataset("is_bd", (0, 0, 1))
        f.create_dataset("Timestamp", (0.01, 0.03, 0.05))
        f.create_dataset("PrevTrendData/Timestamp", (0.02, 0.04, 0.06))
        f.create_dataset("clic_label/is_healthy", (1, 0, 0))

    dest_file_path = Path(tmp_path / 'context_file')
    column_wise_handler = ColumnWiseContextDataWriter(dest_file_path, length=10)
    for feature in features:
        column_wise_handler.write_column(feature)

    # ACT
    X, y = simple_select.select_data(tmp_path)

    # ASSERT
    assert X == X_expected
    assert y == y_expected
