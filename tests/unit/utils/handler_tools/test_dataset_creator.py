import numpy as np
import pytest
import h5py
from src.utils.handler_tools import dataset_creator


def test__train_valid_test_split():
    """
    Test train_valid_test_split() function
    """
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))
    splits_expected = 0.7, 0.1, 0.2

    # ACT
    train, valid, test = dataset_creator.train_valid_test_split(X=X, y=y, splits=splits_expected)

    # ASSERT
    assert set(X) == set(train.X).union(valid.X).union(test.X)
    assert set(y) == set(train.y).union(valid.y).union(test.y)
    length = len(X)
    assert set(range(length)) == set(train.idx).union(valid.idx).union(test.idx)
    splits_output = len(train.X) / length, len(valid.X) / length, len(test.X) / length
    assert splits_output == splits_expected


def test__train_valid_test_split_errors():
    # ARRANGE
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))

    # ACT
    with pytest.raises(ValueError):
        dataset_creator.train_valid_test_split(X=X, y=y, splits=(1, 0, 0))


def test__load_dataset(tmpdir):
    """
    Test load_dataset() function
    """
    # ARRANGE
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
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)
        f.create_dataset("is_healthy", data=dummy_is_healthy_labels)
    dataset_name = "simple_select"

    splits_expected = (0.7, 0.2, 0.1)

    # ACT
    np.random.seed(42)
    train, valid, test = dataset_creator.load_dataset(data_path=tmpdir, dataset_name=dataset_name)
    sum_elements = len(train.idx) + len(valid.idx) + len(test.idx)
    splits = (len(train.idx)/sum_elements, len(valid.idx)/sum_elements, len(test.idx)/sum_elements)

    # ASSERT
    assert splits == splits_expected


def test__load_dataset_error(tmp_path):
    # ARRANGE
    dataset_name = "nonexistent_dataset"

    # ACT
    with pytest.raises(AssertionError):
        dataset_creator.load_dataset(data_path=tmp_path, dataset_name=dataset_name)
