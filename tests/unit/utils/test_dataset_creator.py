from unittest.mock import patch
import h5py
import numpy as np
import pytest
from src.utils import dataset_creator


def test__train_valid_test_split():
    """
    Test train_valid_test_split() function
    """
    # ARRANGE
    p = patch.multiple(dataset_creator.DatasetCreator, __abstractmethods__=set())
    p.start()
    creator = dataset_creator.DatasetCreator()
    p.stop()
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))
    splits_expected = 0.7, 0.1, 0.2

    # ACT
    train, valid, test = creator.train_valid_test_split(X=X, y=y, splits=splits_expected)

    # ASSERT
    assert set(X) == set(train.X).union(valid.X).union(test.X)
    assert set(y) == set(train.y).union(valid.y).union(test.y)
    length = len(X)
    assert set(range(length)) == set(train.idx).union(valid.idx).union(test.idx)
    splits_output = len(train.X) / length, len(valid.X) / length, len(test.X) / length
    assert splits_output == splits_expected


def test__train_valid_test_split_errors():
    # ARRANGE
    p = patch.multiple(dataset_creator.DatasetCreator, __abstractmethods__=set())
    p.start()
    creator = dataset_creator.DatasetCreator()
    p.stop()
    X = np.array(range(10, 20))
    y = np.array(range(20, 30))

    # ACT
    with pytest.raises(ValueError):
        creator.train_valid_test_split(X=X, y=y, splits=(1, 0, 0))


@pytest.mark.skip(reason='not finished')
def test__load_dataset(tmpdir, mock_select_events):
    """
    Test load_dataset() function
    """
    # ARRANGE
    p = patch.multiple(dataset_creator.DatasetCreator, __abstractmethods__=set())
    p.start()
    creator = dataset_creator.DatasetCreator()
    p.stop()
    mock_select_events.return_value = np.array([True, False, True, False])
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

    splits_expected = (0.7, 0.2, 0.1)

    # ACT
    np.random.seed(42)

    train, valid, test = dataset_creator.load_dataset(creator=creator, hdf_dir=tmpdir)
    sum_elements = len(train.idx) + len(valid.idx) + len(test.idx)
    splits = (len(train.idx)/sum_elements, len(valid.idx)/sum_elements, len(test.idx)/sum_elements)

    # ASSERT
    assert splits == splits_expected
