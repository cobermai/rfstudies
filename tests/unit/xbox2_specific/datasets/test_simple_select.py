import h5py
import numpy as np
import pandas as pd
import pytest
from src.utils import dataset_creator
from src.xbox2_specific.datasets import simple_select
from src.utils.hdf_tools import hdf_to_df_selection


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
    selector = simple_select.SimpleSelect()

    # ACT
    y_one_hot = selector.one_hot_encode(y=y)

    # ASSERT
    assert (y_one_hot == y_one_hot_expected).all()


def test__scale_data():
    """
    Test scale_data() function
    """
    # ARRANGE
    selector = simple_select.SimpleSelect()
    X = np.array([[[0, 0, 0], [1, 1, 1]]])
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = selector.scale_data(X)
    print(X_output)

    # ASSERT
    assert (X_output == X_expected).all()


def test__select_events(tmpdir):
    """
    Test create_select_events() function
    """
    # ARRANGE
    selector = simple_select.SimpleSelect()
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
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
    dummy_features = np.array([True, True, True, True, True, True])
    dummy_is_healthy_labels = np.array([True, True, True, False, False, False])
    with context_dummy as f:
        f.create_dataset("is_bd_in_40ms", data=dummy_features)
        f.create_dataset("is_bd_in_20ms", data=dummy_features)
        f.create_dataset("is_bd", data=dummy_features)
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)

    df_expected = pd.DataFrame({
        "PrevTrendData__Timestamp": np.array([np.datetime64('2021-08-18T17:59:01'),
                                              np.datetime64('2021-08-18T17:59:03'),
                                              np.datetime64('2021-08-18T17:59:08'),
                                              np.datetime64('2021-08-18T17:59:09')
                                              ]),
        "Timestamp": np.array([np.datetime64('2021-08-18T17:59:04'),
                               np.datetime64('2021-08-18T17:59:06'),
                               np.datetime64('2021-08-18T17:59:07'),
                               np.datetime64('2021-08-18T17:59:08')]),
        "clic_label__is_healthy": np.array([True, False, False, False]),
        "is_bd": np.ones((4,), dtype=bool),
        "is_bd_in_20ms": np.ones((4,), dtype=bool),
        "is_bd_in_40ms": np.ones((4,), dtype=bool)
    })
    # ACT
    np.random.seed(42)
    df_out = selector.select_events(path)
    print("expected")
    print(df_expected.head)
    print("out")
    print(df_out.head)

    # ASSERT
    pd.testing.assert_frame_equal(df_expected, df_out)


@pytest.mark.parametrize("data1, data2",
                         [([10, 20], [30, 40]),
                          ([-1., 3.], [20., 1.234])
                          ])
def test__select_features(data1, data2):
    """
    Test select_features() function
    """
    # ARRANGE
    selector = simple_select.SimpleSelect()
    d = {'Timestamp': [1, 2],
         'PrevTrendData__Timestamp': [3, 4],
         'is_bd': [5, 6],
         'is_healthy': [7, 8],
         'is_bd_in_20ms': [9, 10],
         'is_bd_in_40ms': [11, 12],
         'col1': data1,
         'col2': data2}
    df = pd.DataFrame(data=d)
    X_expected = df[pd.Index(['col1', 'col2'])].to_numpy(dtype=float)
    X_expected = np.nan_to_num(X_expected[..., np.newaxis])

    # ACT
    X_out = selector.select_features(df)

    # ASSERT
    assert (X_out == X_expected).all()


@pytest.mark.parametrize("data",
                         [np.ones((10,), dtype=bool),
                          np.zeros((10,), dtype=bool)
                          ])
def test__select_labels(data):
    """
    Test load_X_data() function
    """
    # ARRANGE
    selector = simple_select.SimpleSelect()
    d = {'is_healthy': data}
    df = pd.DataFrame(data=d)
    y_expected = df['is_healthy'].to_numpy(dtype=float)

    # ACT
    y_out = selector.select_labels(df)

    # ASSERT
    assert (y_out == y_expected).all()


def test__load_dataset(tmpdir):
    """
    Test load_dataset() function
    """
    # ARRANGE
    simple_selector = simple_select.SimpleSelect()
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
    train, valid, test = dataset_creator.load_dataset(creator=simple_selector, data_path=tmpdir / "context.hdf")
    sum_elements = len(train.idx) + len(valid.idx) + len(test.idx)
    splits = (len(train.idx) / sum_elements, len(valid.idx) / sum_elements, len(test.idx) / sum_elements)

    # ASSERT
    assert splits == splits_expected
