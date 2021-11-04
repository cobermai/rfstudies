from collections import namedtuple
import h5py
import numpy as np
import xarray as xr
import pytest
from src.utils import dataset_creator
from src.xbox2_specific.datasets import XBOX2_event_all_bd_20ms

data = namedtuple("data", ["X", "y", "idx"])

@pytest.mark.parametrize("y, \
                         y_one_hot_expected",
                         [(np.array([1, 0, 1]),
                           np.array([[0, 1], [1, 0], [0, 1]])),
                          (np.array([0, 1, 0]),
                           np.array([[1, 0], [0, 1], [1, 0]])),
                          (np.zeros(1),
                           np.array([[1]]))
                          ])
def test__one_hot(y, y_one_hot_expected):
    """
    Test one_hot function of dataset_creator
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    labels_test = xr.DataArray(data=y, dims=["is_bd_in_20ms"])
    train = data(None, labels_test, None)
    valid = data(None, labels_test, None)
    test = data(None, labels_test, None)

    # ACT
    train_out, valid_out, test_out = selector.one_hot_encode(train, valid, test)

    # ASSERT
    assert (train_out.y == y_one_hot_expected).all()
    assert (valid_out.y == y_one_hot_expected).all()
    assert (test_out.y == y_one_hot_expected).all()

@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("manual_scale, input_data",
                         [(None, 4*np.ones(4)),
                          ([1, 2, 3, 4], 4*np.ones(4))
                          ])
def test__scale_data(manual_scale, input_data):
    """
    Test scale_data() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()

    df = pd.DataFrame({"run_no": [1, 2, 3, 4],
                       "test": input})
    train = data(df, None, None)
    valid = data(df, None, None)
    test = data(df, None, None)

    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = selector.scale_data(train, valid, test, manual_scale=manual_scale)
    print(X_output)

    # ASSERT
    assert (X_output == X_expected).all()


@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("dummy_features, selection_filter_expected",
                         [(np.array([True, True, True, True]), np.array([False, True, True, True])),
                          (np.array([False, False, False, False]), np.array([False, False, False, False]))
                          ])
def test__select_events(tmpdir, dummy_features, selection_filter_expected):
    """
    Test create_breakdown_selection_filter() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    path = tmpdir.join("dummy.hdf")
    context_dummy = h5py.File(path, 'w')
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
        f.create_dataset("is_bd_in_40ms", data=dummy_features)
        f.create_dataset("is_bd_in_20ms", data=dummy_features)
        f.create_dataset("is_bd", data=dummy_features)
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)

    # ACT
    np.random.seed(42)
    selection_filter_out = selector.select_events(path)

    # ASSERT
    assert (selection_filter_expected == selection_filter_out).all()

@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("dummy_data",
                         [([10, 20]),
                          ([-1., 3.])
                          ])
def test__select_features(dummy_data):
    """
    Test select_features() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    d = {'Timestamp': [1, 2],
         'PrevTrendData__Timestamp': [3, 4],
         'is_bd': [5, 6],
         'is_healthy': [7, 8],
         'is_bd_in_20ms': [9, 10],
         'is_bd_in_40ms': [11, 12]
         }
    selection_list = ["DC_Down__D1", "DC_Down__D9", "DC_Down__tsfresh__mean", "DC_Down__tsfresh__maximum",
                          "DC_Down__tsfresh__median", "DC_Down__tsfresh__minimum",
                          "DC_Up__D1", "DC_Up__D9", "DC_Up__tsfresh__mean", "DC_Up__tsfresh__maximum",
                          "DC_Up__tsfresh__median", "DC_Up__tsfresh__minimum",
                          "PEI_Amplitude__pulse_length", "PEI_Amplitude__pulse_amplitude",
                          "PKI_Amplitude__pulse_length", "PKI_Amplitude__pulse_amplitude",
                          "PSI_Amplitude__pulse_length", "PSI_Amplitude__pulse_amplitude"]
    for name in selection_list:
        d[name] = dummy_data
    df = pd.DataFrame(data=d)
    X_expected = df[pd.Index(selection_list)].to_numpy(dtype=float)
    X_expected = np.nan_to_num(X_expected[..., np.newaxis])

    # ACT
    X_out = selector.select_features(df)

    # ASSERT
    assert (X_out == X_expected).all()

@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("data",
                         [np.ones((10,), dtype=bool),
                          np.zeros((10,), dtype=bool)
                          ])
def test__select_labels(data):
    """
    Test load_X_data() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    d = {'is_healthy': data}
    df = pd.DataFrame(data=d)
    y_expected = df['is_healthy'].to_numpy(dtype=float)

    # ACT
    y_out = selector.select_labels(df)

    # ASSERT
    assert (y_out == y_expected).all()


@pytest.mark.skip(reason="Needs to be updated for new code structure")
def test__load_dataset(tmpdir):
    """
    Test load_dataset() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
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
        selection_list = ["DC_Down__D1", "DC_Down__D9", "DC_Down__tsfresh__mean", "DC_Down__tsfresh__maximum",
                          "DC_Down__tsfresh__median", "DC_Down__tsfresh__minimum",
                          "DC_Up__D1", "DC_Up__D9", "DC_Up__tsfresh__mean", "DC_Up__tsfresh__maximum",
                          "DC_Up__tsfresh__median", "DC_Up__tsfresh__minimum",
                          "PEI_Amplitude__pulse_length", "PEI_Amplitude__pulse_amplitude",
                          "PKI_Amplitude__pulse_length", "PKI_Amplitude__pulse_amplitude",
                          "PSI_Amplitude__pulse_length", "PSI_Amplitude__pulse_amplitude"]
        for name in selection_list:
            f.create_dataset(name, data=np.ones((10,)))
        f["Timestamp"] = dummy_event_timestamps.astype(h5py.opaque_dtype(dummy_event_timestamps.dtype))
        f["PrevTrendData/Timestamp"] = dummy_trend_timestamps.astype(h5py.opaque_dtype(dummy_trend_timestamps.dtype))
        f.create_dataset("clic_label/is_healthy", data=dummy_is_healthy_labels)
        f.create_dataset("is_healthy", data=dummy_is_healthy_labels)

    splits_expected = (0.7, 0.2, 0.1)

    # ACT
    np.random.seed(42)
    train, valid, test = dataset_creator.load_dataset(creator=selector, data_path=tmpdir)
    sum_elements = len(train.idx) + len(valid.idx) + len(test.idx)
    splits = (len(train.idx) / sum_elements, len(valid.idx) / sum_elements, len(test.idx) / sum_elements)

    # ASSERT
    assert splits == splits_expected
