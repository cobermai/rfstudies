from collections import namedtuple

import h5py
import numpy as np
import pytest
import xarray as xr

from src.utils import dataset_creator
from src.xbox2_specific.datasets import XBOX2_event_all_bd_20ms

data = namedtuple("data", ["X", "y", "idx"])


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


@pytest.mark.parametrize("dummy_data, dummy_label",
                         [(np.ones(shape=(3, 2, 1)), np.array([0, 1, 0]))
                          ])
def test__select_features(dummy_data, dummy_label):
    """
    Test select_features() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    dummy_data_array = xr.DataArray(data=dummy_data,
                                    dims=["event", "sample", "feature"]
                                    )
    dummy_data_array = dummy_data_array.assign_coords(is_bd_in_20ms=("event", dummy_label))

    output_expected = xr.DataArray(data=dummy_data,
                                   dims=["event", "sample", "feature"]
                                   )

    # ACT
    output = selector.select_features(dummy_data_array)

    # ASSERT
    assert (output == output_expected).all()


@pytest.mark.parametrize("dummy_data, dummy_label",
                         [(np.ones(shape=(3, 2, 1)), np.array([0, 1, 0]))
                          ])
def test__select_labels(dummy_data, dummy_label):
    """
    Test load_X_data() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    dummy_data_array = xr.DataArray(data=dummy_data,
                                    dims=["event", "sample", "feature"]
                                    )
    dummy_data_array = dummy_data_array.assign_coords(is_bd_in_20ms=("event", dummy_label))

    output_expected = dummy_data_array["is_bd_in_20ms"]

    # ACT
    output = selector.select_labels(dummy_data_array)

    # ASSERT
    assert (output == output_expected).all()


def test__train_valid_test_split():
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    feature_list = ["PEI Amplitude",
                    "PSI Amplitude",
                    "PSR Amplitude",
                    "PKI Amplitude",
                    "DC Up",
                    "DC Down"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    events = np.arange(0, 100)
    samples = np.arange(0, 2)
    features = np.arange(0, 6)

    dummy_data, _, _ = np.meshgrid(events, samples, features, indexing='ij')

    dummy_data_array = xr.DataArray(data=dummy_data,
                                    dims=["event", "sample", "feature"],
                                    coords={"feature": feature_names})
    dummy_timestamps = np.array([np.datetime64('2021-08-18T17:59:01') + np.timedelta64(20, 's') for i in range(100)])
    dummy_run_no = np.concatenate([np.ones(10), 2*np.ones(10), 3*np.ones(20), 4*np.ones(10), 5*np.ones(10),
                                   6*np.ones(10), 7*np.ones(10), 8*np.ones(10), 9*np.ones(10)])
    dummy_data_array = dummy_data_array.assign_coords(timestamp_event=("event", dummy_timestamps))
    dummy_data_array = dummy_data_array.assign_coords(run_no=("event", dummy_run_no))
    dummy_labels = np.ones(len(dummy_data_array))
    dummy_data_array = dummy_data_array.assign_coords(is_bd_in_20ms=("event", dummy_labels))

    X_data_array = dummy_data_array.drop_vars("is_bd_in_20ms")
    y_data_array = dummy_data_array["is_bd_in_20ms"]

    splits_expected = (0.7, 0.20, 0.10)
    manual_split = None
    # ACT
    train_out, valid_out, test_out = selector.train_valid_test_split(X_data_array, y_data_array,
                                                                     splits=splits_expected,
                                                                     manual_split=manual_split)

    assert len(train_out.X)/len(dummy_data_array) == splits_expected[0]
    assert len(valid_out.X)/len(dummy_data_array) == splits_expected[1]
    assert len(test_out.X)/len(dummy_data_array) == splits_expected[2]


@pytest.mark.skip(reason="Not finished yet")
@pytest.mark.parametrize("manual_scale",
                         [None,
                          [1, 2, 3]
                          ])
def test__scale_data(manual_scale):
    """
    Test scale_data() function
    """
    # ARRANGE
    selector = XBOX2_event_all_bd_20ms.XBOX2EventAllBD20msSelect()
    feature_list = ["PEI Amplitude",
                    "PSI Amplitude",
                    "PSR Amplitude",
                    "PKI Amplitude",
                    "DC Up",
                    "DC Down"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    events = np.arange(0, 3)
    samples = np.arange(0, 2)
    features = np.arange(0, 6)

    dummy_data, _, _ = np.meshgrid(events, samples, features, indexing='ij')

    dummy_data_array = xr.DataArray(data=dummy_data,
                                    dims=["event", "sample", "feature"],
                                    coords={"feature": feature_names})
    dummy_timestamps = np.array([np.datetime64('2021-08-18T17:59:01'),
                                 np.datetime64('2021-08-18T17:59:02'),
                                 np.datetime64('2021-08-18T17:59:03')])
    dummy_data_array = dummy_data_array.assign_coords(timestamp_event=("event", dummy_timestamps))
    dummy_data_array = dummy_data_array.assign_coords(run_no=("event", np.array([1, 2, 3])))

    train = data(dummy_data_array, None, None)
    valid = data(dummy_data_array, None, None)
    test = data(dummy_data_array, None, None)

    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    train_out, valid_out, test_out = selector.scale_data(train, valid, test, manual_scale=manual_scale)
    print(train_out.X)

    # ASSERT
    assert train_out.X.values == X_expected


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
