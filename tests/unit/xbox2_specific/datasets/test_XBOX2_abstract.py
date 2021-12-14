from collections import namedtuple
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import xarray as xr

from src import dataset_creator
from src.xbox2_specific.datasets import XBOX2_abstract

data = namedtuple("data", ["X", "y", "idx"])


@patch.multiple(XBOX2_abstract.XBOX2Abstract, __abstractmethods__=set())
def test__train_valid_test_split():
    # ARRANGE
    selector = XBOX2_abstract.XBOX2Abstract()
    feature_list = ["Loadside win", "Tubeside win",
                    "Collector", "Gun", "IP before PC",
                    "PC IP", "WG IP", "IP Load",
                    "IP before structure", "US Beam Axis IP",
                    "Klystron Flange Temp", "Load Temp",
                    "PC Left Cavity Temp", "PC Right Cavity Temp",
                    "Bunker WG Temp", "Structure Input Temp",
                    "Chiller 1", "Chiller 2", "Chiller 3",
                    "PKI FT avg", "PSI FT avg", "PSR FT avg",
                    "PSI max", "PSR max", "PEI max",
                    "DC Down min", "DC Up min",
                    "PSI Pulse Width"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    events = np.arange(0, 100)
    samples = np.arange(0, 2)
    features = np.arange(0, 28)

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

    splits_expected = (0.7, 0.2, 0.1)
    manual_split = None
    # ACT
    train_out, valid_out, test_out = selector.train_valid_test_split(X_data_array, y_data_array,
                                                                     splits=splits_expected,
                                                                     manual_split=manual_split)

    assert len(train_out.X)/len(dummy_data_array) == splits_expected[0]
    assert len(valid_out.X)/len(dummy_data_array) == splits_expected[1]
    assert len(test_out.X)/len(dummy_data_array) == splits_expected[2]


@patch.multiple(XBOX2_abstract.XBOX2Abstract, __abstractmethods__=set())
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
    selector = XBOX2_abstract.XBOX2Abstract()
    feature_list = ["Loadside win", "Tubeside win",
                    "Collector", "Gun", "IP before PC",
                    "PC IP", "WG IP", "IP Load",
                    "IP before structure", "US Beam Axis IP",
                    "Klystron Flange Temp", "Load Temp",
                    "PC Left Cavity Temp", "PC Right Cavity Temp",
                    "Bunker WG Temp", "Structure Input Temp",
                    "Chiller 1", "Chiller 2", "Chiller 3",
                    "PKI FT avg", "PSI FT avg", "PSR FT avg",
                    "PSI max", "PSR max", "PEI max",
                    "DC Down min", "DC Up min",
                    "PSI Pulse Width"]
    feature_names = [feature.replace("/", "__").replace(" ", "_") for feature in feature_list]
    events = np.arange(0, 3)
    samples = np.arange(0, 2)
    features = np.arange(0, 28)

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


@patch.multiple(XBOX2_abstract.XBOX2Abstract, __abstractmethods__=set())
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
    selector = XBOX2_abstract.XBOX2Abstract()
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
