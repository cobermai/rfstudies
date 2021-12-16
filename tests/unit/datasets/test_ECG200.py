from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.datasets.ECG200 import ECG200


def test__select_events():
    # ARRANGE
    creator = ECG200()
    path = Path("src/datasets/ECG200")

    # ACT
    data_array_out = creator.select_events(path)

    # ASSERT
    assert len(data_array_out) == 200
    assert hasattr(data_array_out, "is_train")


def test__select_features():
    # ARRANGE
    creator = ECG200()
    data = np.ones((10, 10))
    data_array = xr.DataArray(data=data)
    features_out_expected = data_array[:, :-1]

    # ACT
    features_out = creator.select_features(data_array)

    # ASSERT
    (features_out.values == features_out_expected.values).all()


#
def test__select_labels():
    creator = ECG200()
    data = np.ones((10, 10))
    data[:, 0::2] = -1
    data_array = xr.DataArray(data=data)
    labels_out_expected = data_array[:, -1]
    labels_out_expected[labels_out_expected == -1] = 0

    # ACT
    labels_out = creator.select_labels(data_array)

    # ASSERT
    (labels_out.values == labels_out_expected.values).all()


# def test__train_valid_test_split():
#     # ARRANGE
#     creator = ECG200()
#     creator.train_valid_test_split()
#
#     # ACT
#
#     # ASSERT
#
# def test__scale_data():
#     # ARRANGE
#     creator = ECG200()
#     creator.scale_data()
#
#     # ACT
#
#     # ASSERT
#
# def test__one_hot_encode():
#     # ARRANGE
#     creator = ECG200()
#     creator.one_hot_encode()
#
#     # ACT
#
#     # ASSERT
