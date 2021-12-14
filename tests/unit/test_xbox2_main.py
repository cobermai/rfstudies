"""
tests the main module
"""
from pathlib import Path

import pytest

import xbox2_main


@pytest.mark.parametrize("transform, features",
                         [('True', 'False'), ('False', 'True'),
                          ('True', 'True'), ('False', 'False')]
                         )
def test__parse_input_arguments(transform, features):
    # ARRANGE
    file_path_expected = 'some_file_path'
    data_path_expected = ' some_data_path'
    dataset_name_expected = 'some_dataset_name'
    transform_to_hdf5_expected = transform
    calculate_features_expected = features
    input_args = ['--file_path', file_path_expected,
                  '--data_path', data_path_expected,
                  '--dataset_name', dataset_name_expected,
                  '--transform_to_hdf5', transform_to_hdf5_expected,
                  '--calculate_features', calculate_features_expected]

    # ACT
    args_out = xbox2_main.parse_input_arguments(args=input_args)

    # ASSERT
    assert args_out.file_path == Path(file_path_expected)
    assert args_out.data_path == Path(data_path_expected)
    assert args_out.dataset_name == dataset_name_expected
    assert args_out.transform_to_hdf5 == bool(transform_to_hdf5_expected)
    assert args_out.calculate_features == bool(calculate_features_expected)


@pytest.mark.skip(reason="currently no way of testing this")
def test__transformation():
    assert True


@pytest.mark.skip(reason="currently no way of testing this")
def test__feature_handling():
    assert True


@pytest.mark.skip(reason="currently no way of testing this")
def test__modeling():
    assert True




