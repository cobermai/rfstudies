from pathlib import Path
import h5py
import numpy as np
import pytest
from src.utils.handler_tools.context_data_writer import ColumnWiseContextDataWriter
from src.xbox2_specific.datasets import XBOX2_trend_bd20ms


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
    y_one_hot = XBOX2_trend_bd20ms.one_hot_encode(y=y)

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
    X_output = XBOX2_trend_bd20ms.scale_data(X)
    print(X_output)
    # ASSERT
    assert (X_output == X_expected).all()


@pytest.mark.skip(reason="no way of currently testing this")
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



