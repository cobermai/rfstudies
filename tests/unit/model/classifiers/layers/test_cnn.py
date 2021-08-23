import numpy as np
import pytest
import tensorflow as tf
from src.model.classifiers.layers import cnn


def test__cnn():
    """
    Function that tests the structure CNNLayer
    """
    # ARRANGE
    layer_cnn = cnn.CNNBlock(2, 2)
    x = tf.ones((3, 3, 3))
    # ACT
    y = layer_cnn(x)
    # ASSERT
    assert not np.isnan(y).all()


def test__shortcut_errors():
    with pytest.raises(ValueError):
        cnn.CNNBlock(0, 0)

    with pytest.raises(ValueError):
        cnn.CNNBlock(-1, -1)
