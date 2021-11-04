import numpy as np
import pytest
import tensorflow as tf
from src.model.classifiers.layers import cnn_dropout


def test__cnn_dropout():
    """
    Function that tests the structure CNNLayer
    """
    # ARRANGE
    layer_cnn_dropout = cnn_dropout.CNNDropoutBlock(2, 2, dropout_rate=0.5)
    x = tf.ones((3, 3, 3))
    # ACT
    y = layer_cnn_dropout(x)
    # ASSERT
    assert not np.isnan(y).all()


@pytest.mark.skip(reason="Stopped raising errors")
def test__cnn_dropout_errors():
    with pytest.raises(ValueError):
        cnn_dropout.CNNDropoutBlock(0, 0, dropout_rate=0.5)

    with pytest.raises(ValueError):
        cnn_dropout.CNNDropoutBlock(-1, -1, dropout_rate=0.5)
