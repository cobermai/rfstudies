import numpy as np
import tensorflow as tf
from src.model.classifiers.layers import cnn_dropout


def test__cnn_dropout():
    """
    Function that tests the structure CNNLayer
    """
    # ARRANGE
    layer_cnn_dropout = cnn_dropout.CNNDropoutBlock(2, 2)
    x = tf.ones((3, 3, 3))
    # ACT
    y = layer_cnn_dropout(x)
    # ASSERT
    assert not np.isnan(y).all()
