import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pytest
from src.model.classifiers.layers import cnn

@pytest.mark.skip(reason="no way of currently testing this")
def test__cnn():
    """
    Function that tests the structure CNNLayer
    """
    # ARRANGE
    layer_model_expected_names = ['conv1d', 'batch_normalization', 'relu']

    # ACT
    layer_cnn = cnn.CNNBlock(2, 2)

    print(layer_cnn.get_config())
    a = tf.constant(4, 4, (4, 4, 4))
    a.dtype = float
    layer_cnn.call(a)
    layer_model_out_names = []
    for layer_model_out in layer_cnn.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)

    print(layer_model_out_names)
    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names


