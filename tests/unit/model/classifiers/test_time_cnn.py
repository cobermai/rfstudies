import tensorflow.keras as keras
from src.model.classifiers import time_cnn


def test__time_cnn():
    """
    Function that tests the structure TimeCNNBlock
    """
    # ARRANGE
    layer_model_expected_names = ['conv1d', 'conv1d_1', 'average_pooling1d', 'conv1d_2',
                                  'conv1d_3','average_pooling1d_1', 'flatten', 'dense']

    # ACT
    model_time_cnn = time_cnn.TimeCNNBlock(2)

    layer_model_out_names = []
    for layer_model_out in model_time_cnn.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names


