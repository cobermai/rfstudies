# import numpy as np
import tensorflow.keras as keras
# from tensorflow.keras import layers
# from src.model import classifier
from src.model.classifiers import fcn


def test__call(tmp_path):
    """
    Function that tests the call() method of ClassifierFCN
    """
    # ARRANGE
    layer_model_expected_names = ['conv1d', 'batch_normalization', 'conv1d_1', 'batch_normalization_1',
                                  'conv1d_2', 'batch_normalization_2', 'global_average_pooling1d', 'dense']

    # ACT
    model_fcn = fcn.ClassifierFCN(1, 2)
    layer_model_out_names = []
    for layer_model_out in model_fcn.layers:
        layer_model_out_name = layer_model_out.get_config()['name']
        layer_model_out_names.append(layer_model_out_name)

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names
