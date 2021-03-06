from tensorflow import keras

from src.model.classifiers import fcn


def test__fcn():
    """
    Function that tests the structure FCNBlock
    """
    # ARRANGE
    layer_model_expected_names = ['cnn_block', 'cnn_block_1', 'cnn_block_2',
                                  'global_average_pooling1d', 'dense']
    param_count_expected = 265986

    # ACT
    model_fcn = fcn.FCNBlock(2)

    layer_model_out_names = []
    for layer_model_out in model_fcn.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)

    model_fcn.build((None, 195, 1))
    param_count = model_fcn.count_params()

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names
    assert param_count == param_count_expected


