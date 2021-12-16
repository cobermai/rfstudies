from tensorflow import keras

from src.model.classifiers import fcn_2dropout


def test__fcn_2dropout():
    """
    Function that tests the structure FCN2DropoutBlock
    """
    # ARRANGE
    layer_model_expected_names = ['cnn_block', 'cnn_dropout_block', 'cnn_block_1',
                                  'global_average_pooling1d', 'dropout_1', 'dense']
    param_count_expected = 265986

    # ACT
    model_fcn_2dropout = fcn_2dropout.FCN2DropoutBlock(2)

    layer_model_out_names = []
    for layer_model_out in model_fcn_2dropout.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)

    model_fcn_2dropout.build((None, 195, 1))
    param_count = model_fcn_2dropout.count_params()

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names
    assert param_count == param_count_expected
