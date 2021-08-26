import tensorflow.keras as keras
from src.model.classifiers import inception


def test__inception():
    """
    Function that tests the structure InceptionBlock
    """
    # ARRANGE
    layer_model_expected_names = ['inception_sub_block', 'inception_sub_block_1', 'shortcut_block', 'shortcut_block_1',
                                  'global_average_pooling1d', 'add', 'activation_2', 'dense']

    # ACT
    model_inception = inception.InceptionBlock(2)

    layer_model_out_names = []
    for layer_model_out in model_inception.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names


