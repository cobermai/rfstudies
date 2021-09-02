import tensorflow.keras as keras
from src.model.classifiers import inception


def test__inception():
    """
    Function that tests the structure of InceptionBlock
    """
    # ARRANGE
    layer_model_expected_names = ['inception_sub_block', 'inception_sub_block_1', 'inception_sub_block_2',
                                  'inception_sub_block_3', 'inception_sub_block_4', 'inception_sub_block_5',
                                  'shortcut_block', 'shortcut_block_1', 'global_average_pooling1d', 'add',
                                  'add_1', 'activation_6', 'activation_7', 'dense']


    # ACT
    model_inception = inception.InceptionBlock(2)

    layer_model_out_names = []
    for layer_model_out in model_inception.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)
    print(layer_model_out_names)
    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names


