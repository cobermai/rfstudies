import tensorflow.keras as keras
from src.model.classifiers import resnet


def test__resnet():
    """
    Function that tests the structure ResnetBlock
    """
    # ARRANGE
    layer_model_expected_names = ['resnet_sub_block', 'resnet_sub_block_1', 'cnn_block_4', 'cnn_block_5', 'conv1d_10',
                                  'batch_normalization_10', 'batch_normalization_11', 'shortcut_block_2', 'add_2',
                                  'activation_8', 'global_average_pooling1d', 'dense']
    param_count_expected = 506818

    # ACT
    model_resnet = resnet.ResnetBlock(2)

    layer_model_out_names = []
    for layer_model_out in model_resnet.layers:
        layer_model_out_name = layer_model_out.name
        layer_model_out_names.append(layer_model_out_name)

    model_resnet.build((None, 195, 1))
    param_count = model_resnet.count_params()

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names
    assert param_count == param_count_expected
