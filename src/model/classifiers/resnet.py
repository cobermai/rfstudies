from abc import ABC

from tensorflow.keras import Model, layers

from src.model.classifiers.layers.cnn import CNNBlock
from src.model.classifiers.layers.shortcut import ShortcutBlock


class ResnetSubBlock(layers.Layer):
    """
    Block for use in Resnet model
    """

    def __init__(self, n_feature_maps):
        """
        Initializes ResnetSubBlock
        :param n_feature_maps: number of filters for each CNNBlock
        """
        super(ResnetSubBlock, self).__init__()
        # CNN Block 1
        self.cnn1 = CNNBlock(filters=n_feature_maps, kernel_size=8)
        # CNN Block 2
        self.cnn2 = CNNBlock(filters=n_feature_maps, kernel_size=5)
        # CNN Block 3
        self.conv = layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')
        self.bn = layers.BatchNormalization()
        # Output Block
        self.shortcut = ShortcutBlock(filters=n_feature_maps, kernel_size=1)
        self.add = layers.Add()
        self.relu = layers.Activation(activation='relu')
        self.n_feature_maps = n_feature_maps

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds resnet sub block
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
        """
        x = self.cnn1(input_tensor)
        y = self.cnn2(x)
        z = self.conv(y)
        z = self.bn(z)
        shortcut_y = self.shortcut(input_tensor)
        out = self.add([shortcut_y, z])
        out = self.relu(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_feature_maps': self.n_feature_maps
        })
        return config


class ResnetBlock(Model, ABC):
    """
    Resnet neural network, originally proposed by Wang et. al. in Time series classification from scratch with
    deep neural networks: A strong baseline, in Proceedings of the International Joint Conference
    on Neural Networks, Vol. 2017-May (2017)
    """
    def __init__(self, num_classes):
        """
        Initializes ResnetBlock
        :param num_classes: number of classes in input
        """
        super(ResnetBlock, self).__init__()
        self.resnet1 = ResnetSubBlock(n_feature_maps=64)
        self.resnet2 = ResnetSubBlock(n_feature_maps=128)
        self.cnn1 = CNNBlock(filters=128, kernel_size=8)
        self.cnn2 = CNNBlock(filters=128, kernel_size=5)
        self.conv = layers.Conv1D(filters=128, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.shortcut = ShortcutBlock(filters=128, kernel_size=1)
        self.add = layers.Add()
        self.relu = layers.Activation(activation='relu')
        self.gap = layers.GlobalAveragePooling1D()
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds Resnet model from 2 ResnetSubBlocks and a custom resnet sub block where a conv layer is skipped.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
        """
        # Resnet Block 1
        out1 = self.resnet1(input_tensor)
        # Resnet Block 2
        out2 = self.resnet2(out1)
        # Resnet Block 3
        x = self.cnn1(out2)
        y = self.cnn2(x)
        z = self.conv(y)
        z = self.bn1(z)
        shortcut_y = self.bn2(out2)
        out3 = self.add([shortcut_y, z])
        out3 = self.relu(out3)
        # Resnet output Block
        gap = self.gap(out3)
        out = self.out(gap)
        return out
