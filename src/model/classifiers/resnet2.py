from tensorflow.keras import layers
from src.model.classifiers.layers.cnn import CNNBlock
from src.model.classifiers.layers.shortcut import ShortcutBlock


class ResnetSubBlock(layers.Layer):
    """Block for use in Resnet model"""

    def __init__(self, n_feature_maps):
        super(ResnetSubBlock, self).__init__()
        self.cnn1 = CNNBlock(filters=n_feature_maps, kernel_size=8)
        self.cnn2 = CNNBlock(filters=n_feature_maps, kernel_size=5)
        self.conv = layers.Conv1D(filters=n_feature_maps, kernel_size=3 padding='same')
        self.bn = layers.BatchNormalization()
        self.shortcut = ShortcutBlock()
        self.add = layers.Add()
        self.relu = layers.Activation(activation='relu')
        self.gap = layers.GlobalAveragePooling1D()

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds model out of 3 convolutional layers with batch normalization and the relu activation function.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        """
        x = self.cnn1(input_tensor)
        y = self.cnn2(x)
        z = self.conv(y)
        z = self.bn(z)
        shortcut_y = self.shortcut(input_tensor)
        out = self.add([shortcut_y, z])
        out = self.relu(out)
        return out


class ResnetBlock(layers.Layer):
    """Resnet neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""

    def __init__(self, nb_classes):
        super(ResnetBlock, self).__init__()
        self.resnet1 = ResnetSubBlock(n_feature_maps=64)
        self.resnet2 = ResnetSubBlock(n_feature_maps=128)
        self.cnn1 = CNNBlock(filters=128, kernel_size=8)
        self.cnn2 = CNNBlock(filters=128, kernel_size=5)
        self.conv = layers.Conv1D(filters=128, kernel_size=3, padding='same')
        self.bn = layers.BatchNormalization()
        self.shortcut = ShortcutBlock()
        self.add = layers.Add()
        self.gap = layers.GlobalAveragePooling1D()
        self.out = layers.Dense(nb_classes, activation='softmax')

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds Resnet model from 2 ResnetSubBlocks and a custom resnet subblock where a conv is skipped.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        """
        # Block 1
        out1 = self.resnet1(input_tensor)
        # Block 2
        out2 = self.resnet2(out1)
        # Block 3
        x = self.cnn1(out2)
        y = self.cnn2(x)
        z = self.conv(y)
        z = self.bn(z)
        shortcut_y = self.bn(x)
        out3 = self.add([shortcut_y, z])
        out3 = self.relu(out3)
        # output stage
        gap = self.gap(out3)
        out = self.out(gap)
        return out
