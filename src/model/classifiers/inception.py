from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras import layers
from src.model.classifiers.layers.shortcut import ShortcutBlock


class InceptionSubBlock(layers.Layer):
    """
    Inception Sub Block for use in inception model
    """

    def __init__(self, input_shape):
        super(InceptionSubBlock, self).__init__()
        self.bottleneck = layers.Conv1D(filters=32, kernel_size=1, input_shape=input_shape,
                                        padding='same', activation='linear', use_bias=False)
        self.conv1 = layers.Conv1D(filters=32, kernel_size=40,
                                   strides=1, padding='same',
                                   activation='linear', use_bias=False)
        self.conv2 = layers.Conv1D(filters=32, kernel_size=20,
                                   strides=1, padding='same',
                                   activation='linear', use_bias=False)
        self.conv3 = layers.Conv1D(filters=32, kernel_size=10,
                                   strides=1, padding='same',
                                   activation='linear', use_bias=False)
        self.max_pool = layers.MaxPool1D(pool_size=3, strides=1, padding='same')
        self.conv4 = layers.Conv1D(filters=32, kernel_size=1,
                                   strides=1, padding='same',
                                   activation='linear', use_bias=False)
        self.concatenate = layers.Concatenate(axis=2)
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation(activation='relu')

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds Inception model.
        """
        x = self.bottleneck(input_tensor)
        conv_a = self.conv1(x)
        conv_b = self.conv2(x)
        conv_c = self.conv3(x)
        max_pool = self.max_pool(input_tensor)
        conv_d = self.conv4(max_pool)
        conv_list = [conv_a, conv_b, conv_c, conv_d]
        x = self.concatenate(conv_list)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionBlock(Model, ABC):
    """
    Inception neural network, initially proposed by https://github.com/hfawaz/InceptionTime
    """
    def __init__(self, num_classes):
        super(InceptionBlock, self).__init__()
        self.inception1 = InceptionSubBlock(input_shape=(None, 195, 1))
        self.inception2 = InceptionSubBlock(input_shape=(None, 195, 128))
        self.shortcut1 = ShortcutBlock(filters=128, kernel_size=1, bias=False)
        self.shortcut2 = ShortcutBlock(filters=128, kernel_size=1, bias=False)
        self.gap = layers.GlobalAveragePooling1D()
        self.add = layers.Add()
        self.relu = layers.Activation(activation='relu')
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds Inception model.
        """
        # Inception block 1
        x = self.inception1(input_tensor)
        # Inception block 2
        x = self.inception2(x)
        # # Inception block 3 + shortcut
        x = self.inception2(x)
        shortcut = self.shortcut1(input_tensor)
        x = self.add([x, shortcut])
        x = self.relu(x)
        y = x
        # # Inception block 4
        x = self.inception2(x)
        # # Inception block 5 + shortcut
        x = self.inception2(x)
        shortcut = self.shortcut2(y)
        x = self.add([x, shortcut])
        x = self.relu(x)
        # Output stage
        gap = self.gap(x)
        out = self.out(gap)
        return out
