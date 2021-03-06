from abc import ABC

from tensorflow.keras import Model, layers

from src.model.classifiers.layers.shortcut import ShortcutBlock


class InceptionSubBlock(layers.Layer):
    """
    Inception Sub Block for use in inception model
    """

    def __init__(self):
        """
        Initializes InceptionSubBlock
        """
        super(InceptionSubBlock, self).__init__()
        self.bottleneck = layers.Conv1D(filters=32, kernel_size=1,
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
        Function builds Inception SubBlock.
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
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

    def get_config(self):
        config = super().get_config().copy()
        return config


class InceptionBlock(Model, ABC):
    """
    Inception neural network, initially proposed by https://github.com/hfawaz/InceptionTime
    """

    def __init__(self, num_classes):
        """
        Initializes InceptionBlock
        :param num_classes: number of classes in input
        """
        super(InceptionBlock, self).__init__()
        self.inception1 = InceptionSubBlock()
        self.inception2 = InceptionSubBlock()
        self.inception3 = InceptionSubBlock()
        self.inception4 = InceptionSubBlock()
        self.inception5 = InceptionSubBlock()
        self.inception6 = InceptionSubBlock()
        self.shortcut1 = ShortcutBlock(filters=128, kernel_size=1, bias=False)
        self.shortcut2 = ShortcutBlock(filters=128, kernel_size=1, bias=False)
        self.gap = layers.GlobalAveragePooling1D()
        self.add1 = layers.Add()
        self.add2 = layers.Add()
        self.relu1 = layers.Activation(activation='relu')
        self.relu2 = layers.Activation(activation='relu')
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds Inception model.
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
        """
        # Inception block 1
        x = self.inception1(input_tensor)
        # Inception block 2
        x = self.inception2(x)
        # # Inception block 3 + shortcut
        x = self.inception3(x)
        shortcut = self.shortcut1(input_tensor)
        x = self.add1([x, shortcut])
        x = self.relu1(x)
        y = x
        # # Inception block 4
        x = self.inception4(x)
        # # Inception block 5 + shortcut
        x = self.inception5(x)
        shortcut = self.shortcut2(y)
        x = self.add2([x, shortcut])
        x = self.relu2(x)
        # Output stage
        gap = self.gap(x)
        out = self.out(gap)
        return out
