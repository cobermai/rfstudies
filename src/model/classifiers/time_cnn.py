from abc import ABC

from tensorflow.keras import Model, layers


class TimeCNNBlock(Model, ABC):
    """
    TimeCNN network, originally proposed by Zhao et. al. in Convolutional neural networks for
    time series classification, Journal of Systems Engineering and Electronics 28, 162 (2017).
    """
    def __init__(self, num_classes):
        """
        Initializes TimeCNNBlock
        :param num_classes: number of classes in input
        """
        super(TimeCNNBlock, self).__init__()
        self.conv1_valid = layers.Conv1D(filters=6, kernel_size=7, padding='valid', activation='sigmoid')
        self.conv1_same = layers.Conv1D(filters=6, kernel_size=7, padding='same', activation='sigmoid')
        self.AvePool1 = layers.AveragePooling1D(pool_size=3)
        self.conv2_valid = layers.Conv1D(filters=12, kernel_size=7, padding='valid', activation='sigmoid')
        self.conv2_same = layers.Conv1D(filters=12, kernel_size=7, padding='same', activation='sigmoid')
        self.AvePool2 = layers.AveragePooling1D(pool_size=3)
        self.flatten = layers.Flatten()
        self.out = layers.Dense(units=num_classes, activation='sigmoid')

    def call(self, input_tensor, training=None, mask=None):
        """
        Builds TimeCNN model
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
        """
        if input_tensor.shape[1] < 60:
            x = self.conv1_same(input_tensor)
        else:
            x = self.conv1_valid(input_tensor)
        x = self.AvePool1(x)
        if input_tensor.shape[1] < 60:
            x = self.conv2_same(x)
        else:
            x = self.conv2_valid(x)
        x = self.AvePool2(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
