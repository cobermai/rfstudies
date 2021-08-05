from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras import layers


class TimeCNNBlock(Model, ABC):
    """Convolutional Neural Net Block"""
    def __init__(self, num_classes):
        super(TimeCNNBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters=6, kernel_size=7, padding='valid', activation='sigmoid')
        self.AvePool = layers.AveragePooling1D(pool_size=3)
        self.conv2 = layers.Conv1D(filters=12, kernel_size=7, padding='valid', activation='sigmoid')
        self.flatten = layers.Flatten()
        self.out = layers.Dense(units=num_classes,activation='sigmoid')

    def call(self, input_tensor, training=None, mask=None):
        x = self.conv1(input_tensor)
        x = self.AvePool(x)
        x = self.conv2(x)
        x = self.AvePool(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
