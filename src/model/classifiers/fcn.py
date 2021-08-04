from tensorflow.keras import layers
from src.model.classifiers.layers.cnn import CNNBlock


class FCNBlock(layers.Layer):
    """Fully convolutional neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""
    def __init__(self):
        super(FCNBlock, self).__init__()
        self.cnn1 = CNNBlock(filters=128, kernel_size=8)
        self.cnn2 = CNNBlock(filters=256, kernel_size=5)
        self.cnn3 = CNNBlock(filters=128, kernel_size=3)
        self.gap = layers.GlobalAveragePooling1D()


    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds model out of 3 convolutional layers with batch normalization and the relu activation function.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        """
        x = self.cnn1(input_tensor)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.gap(x)
        return x
