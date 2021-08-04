from tensorflow.keras import layers
from src.model.classifiers.layers.cnn import CNNBlock
from src.model.classifiers.layers.cnn_dropout import CNNDropoutBlock


class FCN2DropoutBlock(layers.Layer):
    """
    Fully convolutional neural network, with two dropout layers
    Initially proposed by Lukas Felsberger et. al. in "Lecture Notes in Computer Science, Vol. 12279 LNCS (2020)"
    """

    def __init__(self):
        super(FCN2DropoutBlock, self).__init__()
        self.cnn1 = CNNBlock(filters=128, kernel_size=8)
        self.cnn2_dropout = CNNDropoutBlock(filters=256, kernel_size=5)
        self.cnn3 = CNNBlock(filters=128, kernel_size=3)
        self.gap = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(0.5)

    def call(self, input_tensor, training=None, mask=None):
        """
        Function builds model out of 3 convolutional layers with batch normalization and the relu activation function.
        Second convolutional layer has a dropout of 0.5.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        """
        x = self.cnn1(input_tensor)
        x = self.cnn2_dropout(x)
        x = self.cnn3(x)
        x = self.gap(x)
        x = self.dropout(x)
        return x