from abc import ABC
from tensorflow.keras import Model, layers
from src.model.classifiers.layers.cnn import CNNBlock


class FCNBlock(Model, ABC):
    """Fully Convolutional Neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""
    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Initializes FCNBlock
        :param num_classes: Number of classes in data set
        """
        super(FCNBlock, self).__init__(**kwargs)
        self.input_layer = layers.Input(input_shape)
        self.cnn1 = layers.Dense(100, activation='relu')
        self.gap = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(num_classes, activation='softmax')
        self.out = self.call(self.input_layer)

        # Reinitial
        super(FCNBlock, self).__init__(inputs=self.input_layer, outputs=self.out, **kwargs)

    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out
        )

    def call(self, input_tensor, training=False, mask=None):
        """
        Function builds FCN model out of 3 convolutional layers with batch normalization and the relu
        activation function. In the end there is a global average pooling layer which feeds the output into a
        softmax classification layer.
        :param input_tensor: input to model
        :param training: bool for specifying whether model should be training
        :param mask: mask for specifying whether some values should be skipped
        """
        x = self.cnn1(input_tensor)
        x = self.gap(x)
        x = self.dense(x)
        return x

