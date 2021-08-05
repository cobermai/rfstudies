from tensorflow.keras import layers


class CNNBlock(layers.Layer):
    """Convolutional Neural Net Block"""
    def __init__(self, filters, kernel_size):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation(activation='relu')

    def call(self, input_tensor, training=None, mask=None):
        x = self.conv(input_tensor)
        x = self.bn(x, training=False)
        x = self.relu(x)
        return x
