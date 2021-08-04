from tensorflow.keras import layers


class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=None, mask=None):
        x = self.conv(input_tensor)
        x = self.bn(x, training=False)
        return x




