from tensorflow.keras import layers


class CNNDropoutBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dropout_rate):
        super(CNNDropoutBlock, self).__init__()
        self.conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=None, mask=None):
        x = self.conv(input_tensor)
        x = self.dropout(x)
        x = self.bn(x, training=False)
        return x
