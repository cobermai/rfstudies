from tensorflow.keras import layers


class ShortcutBlock(layers.Layer):
    """Shortcut block"""
    def __init__(self, filters, kernel_size, bias=True):
        super(ShortcutBlock, self).__init__()
        self.conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=bias)
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=None, mask=None):
        x = self.conv(input_tensor)
        x = self.bn(x, training=False)
        return x
