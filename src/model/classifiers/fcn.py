from abc import ABC

import tensorflow.keras as keras
from tensorflow.keras import layers

class ClassifierFCN(keras.Model, ABC):
    """Fully convolutional neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""

    def __init__(self, input_shape, nb_classes, name="fcn", **kwargs):
        super(ClassifierFCN, self).__init__(name=name, **kwargs)
        # Input Layer
        self.input_layer = keras.layers.Input(input_shape)
        # Layer of Block 1
        self.conv1 = layers.Conv1D(filters=128, kernel_size=8, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        # Layer of Block 1
        self.conv2 = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        # Layer of Block 1
        self.conv3 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        # Gap Layer
        self.gap = layers.GlobalAveragePooling1D()
        # Output
        self.dense = layers.Dense(nb_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        """
        Function builds model out of 3 convolutional layers with batch normalization and the relu activation function.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        """
        # forward pass: block 1
        x = self.input_layer(inputs)
        x = self.conv1(x)
        x = self.bn1(x)

        # forward pass: block 2
        x = self.conv2(x)
        x = self.bn2(x)

        # forward pass: block 3
        x = self.conv3(x)
        x = self.bn3(x)

        # gap and classifier
        x = self.gap(x)
        return self.dense(x)


"""



        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        #return model
"""
