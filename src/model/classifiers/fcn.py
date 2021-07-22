import tensorflow.keras as keras

class ClassifierFcn:
    """Fully convolutional neural network, initially proposed by https://github.com/hfawaz/dl-4-tsc"""
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        """Initialization of model parameters"""
        self.output_directory = output_directory
        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        """
        Function builds model out of 3 convolutional layers with batch normalization and the relu activation function.
        In the end there is a global average pooling layer which feeds the output into a softmax classification layer.
        """
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Function model with specified input data, which is a three-dimensional tensor (`K`, `N`, `M`) where `K` is the
        number of events, `N` is the number of input samples, and `M` is the number of input features.
        :return: hist: history of model during training
        """
        batch_size = 16
        nb_epochs = 2
        class_weight = {0: 10, 1: 1}

        mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))

        hist = self.model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(X_valid, y_valid), callbacks=self.callbacks,
                              class_weight=class_weight)

        return hist
