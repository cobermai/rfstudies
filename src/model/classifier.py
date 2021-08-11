"""
model setup according to https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""
from pathlib import Path
from tensorflow import keras
from src.model.classifiers import fcn
from src.model.classifiers import fcn_2dropout
from src.model.classifiers import resnet2
from src.model.classifiers import time_cnn
from src.model.classifiers import inception


class Classifier:
    """
    Classifier class which acts as wrapper for tensorflow models.
    """

    def __init__(self,
                 output_directory: Path,
                 classifier_name: str,
                 num_classes: int,
                 monitor: str,
                 loss: str,
                 optimizer: str,
                 epochs: int,
                 batch_size: int,
                 build=True
                 ):
        """
        Initializes the Classifier with specified settings
        :param output_directory: Directory for model output.
        :param classifier_name: Name of classifier, e.g. 'fcn'.
        :param num_classes: number of classes in input
        :param monitor: Name of performance variable to monitor.
        :param loss: Name of loss function to use in training.
        :param optimizer: Name of optimizer used in training.
        :param epochs: Number of epochs.
        :param batch_size: Number of input data used in each batch.
        :param build: Bool stating whether the model is to be build.
        """
        self.output_directory = output_directory
        self.classifier_name = classifier_name
        output_directory.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.monitor = monitor
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        if build:
            self.model = self.build_classifier()

    def build_classifier(self, **kwargs):
        """
        Builds classifier model
        **kwargs: Keyword arguments for tf.keras.Model.compile method
        :return: Tensorflow model for classification of time series data
        """
        if self.classifier_name == 'fcn':
            model = fcn.FCNBlock(self.num_classes)
        elif self.classifier_name == 'fcn_2dropout':
            model = fcn_2dropout.FCN2DropoutBlock(self.num_classes)
        elif self.classifier_name == 'resnet':
            model = resnet2.ResnetBlock(self.num_classes)
        elif self.classifier_name == 'time_cnn':
            model = time_cnn.TimeCNNBlock(self.num_classes)
        elif self.classifier_name == 'inception':
            model = inception.InceptionBlock(self.num_classes)
        else:
            raise AssertionError

        metrics = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),
        ]

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=metrics,
                      **kwargs)

        return model

    def fit_classifier(self, train, valid, **kwargs):
        """
        Trains classifier model on input data
        :param train: named tuple containing training set
        :param valid: named tuple containing validation set
        **kwargs: Keyword arguments for tf.keras.Model.fit method
        """
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=50,
            min_lr=0.0001)

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.output_directory / 'best_model.hdf5',
            save_weights_only=True,
            monitor=self.monitor,
            save_best_only=True)

        self.model.fit(
            x=train.X,
            y=train.y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(valid.X, valid.y),
            callbacks=[reduce_lr, model_checkpoint],
            **kwargs)
