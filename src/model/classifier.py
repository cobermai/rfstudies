"""
model setup according to https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""
import os
from abc import ABC
import tensorflow.keras as keras
from src.model.classifiers import fcn
from src.model.classifiers import fcn_2dropout
from src.model.classifiers import resnet2

class Classifier(keras.Model, ABC):
    """
    Classifier class which acts as wrapper for tensorflow models.
    """
    def __init__(self, classifier_name, num_classes, output_directory):
        """Initialization of input data and hyperparameters"""
        super(Classifier, self).__init__()
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.output_directory = output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        if self.classifier_name == 'fcn':
            self.core_block = fcn.FCNBlock(num_classes)
        if self.classifier_name == 'fcn_2dropout':
            self.core_block = fcn_2dropout.FCN2DropoutBlock(num_classes)
        if self.classifier_name == 'resnet':
            self.core_block = resnet2.ResnetBlock(num_classes)

    def call(self, input_tensor, training=None, mask=None):
        """
        Function loads specified tensorflow model from model directory
        :return: specified tensorflow model from model directory
        """
        x = self.model(input_tensor)
        return x


