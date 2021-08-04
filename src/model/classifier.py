# model setup according to https://www.tensorflow.org/guide/keras/custom_layers_and_models
import os
from abc import ABC
from pathlib import Path
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras
from src.model.classifiers import fcn


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
        self.output_layer = keras.layers.Dense(num_classes, activation='softmax')
        if self.classifier_name == 'fcn':
            self.CoreBlock = fcn.FCNBlock()

    def call(self, input_tensor, training=None, mask=None):
        """
        Function loads specified tensorflow model from model directory
        :return: specified tensorflow model from model directory
        """
        x = self.CoreBlock(input_tensor)
        x = self.output_layer(x)
        return x

    def eval_classifications(self, y_test, probabilities):
        """
        Function makes prediction with given test data
        :param y_test: true labels of test data
        :param probabilities: predicted labels of test data
        """
        predictions = np.argmax(probabilities, axis=1)
        df_results = pd.DataFrame([0])
        df_results["classifier_name"] = self.classifier_name

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

        df_results["balanced_accuracy"] = metrics.balanced_accuracy_score(y_test, predictions)
        df_results["bd_rate"] = tn / (tn + fp)
        df_results["roc_auc_score"] = metrics.roc_auc_score(y_test, probabilities[:, 1])
        df_results["f1_score"] = metrics.f1_score(y_test, predictions)

        df_results["train_time"] = self.train_time

        df_results["cm_tp"] = tp
        df_results["cm_tn"] = tn
        df_results["cm_fp"] = fp
        df_results["cm_fn"] = fn

        def calc_class_imbalance(y):
            return sum(y) / (len(y) - sum(y))

        df_results["n_train_healthy"] = sum(self.train.y)
        df_results["n_train_bd"] = (len(self.train.y) - sum(self.train.y))
        df_results["n_test_healthy"] = sum(y_test)
        df_results["n_test_bd"] = (len(y_test) - sum(y_test))
        df_results["class_imbalance_train"] = calc_class_imbalance(self.train.y)
        df_results["class_imbalance_test"] = calc_class_imbalance(y_test)

        df_results.to_csv(self.output_directory / "df_metrics.csv")
