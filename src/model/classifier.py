import os
import time
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import tensorflow.keras as keras


class Classifier:
    """
    Classifier class which acts as wrapper for tensorflow models.
    """
    def __init__(self, X_train, X_valid, y_train, y_valid, idx_train, idx_valid, build=True):
        """Initialization of input data and hyperparameters"""
        self.X_train = X_train
        self.y_train = y_train
        self.idx_train = idx_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.idx_valid = idx_valid
        self.nb_classes = len(np.unique(np.concatenate((y_train, y_valid), axis=0)))
        self.classifier_name = "fcn"
        self.output_directory = "/home/cobermai/PycharmProjects/mlframework/src/output/"
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory[:-1])
        self.input_shape = X_train.shape[1:]
        self.train_time = 0
        self.train_hist = []
        if build:
            self.classifier = self.create_classifier
        return

    def create_classifier(self):
        """
        Function loads specified tensorflow model from model directory
        :return: specified tensorflow model from model directory
        """
        if self.classifier_name == 'fcn':
            from model.classifiers import fcn
            return fcn.Classifier_FCN(self.output_directory, self.input_shape, self.nb_classes, verbose=True)

    def fit_classifier(self):
        """
        Function fits selected tensorflow model with specified input data
        """
        # transform the labels from integers to one hot vectors
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((self.y_train, self.y_valid), axis=0).reshape(-1, 1))
        y_train_hot = enc.transform(self.y_train.reshape(-1, 1)).toarray()
        y_valid_hot = enc.transform(self.y_valid.reshape(-1, 1)).toarray()

        start_time = time.time()
        self.train_hist = self.classifier.fit(self.X_train, y_train_hot, self.X_valid, y_valid_hot)
        self.train_time = time.time() - start_time

    def predict(self, X_test):
        """
        Function makes prediction with given test data
        :param X_test: data to make predictions with
        """
        model_path = self.output_directory + 'best_model.hdf5'
        classifier = keras.models.load_model(model_path)
        y_pred = classifier.predict(X_test)
        return y_pred

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
        df_results["n_train_healthy"] = sum(self.y_train)
        df_results["n_train_bd"] = (len(self.y_train) - sum(self.y_train))
        df_results["n_test_healthy"] = sum(y_test)
        df_results["n_test_bd"] = (len(y_test) - sum(y_test))
        df_results["class_imbalance_train"] = calc_class_imbalance(self.y_train)
        df_results["class_imbalance_test"] = calc_class_imbalance(y_test)

        df_results.to_csv(self.output_directory + "df_metrics.csv")
