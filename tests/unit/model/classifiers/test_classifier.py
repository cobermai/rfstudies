from collections import namedtuple
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import tensorflow.keras as keras
from src.model import classifier
from src.model.classifiers import fcn


def test__call(tmp_path):
    '''
    Test call function of class Classifier
    '''
    # ARRANGE
    data = namedtuple("data", ["X", "y", "idx"])
    X_train = np.empty(1)
    y_train = np.empty(1)
    idx_train = np.empty(1)
    X_valid = np.empty(1)
    y_valid = np.empty(1)
    idx_valid = np.empty(1)

    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)

    nb_classes = len(np.unique(np.concatenate((train.y, valid.y), axis=0)))
    data_shape = train.X.shape[1:]

    model_expected = fcn.ClassifierFCN(data_shape, nb_classes)
    layer_model_expected_names = []
    for layer_model_expected in model_expected.layers:
        layer_model_expected_name = layer_model_expected.get_config()['name']
        layer_model_expected_names.append(layer_model_expected_name)

    keras.backend.clear_session()

    clf = classifier.Classifier(train_data=train,
                                valid_data=valid,
                                classifier_name="fcn",
                                output_directory=tmp_path)
    inputs = 0
    # ACT
    model_out = clf.call(inputs)
    layer_model_out_names = []
    for layer_model_out in model_out.layers:
        layer_model_out_name = layer_model_out.get_config()['name']
        layer_model_out_names.append(layer_model_out_name)

    keras.backend.clear_session()

    # ASSERT
    assert layer_model_expected_names == layer_model_out_names


def test__one_hot(tmp_path):
    '''
    Test one_hot function of class Classifier
    '''
    # ARRANGE
    data = namedtuple("data", ["X", "y", "idx"])
    X_train = np.empty(1)
    y_train = np.array(['good', 'bad', 'good'])
    idx_train = np.empty(1)
    X_valid = np.empty(1)
    y_valid = np.array(['bad', 'good', 'bad'])
    idx_valid = np.empty(1)

    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)

    clf = classifier.Classifier(train_data=train,
                                valid_data=valid,
                                classifier_name="fcn",
                                output_directory=tmp_path)

    y_train_hot_expected = np.array([[0, 1], [1, 0], [0, 1]])
    y_valid_hot_expected = np.array([[1, 0], [0, 1], [1, 0]])

    # ACT
    y_train_hot, y_valid_hot = clf.one_hot()

    # ASSERT
    assert (y_train_hot == y_train_hot_expected).all()
    assert (y_valid_hot == y_valid_hot_expected).all()


def test__eval_classifications(tmp_path):
    '''
    Test eval_classifications function of class Classifier
    '''
    # ARRANGE
    data = namedtuple("data", ["X", "y", "idx"])
    X_train = np.empty(1)
    y_train = np.array([True, False, True])
    idx_train = np.empty(1)
    X_valid = np.empty(1)
    y_valid = np.array([False, True, False])
    idx_valid = np.empty(1)

    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)

    classifier_name = "fcn"
    clf = classifier.Classifier(train_data=train,
                                valid_data=valid,
                                classifier_name=classifier_name,
                                output_directory=tmp_path)

    probabilities = np.array([[0.1, 0.7], [0.5, 0.2], [0.3, 0.9]])
    df_results_expected = pd.DataFrame([0])
    df_results_expected["classifier_name"] = classifier_name
    df_results_expected["balanced_accuracy"] = 0.75
    df_results_expected["bd_rate"] = 0.5,
    df_results_expected["roc_auc_score"] = 1.0,
    df_results_expected["f1_score"] = 2 / 3,
    df_results_expected["train_time"] = 2,
    df_results_expected["cm_tp"] = 1,
    df_results_expected["cm_tn"] = 1,
    df_results_expected["cm_fp"] = 1,
    df_results_expected["cm_fn"] = 0,
    df_results_expected["n_train_healthy"] = 2,
    df_results_expected["n_train_bd"] = 1,
    df_results_expected["n_test_healthy"] = 1,
    df_results_expected["n_test_bd"] = 2,
    df_results_expected["class_imbalance_train"] = 2.0,
    df_results_expected["class_imbalance_test"] = 0.5

    # Write to CSV and read from CSV due to simulate behaviour in eval_classifications
    data_path_expected = tmp_path / "df_metrics_expected.csv"
    df_results_expected.to_csv(data_path_expected)
    df_results_expected = pd.read_csv(data_path_expected)

    y_test = np.array([False, False, True])

    # ACT
    clf.train_time = 2
    clf.eval_classifications(y_test, probabilities)
    data_path = tmp_path / "df_metrics.csv"
    df_results = pd.read_csv(data_path)

    # ASSERT
    assert_frame_equal(df_results, df_results_expected)
