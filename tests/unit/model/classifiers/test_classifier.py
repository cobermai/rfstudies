from collections import namedtuple
import numpy as np
import pandas as pd
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

    clf = classifier.Classifier(train_data=train,
                     valid_data=valid,
                     classifier_name="fcn",
                     output_directory=tmp_path)

    model_expected = fcn.ClassifierFCN(clf.data_shape, clf.nb_classes)
    print(clf.nb_classes)

    # ACT
    model_out = clf.call(clf.nb_classes)

    print('expected')
    print(model_expected.layers[2].get_config())
    print('called')
    print(model_out.layers[2].get_config()['name'])
    print('test')
    print(model_expected.layers[2].get_config()['name'] in model_out.layers[2].get_config()['name'])
    layer_model_expected_names = []
    layer_model_out_names = []
    sep = '_' # used for removing numbering of layers after layer type, e.g. Conv1D_3 -> Conv1D
    for layer_model_expected, layer_model_out in zip(model_expected.layers, model_out.layers):
        layer_model_expected_name = layer_model_expected.get_config()['name'].split(sep, 1)[0]
        layer_model_expected_names.append(layer_model_expected_name)
        layer_model_out_name = layer_model_out.get_config()['name'].split(sep, 1)[0]
        layer_model_out_names.append(layer_model_out_name)
    print(layer_model_expected_names == layer_model_out_names)

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
    X_test = np.empty(1)
    y_test = np.array([False, False, True])
    idx_test = np.empty(1)

    train = data(X_train, y_train, idx_train)
    valid = data(X_valid, y_valid, idx_valid)
    test = data(X_test, y_test, idx_test)

    clf = classifier.Classifier(train_data=train,
                                valid_data=valid,
                                classifier_name="fcn",
                                output_directory=tmp_path)

    probabilities = np.array([[0.1, 0.7], [0.5, 0.2], [0.3, 0.9]])

    train_time = 2

    balanced_accuracy_expected = 0.75
    tn_expected, fp_expected, fn_expected, tp_expected = 1, 1, 0, 1
    bd_rate_expected = tn_expected / (tn_expected + fp_expected)
    roc_auc_score_expected = 1
    f1_score_expected = 2/3
    n_train_healthy_expected = train.y.sum()
    n_train_bd_expected = len(train.y) - train.y.sum()
    n_test_healthy_expected = test.y.sum()
    n_test_bd_expected = len(test.y) - test.y.sum()
    class_imbalance_train_expected = train.y.sum() / (len(train.y) - sum(train.y))
    class_imbalance_test_expected = test.y.sum() / (len(test.y) - sum(test.y))

    # ACT
    clf.train_time = train_time
    clf.eval_classifications(y_test, probabilities)
    data_path = tmp_path / "df_metrics.csv"
    df_results = pd.read_csv(data_path)

    # ASSERT
    assert (df_results["balanced_accuracy"] == balanced_accuracy_expected).all()
    assert (df_results["bd_rate"] == bd_rate_expected).all()
    assert (df_results["roc_auc_score"] == roc_auc_score_expected).all()
    assert (df_results["f1_score"] == f1_score_expected).all()
    assert (df_results["train_time"] == train_time).all()
    assert (df_results["cm_tp"] == tp_expected).all()
    assert (df_results["cm_tn"] == tn_expected).all()
    assert (df_results["cm_fp"] == fp_expected).all()
    assert (df_results["cm_fn"] == fn_expected).all()
    assert (df_results["n_train_healthy"] == n_train_healthy_expected).all()
    assert (df_results["n_train_bd"] == n_train_bd_expected).all()
    assert (df_results["n_test_healthy"] == n_test_healthy_expected).all()
    assert (df_results["n_test_bd"] == n_test_bd_expected).all()
    assert (df_results["class_imbalance_train"] == class_imbalance_train_expected).all()
    assert (df_results["class_imbalance_test"] == class_imbalance_test_expected).all()
