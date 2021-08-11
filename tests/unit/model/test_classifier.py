import pytest
import tensorflow.keras as keras
from src.model import classifier



@pytest.mark.parametrize("build, model_expected",
                         [(True, True),
                          (False, False)]
                         )
def test__classifier(tmp_path, build, model_expected):
    """
    Test of Classifier class instantiation
    """
    # ARRANGE
    classifier_name_expected = 'fcn'
    num_classes_expected = 3
    monitor_expected = 'loss'
    loss_expected = 'categorical_crossentropy'
    optimizer_expected = 'adam'
    epochs_expected = 2
    batch_size_expected = 16

    # ACT
    clf = classifier.Classifier(classifier_name=classifier_name_expected,
                                num_classes=num_classes_expected,
                                output_directory=tmp_path,
                                monitor=monitor_expected,
                                loss=loss_expected,
                                optimizer=optimizer_expected,
                                epochs=epochs_expected,
                                batch_size=batch_size_expected,
                                build=build
                                )

    # ASSERT
    assert clf.classifier_name == classifier_name_expected
    assert clf.num_classes == num_classes_expected
    assert clf.output_directory == tmp_path
    assert clf.monitor == monitor_expected
    assert clf.loss == loss_expected
    assert clf.optimizer == optimizer_expected
    assert clf.epochs == epochs_expected
    assert clf.batch_size == batch_size_expected
    assert hasattr(clf, 'model') == model_expected


def test__build_classifier(tmp_path):
    # ARRANGE
    classifier_name_expected = 'fcn'
    num_classes_expected = 3
    monitor_expected = 'loss'
    loss_expected = 'categorical_crossentropy'
    optimizer_expected = 'adam'
    epochs_expected = 2
    batch_size_expected = 16
    build = False
    clf = classifier.Classifier(classifier_name=classifier_name_expected,
                                num_classes=num_classes_expected,
                                output_directory=tmp_path,
                                monitor=monitor_expected,
                                loss=loss_expected,
                                optimizer=optimizer_expected,
                                epochs=epochs_expected,
                                batch_size=batch_size_expected,
                                build=build)



    # ACT
    model_out = clf.build_classifier()
    keras.backend.clear_session()

    # ASSERT
    assert isinstance(model_out, keras.Model)


def test__build_classifier_errors(tmp_path):
    # ARRANGE
    classifier_name_expected = 'dummy_name'
    num_classes_expected = 3
    monitor_expected = 'loss'
    loss_expected = 'categorical_crossentropy'
    optimizer_expected = 'adam'
    epochs_expected = 2
    batch_size_expected = 16
    build = False
    clf = classifier.Classifier(classifier_name=classifier_name_expected,
                                num_classes=num_classes_expected,
                                output_directory=tmp_path,
                                monitor=monitor_expected,
                                loss=loss_expected,
                                optimizer=optimizer_expected,
                                epochs=epochs_expected,
                                batch_size=batch_size_expected,
                                build=build)

    # ACT
    with pytest.raises(AssertionError):
        clf.build_classifier()

# def test__eval_classifications(tmp_path):
#     """
#     Test eval_classifications function of class Classifier
#     """
#     # ARRANGE
#     classifier_name = "fcn"
#     num_classes = 2
#     clf = classifier.Classifier(classifier_name=classifier_name,
#                                 num_classes=2,
#                                 output_directory=tmp_path)
#
#     probabilities = np.array([[0.1, 0.7], [0.5, 0.2], [0.3, 0.9]])
#
#     df_results_expected = pd.DataFrame([0])
#     df_results_expected["classifier_name"] = classifier_name
#     df_results_expected["balanced_accuracy"] = 0.75
#     df_results_expected["bd_rate"] = 0.5,
#     df_results_expected["roc_auc_score"] = 1.0,
#     df_results_expected["f1_score"] = 2 / 3,
#     df_results_expected["train_time"] = 2,
#     df_results_expected["cm_tp"] = 1,
#     df_results_expected["cm_tn"] = 1,
#     df_results_expected["cm_fp"] = 1,
#     df_results_expected["cm_fn"] = 0,
#     df_results_expected["n_train_healthy"] = 2,
#     df_results_expected["n_train_bd"] = 1,
#     df_results_expected["n_test_healthy"] = 1,
#     df_results_expected["n_test_bd"] = 2,
#     df_results_expected["class_imbalance_train"] = 2.0,
#     df_results_expected["class_imbalance_test"] = 0.5
#
#     # Write to CSV and read from CSV due to simulate behaviour in eval_classifications
#     data_path_expected = tmp_path / "df_metrics_expected.csv"
#     df_results_expected.to_csv(data_path_expected)
#     df_results_expected = pd.read_csv(data_path_expected)
#
#     y_test = np.array([False, False, True])
#
#     # ACT
#     clf.train_time = 2
#     clf.eval_classifications(y_test, probabilities)
#     data_path = tmp_path / "df_metrics.csv"
#     df_results = pd.read_csv(data_path)
#
#     # ASSERT
#     assert_frame_equal(df_results, df_results_expected)

# def test__fcn(tmp_path):
#     """
#     Test FCN model of class Classifier
#     """
#     # ARRANGE
#     data = namedtuple("data", ["X", "y", "idx"])
#
#     train = data(np.array(range(10)), np.zeros(1), np.zeros(1))
#     valid = data(np.zeros(range(10, 20)), np.zeros(1), np.zeros(1))
#
#     classifier_name = "fcn"
#     clf = classifier.Classifier(classifier_name=classifier_name,
#                                 output_directory=tmp_path)
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     before = sess.run(tf.trainable_variables())
#     _ = sess.run(model.train, feed_dict={
#         X_temp: np.ones((1, 1, 100, 1)),
#     })
#     after = sess.run(tf.trainable_variables())
#     for b, a, n in zip(before, after):
#         assert (b != a).any()
