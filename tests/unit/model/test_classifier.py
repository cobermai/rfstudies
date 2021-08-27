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
