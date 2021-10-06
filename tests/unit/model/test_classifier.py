import os.path
import pytest
from tensorflow import keras
from src.model import classifier



@pytest.mark.parametrize("build, output_model_structure, model_expected, plot_existence_expected",
                         [(True, True, True, True),
                          (False, False, False, False),
                          (False, False, False, False),
                          (False, True, False, False),
                          (True, False, True, False)]
                         )
def test__classifier(tmp_path, build, output_model_structure, model_expected, plot_existence_expected):
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
    learning_rate = 1e-3
    reduce_lr_factor = 0.5
    reduce_lr_patience = 50
    min_lr = 0.0001
    input_shape = (16, 8, 4)

    # ACT
    clf = classifier.Classifier(classifier_name=classifier_name_expected,
                                num_classes=num_classes_expected,
                                output_directory=tmp_path,
                                monitor=monitor_expected,
                                loss=loss_expected,
                                optimizer=optimizer_expected,
                                epochs=epochs_expected,
                                batch_size=batch_size_expected,
                                build=build,
                                learning_rate=learning_rate,
                                reduce_lr_factor=reduce_lr_factor,
                                reduce_lr_patience=reduce_lr_patience,
                                min_lr=min_lr,
                                input_shape=input_shape,
                                output_model_structure=output_model_structure
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
    assert os.path.isfile(clf.output_directory / "plot_model_structure.png") == plot_existence_expected


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
    learning_rate = 1e-3
    reduce_lr_factor = 0.5
    reduce_lr_patience = 50
    min_lr = 0.0001
    input_shape = (16, 8, 4)
    clf = classifier.Classifier(classifier_name=classifier_name_expected,
                                num_classes=num_classes_expected,
                                output_directory=tmp_path,
                                monitor=monitor_expected,
                                loss=loss_expected,
                                optimizer=optimizer_expected,
                                epochs=epochs_expected,
                                batch_size=batch_size_expected,
                                build=build,
                                learning_rate=learning_rate,
                                reduce_lr_factor=reduce_lr_factor,
                                reduce_lr_patience=reduce_lr_patience,
                                min_lr=min_lr,
                                input_shape=input_shape
                                )

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
    learning_rate = 1e-3
    reduce_lr_factor = 0.5
    reduce_lr_patience = 50
    min_lr = 0.0001
    input_shape = (16, 8, 4)
    clf = classifier.Classifier(classifier_name=classifier_name_expected,
                                num_classes=num_classes_expected,
                                output_directory=tmp_path,
                                monitor=monitor_expected,
                                loss=loss_expected,
                                optimizer=optimizer_expected,
                                epochs=epochs_expected,
                                batch_size=batch_size_expected,
                                build=build,
                                learning_rate=learning_rate,
                                reduce_lr_factor=reduce_lr_factor,
                                reduce_lr_patience=reduce_lr_patience,
                                min_lr=min_lr,
                                input_shape=input_shape
                                )

    # ACT
    with pytest.raises(AssertionError):
        clf.build_classifier()
