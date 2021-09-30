import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
from src.model.sample_explainers import gradient_shap


def test__ShapGradientExplorer():
    # ARRANGE

    # ACT
    explainer = gradient_shap.ShapGradientExplainer()

    # ASSERT
    assert hasattr(explainer, "build_explainer")
    assert hasattr(explainer, "get_sample_importance")


@pytest.mark.skip(reason="not finished")
def test__build_explainer():
    """
    Function for testing build_explainer method of ShapGradientExplainer
    """
    # ARRANGE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 2)),
        tf.keras.layers.Dense(2, activation='relu')
    ])
    X_reference = np.ones(shape=(200, 4))
    explainer = gradient_shap.ShapGradientExplainer()

    background_size = 100
    background = X_reference[np.random.choice(X_reference.shape[0], background_size, replace=False)]
    explainer_model_expected = shap.GradientExplainer(model, background)

    # ACT
    explainer_model_out = explainer.build_explainer(model, X_reference)

    # ASSERT
    assert(explainer_model_out == explainer_model_expected)


@pytest.mark.skip(reason="not finished")
def test__get_sample_importance():
    """
    Function for testing get_sample_importance method of ShapGradientExplainer
    """
    # ARRANGE
    keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 2)),
        tf.keras.layers.Dense(2, activation='relu')
    ])
    X_reference = np.ones(shape=(200, 2, 2))
    explainer = gradient_shap.ShapGradientExplainer()
    explainer_model = explainer.build_explainer(model, X_reference)
    X_to_explain = np.ones(shape=(200, 2, 2))
    sample_importance_expected = explainer_model.shap_values(X_to_explain)

    # ACT
    sample_importance = explainer.get_sample_importance(explainer_model, X_to_explain)

    # ASSERT
    assert len(sample_importance) == len(sample_importance_expected)
