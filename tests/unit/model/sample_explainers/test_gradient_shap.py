import pytest
import numpy as np
import tensorflow as tf
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
    background = X_reference[np.random.choice(X_reference.shape[0], 100, replace=False)]
    shap_explainer_expected = shap.GradientExplainer(model, background)

    # ACT
    explainer.build_explainer(model, X_reference)

    # ASSERT
    assert hasattr(explainer, "shap_explainer")
    assert explainer.shap_explainer == shap_explainer_expected


@pytest.mark.skip(reason="not finished")
def test__get_sample_importance():
    """
    Function for testing get_sample_importance method of ShapGradientExplainer
    """
    # ARRANGE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 2)),
        tf.keras.layers.Dense(2, activation='relu')
    ])
    X_reference = np.ones(shape=(200, 4))
    explainer = gradient_shap.ShapGradientExplainer()
    X_to_explain = np.ones(shape=(200, 4))

    sample_importance_expected = explainer.shap_explainer.shap_values(X_to_explain)
    # ACT
    explainer.build_explainer(model, X_reference)
    sample_importance = explainer.get_sample_importance(X_to_explain)

    # ASSERT
    assert sample_importance == sample_importance_expected
