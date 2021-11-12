import pytest
import numpy as np
import tensorflow as tf
import shap
from shap.explainers._deep import Deep
from src.model.sample_explainers import deep_shap


def test__ShapDeepExplorer():
    """
    Function for testing instantiation of ShapDeepExplorer class
    """
    # ARRANGE

    # ACT
    explainer = deep_shap.ShapDeepExplainer()

    # ASSERT
    assert hasattr(explainer, "build_explainer")
    assert hasattr(explainer, "get_sample_importance")


def test__build_explainer():
    """
    Function for testing build_explainer method of ShapDeepExplainer
    """
    # ARRANGE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 2)),
        tf.keras.layers.Dense(2, activation='relu')
    ])
    X_reference = np.ones(shape=(200, 2, 2))
    explainer = deep_shap.ShapDeepExplainer()
    background_size = 100
    background = X_reference[np.random.choice(X_reference.shape[0], background_size, replace=False)]
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer_model_expected = shap.DeepExplainer(model, background)

    # ACT
    explainer_model_out = explainer.build_explainer(model, X_reference)

    # ASSERT
    assert(type(explainer_model_out) is type(explainer_model_expected))


@pytest.mark.skip(reason="not finished")
def test__get_sample_importance():
    """
    Function for testing get_sample_importance method of ShapDeepExplainer
    """
    # ARRANGE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 2)),
        tf.keras.layers.Dense(2, activation='relu')
    ])
    X_reference = np.ones(shape=(200, 2, 2))
    explainer = deep_shap.ShapDeepExplainer()
    explainer_model = explainer.build_explainer(model, X_reference)
    X_to_explain = np.ones(shape=(200, 2, 2))
    sample_importance_expected = explainer_model = explainer_model.shap_values(X_to_explain)

    # ACT
    sample_importance = explainer.get_sample_importance(explainer_model, X_to_explain)

    # ASSERT
    comparison_list = np.array([sample_importance[i] == sample_importance_expected[i]
                                for i in range(len(sample_importance))])
    assert np.all(comparison_list)

