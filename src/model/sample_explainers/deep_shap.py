import typing
import shap
from shap.explainers._deep import Deep
import numpy as np
from tensorflow.keras import Model
from src.model.explainer import ExplainerCreator


class ShapDeepExplainer(ExplainerCreator):
    """
    Subclass of ExplainerCreator to explain predictions of tensorflow functional API models with shap deep explainer.
    """
    shap_explainer: Deep

    def build_explainer(self, model: Model, X_reference: np.ndarray):
        """
        Method to build model explainer
        :param model: tensorflow functional API model
        :param X_reference: array of data which should be explained
        """
        background = X_reference[np.random.choice(X_reference.shape[0], 100, replace=False)]
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        self.shap_explainer = shap.DeepExplainer(model, background)

    def get_sample_importance(self, X_to_explain: np.ndarray) -> typing.Union[np.ndarray, list]:
        """
        Method to get sample importance values
        :param X_to_explain: data which should be explained
        :return: shap_values: list of arrays with importance for each label
        """
        shap_values = self.shap_explainer.shap_values(X_to_explain)
        return shap_values
