import shap
from shap.explainers._gradient import Gradient
import numpy as np
from src.model.explainer import ExplainerCreator
from tensorflow.keras import Model

class ShapGradientExplainer(ExplainerCreator):
    """
    Subclass of ExplainerCreator to explain predictions of tensorflow functional API models with shap gradient
    explainer.
    """
    shap_explainer: Gradient

    def build_explainer(self, model: Model, X_reference: np.ndarray) -> np.ndarray:
        """
        method to select events for dataset
        :param model: tensorflow functional API model
        :param X_reference: array of data which should be explained
        """
        background = X_reference[np.random.choice(X_reference.shape[0], 100, replace=False)]
        self.shap_explainer = shap.GradientExplainer(model, background)

    def get_sample_importance(self, X_to_explain: np.ndarray) -> np.ndarray:
        """
        method to select events for dataset
        :param X_to_explain: data which should be explained
        :return: shap_values: list of arrays with importance for each label
        """
        shap_values = self.shap_explainer.shap_values(X_to_explain)
        return shap_values
