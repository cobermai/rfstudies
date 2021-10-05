import typing
import shap
import numpy as np
from tensorflow.keras import Model
from src.model.explainer import ExplainerCreator


class ShapGradientExplainer(ExplainerCreator):
    """
    Subclass of ExplainerCreator to explain predictions of tensorflow functional API models with shap gradient
    explainer.
    """

    @staticmethod
    def build_explainer(model: Model, X_reference: np.ndarray) -> typing.Any:
        """
        Method to build model explainer
        :param model: tensorflow functional API model
        :param X_reference: array of data which should be explained
        :return: shap deep explainer class
        """
        background_size = 100
        background = X_reference[np.random.choice(X_reference.shape[0], background_size, replace=False)]
        return shap.GradientExplainer(model, background)

    @staticmethod
    def get_sample_importance(explainer_model: typing.Any,
                              data_to_explain: np.ndarray) -> typing.Union[np.ndarray, list]:
        """
        Method to get sample importance values
        :param explainer_model: explainable AI model
        :param data_to_explain: data which should be explained
        :return: shap_values: list of arrays with importance for each label
        """
        shap_values = explainer_model.shap_values(data_to_explain)
        return shap_values
