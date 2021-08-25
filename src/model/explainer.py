from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras import Model


class ExplainerCreator(ABC):
    """
    abstract class which acts as a template to create explainers
    """
    @abstractmethod
    def build_explainer(self, model: Model, X_reference: np.ndarray) -> np.ndarray:
        """
        abstract method to build explainer
        """

    @abstractmethod
    def get_sample_importance(self, X_to_explain: np.ndarray) -> list:
        """
        abstract method to use explainer to get important samples
        """


def explain_samples(explainer: ExplainerCreator,
                    model: Model,
                    X_reference: np.ndarray,
                    X_to_explain: np.ndarray) -> np.ndarray:
    """
    :param explainer: any concrete subclass of Explainer to explain prediction
    :param model: tensorflow model of type functional API
    :param X_reference: data, used as reference for explanation
    :param X_to_explain: data which should be explained
    :return: sample_importance: list of arrays with importance for each label
    """
    explainer.build_explainer(model=model, X_reference=X_reference)
    sample_importance = explainer.get_sample_importance(X_to_explain)
    return sample_importance
