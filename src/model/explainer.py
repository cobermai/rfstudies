import typing
from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras import Model


class ExplainerCreator(ABC):
    """
    abstract class which acts as a template to create explainers
    """

    @staticmethod
    @abstractmethod
    def build_explainer(model: Model, X_reference: np.ndarray) -> typing.Any:
        """
        abstract method to build explainer
        """

    @staticmethod
    @abstractmethod
    def get_sample_importance(explainer_model: typing.Any,
                              data_to_explain: np.ndarray) -> typing.Union[np.ndarray, list]:
        """
        abstract method to use explainer to get important samples
        """


def explain_samples(explainer: ExplainerCreator,
                    model: Model,
                    X_reference: np.ndarray,
                    X_to_explain: np.ndarray) -> typing.Union[np.ndarray, list]:
    """
    :param explainer: any concrete subclass of Explainer to explain prediction
    :param model: tensorflow model of type functional API
    :param X_reference: data, used as reference for explanation
    :param X_to_explain: data which should be explained
    :return: sample_importance: list of arrays with importance for each label
    """
    built_explainer = explainer.build_explainer(model=model, X_reference=X_reference)
    sample_importance = explainer.get_sample_importance(explainer_model=built_explainer, data_to_explain=X_to_explain)
    return sample_importance
