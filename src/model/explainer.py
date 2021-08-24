from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from src.model.classifier import Classifier



class Explainer(ABC):
    """
    abstract class which acts as a template to create datasets

    def __init__(self, build=True):
        if build:
            self.model = self.build_explainer()
        super().__init__()
    """

    @abstractmethod
    def build_explainer(self, classifier: Classifier, X_train: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """

    @abstractmethod
    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """

    @abstractmethod
    def get_sample_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """




def explain_samples(explainer: Explainer,
                    classifier: Classifier,
                    X_train: np.ndarray,
                    X_sample: np.ndarray) -> np.ndarray:
    """
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param hdf_dir: input directory with hdf files
    :return: train, valid, test: tuple with data of type named tuple
    """




    explainer.build_explainer(classifier=classifier, X_train=X_train)
    sample_importance = explainer.get_sample_importance(X_sample[1:])
    # feature_importance = explainer.get_feature_importance(X_sample)
    print("asd")
