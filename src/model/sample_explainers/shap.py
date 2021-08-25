import shap
from shap.explainers._gradient import Gradient
import numpy as np
from src.model.explainer import Explainer
from src.model.classifier import Classifier

class Shap_Explainer(Explainer):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """
    shap_explainer: Gradient

    def build_explainer(self, classifier: Classifier, X_train: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        self.shap_explainer = shap.DeepExplainer(classifier.model, background)  # DeepExplainer vs. GradientExplainer

    def get_sample_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """
        shap_values = self.shap_explainer.shap_values(X_sample)
        return shap_values

    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """