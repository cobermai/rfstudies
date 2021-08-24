import shap
import numpy as np
from src.model.explainer import Explainer
from src.model.classifier import Classifier

class Shap_Explainer(Explainer):
    """
    Subclass of DatasetCreator to specify dataset selection. None of the abstract functions from abstract class can
    be overwritten.
    """

    def build_explainer(self, classifier: Classifier, X_train: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        # explain predictions of the model on three images
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        model = classifier.model
        e = shap.DeepExplainer(model, background)
        return e

    def get_sample_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """
        shap_values = self.model.shap_values(X_sample)
        return shap_values

    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        abstract method to select events for dataset
        """