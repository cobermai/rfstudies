import numpy as np
from src.model.explainer import ExplainerCreator
from tensorflow.keras import Model, layers
from tensorflow import keras
# from src.model.concept_explainers import toy_helper_v2
from src.model.concept_explainers import ipca_v2

class ShapConceptExplainer:
    """
    Subclass of ExplainerCreator to explain predictions of tensorflow functional API models with shap gradient
    explainer.
    """
    def seperate_model(self, model: Model):
        """
        seperate last layer from model and return two seper
        :param model: tensorflow functional API model
        :param X_reference: array of data which should be explained
        """
        num_layers = len(model.layers)

        feature_model = Model(inputs=model.input,
                              outputs=model.layers[num_layers - 2].output)

        pred_model_shape = model.layers[num_layers - 2].output.shape
        pred_model_shape = pred_model_shape[1:]  # Remove Batch from front.

        pred_model_input = layers.Input(shape=pred_model_shape)
        x = pred_model_input
        for layer in model.layers[num_layers - 1:]:  # idx + 1
            x = layer(x)

        pred_model = Model(pred_model_input, x)

        return feature_model, pred_model

    def explain(self, model, train, valid, test, output_dir) -> np.ndarray:

        feature_model, pred_model = self.seperate_model(model)

        f_train = feature_model.predict(train.X)
        f_test = feature_model.predict(test.X)

        n_concept = 5

        topic_model_pr, optimizer_reset, optimizer, topic_vector,  n_concept, \
        f_input = ipca_v2.topic_model_new_toy(pred_model,
                                              f_train[..., np.newaxis],
                                              train.y,
                                              f_test[..., np.newaxis],
                                              test.y,
                                              n_concept)

        print('asd')
