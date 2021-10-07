import numpy as np
from src.model.explainer import ExplainerCreator
from tensorflow.keras import Model, layers
from tensorflow import keras
# from src.model.concept_explainers import toy_helper_v2
from src.model.concept_explainers import ipca_v2
import matplotlib.pyplot as plt

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

        n_concept = 5
        batch_size = 100
        verbose = True
        thres = 0.2

        feature_model, pred_model = self.seperate_model(model)

        f_train_2d = feature_model.predict(train.X)
        f_test_2d = feature_model.predict(test.X)

        f_train = np.expand_dims(f_train_2d, axis=(1, 2))
        f_test = np.expand_dims(f_test_2d, axis=(1, 2))



        topic_model_pr, optimizer_reset, optimizer, topic_vector,  n_concept, \
        f_input = ipca_v2.topic_model_new_toy(pred_model,
                                              f_train,
                                              train.y,
                                              f_test,
                                              test.y,
                                              n_concept)


        topic_model_pr.fit(
            f_train,
            train.y,
            batch_size=batch_size,
            epochs=30,
            validation_data=(f_test, test.y),
            verbose=verbose)

        topic_vec = topic_model_pr.layers[1].get_weights()[0]
        topic_vec_n = topic_vec/(np.linalg.norm(topic_vec, axis=0, keepdims=True)+1e-9)

        ipca_v2.get_completeness(pred_model,
                           f_train,
                           train.y,
                           f_test,
                           test.y,
                           n_concept,
                           topic_vec_n[:,:n_concept],
                           verbose=verbose,
                           epochs=10,
                           metric1=['binary_accuracy'],
                           loss1=keras.losses.binary_crossentropy,
                           thres=thres,
                           load='toy_data/latest_topic_toy.h5')

        # visualize the nearest neighbors
        x = train.X
        f_train_n = f_train / (np.linalg.norm(f_train, axis=3, keepdims=True) + 1e-9)
        topic_vec_n = topic_vec / (np.linalg.norm(topic_vec, axis=0, keepdims=True) + 1e-9)
        topic_prob = np.matmul(f_train_n, topic_vec_n)
        n_size = 4
        for i in range(n_concept):
            ind = np.argpartition(topic_prob[:, :, :, i].flatten(), -10)[-10:]
            sim_list = topic_prob[:, :, :, i].flatten()[ind]
            for jc, j in enumerate(ind):
                j_int = int(np.floor(j / (n_size * n_size)))
                a = int((j - j_int * (n_size * n_size)) / n_size)
                b = int((j - j_int * (n_size * n_size)) % n_size)
                f1 = '/volume00/jason/concept_stm/work_toy_test/concept_full_{}_{}.png'.format(i, jc)
                f2 = '/volume00/jason/concept_stm/work_toy_test/concept_{}_{}.png'.format(i, jc)
                # if sim_list[jc]>0.95:
                plt.plot(x[j_int, :, :, :],)
                plt.savefig("concept_{}_{}.png".format(i, jc))



