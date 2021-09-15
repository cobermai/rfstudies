from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
import pytest
from tensorflow import keras
from src.model import explainer


def test__ExplainerCreator():
    # ARRANGE
    p = patch.multiple(explainer.ExplainerCreator, __abstractmethods__=set())

    # ACT
    p.start()
    explainer_instance = explainer.ExplainerCreator()
    p.stop()

    # ASSERT
    assert hasattr(explainer_instance, "build_explainer")
    assert hasattr(explainer_instance, "get_sample_importance")


@pytest.mark.skip(reason="not finished")
@patch.multiple(explainer.ExplainerCreator, __abstractmethods__=set(),
                get_sample_importance=MagicMock(return_value=np.ones((10,), dtype=bool)),
                )
def test__explain_samples():
    # ARRANGE
    explainer_instance = explainer.ExplainerCreator()
    model = MagicMock(return_value=np.ones((10,), dtype=keras.Model))
    X_to_reference = MagicMock(return_value=np.ones((10,), dtype=np.ndarray))
    X_to_explain = MagicMock(return_value=np.ones((10,), dtype=np.ndarray))

    sample_importance_expected = np.ones((10,))

    # ACT
    sample_importance = explainer.explain_samples(explainer_instance, model, X_to_reference, X_to_explain)

    # ASSERT
    sample_importance == sample_importance_expected
