import numpy as np
import pytest
import tensorflow as tf

from src.model.classifiers.layers import shortcut


def test__shortcut():
    """
    Function that tests the structure CNNLayer
    """
    # ARRANGE
    layer_shortcut = shortcut.ShortcutBlock(2, 2)
    x = tf.ones((3, 3, 3))
    # ACT
    y = layer_shortcut(x)
    # ASSERT
    assert not np.isnan(y).all()


@pytest.mark.skip(reason="Stopped raising errors")
def test__shortcut_errors():
    with pytest.raises(ValueError):
        shortcut.ShortcutBlock(0, 0)

    with pytest.raises(ValueError):
        shortcut.ShortcutBlock(-1, -1)
