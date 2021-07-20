"""
tests the main module
"""
import numpy as np
from src import main

def test__scale_data():
    # ARRANGE

    X = np.array([[[0, 0, 0], [1, 1, 1]]])
    X_expected = np.array([[[-1, -1, -1], [1, 1, 1]]])

    # ACT
    X_output = main.scale_data(X)

    # ASSERT
    assert (X_output == X_expected).all()