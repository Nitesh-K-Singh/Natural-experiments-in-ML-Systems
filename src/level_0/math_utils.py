import numpy as np

# import pandas as pd


def sigmoid(t):
    """
    returns logistic output
    """
    return 1 / (1 + np.exp(-t))


def linear_index(df, coeffs, features):
    """
    returns the dot product of features and coefficients
    """
    intercept = coeffs[0]
    betas = coeffs[1:]
    return intercept + df[features].dot(betas)
