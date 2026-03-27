import numpy as np
import pandas as pd

from src.features import generate_features
from src.utils import sigmoid, linear_index


def generate_environment(config):
    """
    generates a dataframe containing user_id, their features, baseline and treatemnt churn indices, baseline and treatment churn probabilities, treatment effect.
    """

    np.random.seed(config.get("seed", 16))

    I = config["I"]

    features = generate_features(config)

    df = pd.DataFrame({"user_id": range(1, I + 1), **features})

    feature_names = list(features.keys())

    df["baseline_index"] = linear_index(
        df, config["outcome_model"]["baseline_coeff"], feature_names
    )

    df["treatment_index"] = linear_index(
        df, config["outcome_model"]["treatment_coeff"], feature_names
    )

    df["oracle_p0"] = sigmoid(df["baseline_index"])
    df["oracle_p1"] = sigmoid(df["baseline_index"] + df["treatment_index"])

    df["oracle_tau"] = (
        df["oracle_p1"] - df["oracle_p0"]
    )  # to be precise this is expectation of tau

    return df
