import numpy as np
import pandas as pd


def draw_potential_outcomes(df, seed=None):
    """
    Draw potential outcomes using a single  uniform shock.

    Y0 = 1(U < p0)
    Y1 = 1(U < p1)

    This guarantees monotonic outcomes when p1 > p0.
    """

    columns = df.columns
    assert {"oracle_p0", "oracle_p1"}.issubset(
        columns
    ), "df must contain columns 'p0' and 'p1'"
    # for later, better to take p0,p1 columns as input

    if seed is not None:
        np.random.seed(seed)

    df_out = df.copy()

    # latent shock
    df_out["U"] = np.random.uniform(0, 1, len(df))

    # potential outcomes
    df_out["Y0"] = (df_out["U"] < df["oracle_p0"]).astype(int)
    df_out["Y1"] = (df_out["U"] < df["oracle_p1"]).astype(int)

    return df_out
