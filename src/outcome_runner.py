from src.outcomes import realised_outcome


def apply_realised_outcomes(df, policy_cols, Y0_col="Y0", Y1_col="Y1"):
    """wrapper"""
    df = df.copy()

    for D_col in policy_cols:
        df = realised_outcome(df, D_col, Y0_col, Y1_col)

    return df
