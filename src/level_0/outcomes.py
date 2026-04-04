def realised_outcome(df, policy_col, y0_col="Y0", y1_col="Y1"):
    """
    Generate realized outcomes for a given policy.

    Parameters
    ----------
    df : DataFrame
    policy_col : column containing treatment assignment
    y0_col : column containing Y0
    y1_col : column containing Y1
    """

    required = {policy_col, y0_col, y1_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    outcome_col = f"Y_{policy_col[2:]}" if policy_col.startswith("D_") else "Y"

    df[outcome_col] = df[policy_col] * df[y1_col] + (1 - df[policy_col]) * df[y0_col]

    return df
