from src.policy import assign_policy


def apply_policies(df, policies, seed=None):
    """wrapper"""
    df = df.copy()

    for p in policies:

        df = assign_policy(
            df, policy=p["name"], K=p.get("K"), score_col=p.get("score_col"), seed=seed
        )

    return df
