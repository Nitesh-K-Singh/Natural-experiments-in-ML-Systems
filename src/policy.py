import numpy as np


def assign_policy(df, policy='random', K=None, score_col=None, seed=None):
    '''
    Assign treatment according to specified policy.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing users and features.
    policy : str
        Policy type: 'random', 'oracle', 'score', 'treat_all', 'treat_none'
    K : int
        Number of users to treat (budget constraint).
    score_col : str
        Column used for ranking (required for 'score').
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with column 'D' (treatment indicator).
    '''

    if seed is not None:
        np.random.seed(seed)

    df = df.copy()
    N = len(df)

    # ---------- treat all ----------
    if policy == 'treat_all':

        K = N

        col = f"D_{policy}"
        df[col] = 1
        return df


    # ---------- treat none ----------
    if policy == 'treat_none':

        K = 0

        col = f"D_{policy}"
        df[col] = 0
        return df


    # ---------- random targeting ----------
    if policy == 'random':

        if K is None:
            raise ValueError('K must be specified for random policy')

        D = np.zeros(N, dtype=int)
        treated = np.random.choice(N, K, replace=False)

        D[treated] = 1
        col = f"D_{policy}"
        df[col] = D
      

        return df


    # ---------- treatment effect targeting ----------
    if policy == 'causal':

        if 'tau' not in df.columns:
            raise ValueError(" causal policy requires column 'tau'")

        if K is None:
            raise ValueError('K must be specified')

        df = df.sort_values('tau', ascending=False)

        col = f"D_{policy}"
        df[col] = 0
        df.iloc[:K, df.columns.get_loc(col)] = 1

        df = df.sort_index()

        return df


    # ---------- score-based policy ----------
    if policy == 'score':

        if score_col is None:
            raise ValueError('score_col must be provided')

        if K is None:
            raise ValueError('K must be specified')

        if score_col not in df.columns:
            raise ValueError(f'Column {score_col} not found')

        df = df.sort_values(score_col, ascending=False)

        col = f"D_{policy}"
        df[col] = 0
        df.iloc[:K, df.columns.get_loc(col)] = 1

        df = df.sort_index()

        return df


    raise ValueError(f'Unknown policy: {policy}')