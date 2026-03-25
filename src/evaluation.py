import numpy as np
import pandas as pd
from src.potential_outcomes import draw_potential_outcomes
from src.policy_runner import apply_policies
from src.outcome_runner import apply_realised_outcomes
from src.utils import get_columns

def run_simulations(df, policies, num_simulations):

    results = []
    policy_cols, outcome_cols = get_columns(policies)[0], get_columns(policies)[1] 
    

    for i in range(num_simulations):
        temp_df = df.copy(deep=True)                            # our features, theta, p0,p1 are same across simulations
        temp_df_po = draw_potential_outcomes(temp_df)                # this is our source of randomness across simulations
        temp_df_policies = apply_policies(temp_df_po, policies)      # this is same for all policies (except random)
        temp_df_outcomes = apply_realised_outcomes(temp_df_policies, policy_cols) # randomness here comes from randomness in potential outcomes (above)

        temp_result =  (temp_df_outcomes[outcome_cols].sum()).to_frame().T
        temp_result['run_id'] = i

        results.append(temp_result)

    return results


def make_result_df(lst):
    result = pd.concat(lst, ignore_index= True)
    result.columns = [col.replace('Y_', '') for col in result.columns]
    result = result[['run_id'] + [c for c in result.columns if c != 'run_id']]

    return result


def get_df_long(df, id_vars, var_name='policy', value_name='outcome'):
    if isinstance(id_vars, str):
        id_vars = [id_vars]

    return df.melt(
        id_vars=id_vars,
        var_name=var_name,
        value_name=value_name
    ).sort_values(id_vars)

def get_summary(df):

    '''assert df in long format'''

 

    summary = df.groupby('policy')['outcome'].agg(['mean','std','count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    summary['ci_lower'] = summary['mean'] - 1.96 * summary['se']   # can also get this via bootstrap
    summary['ci_upper'] = summary['mean'] + 1.96 * summary['se']   # can also get this via bootstrap

    return summary[['mean', 'ci_lower', 'ci_upper']]




def policy_rank_modal(df, exclude=None):
    '''tells how many times each policy wins. 
    Note sum of probabilities can be greater than 1  since
    ties are counted jointly
     '''
    df_rank = df.copy()

    if exclude is not None:
        df_rank = df_rank[~df_rank['policy'].isin(exclude)]

    df_rank['rank'] = df_rank.groupby('run_id')['outcome'] \
        .rank(ascending=False, method='first')

    counts = df_rank[df_rank['rank'] == 1]['policy'].value_counts()
    probs = counts / counts.sum()

    df_probs = probs.reset_index()  
    df_probs.columns = ['policy', 'probability']

    return df_probs