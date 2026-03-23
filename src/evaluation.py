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