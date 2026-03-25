import sys
sys.path.append("..")
from src.utils import *
from src.features import *
from src.environment import *
from src.potential_outcomes import *
from src.policy import *
from src.outcomes import *
from src.policy_runner import *
from src.outcome_runner import *
from src.generate_doc import *
from src.evaluation import *
from src.plotting import *




def main(config_name= 'retention', num_simulations = 1000 ):

    config = load_config(config_name, verbose = False)

    validate_config(config)

    run_dir = create_run_dir('notebook_1', author='nitesh', base_dir='results' )

    save_config(config, run_dir)

    df_base = generate_environment(config)

    save_df(df_base, 'df_base' ,run_dir)

    policies = config.get('policies')
    
   

    result = run_simulations( df_base, policies, num_simulations )
    final_result_df = make_result_df(result)

    save_df(final_result_df, 'final_result_df' , run_dir)

    df_long = get_df_long(final_result_df, id_vars = 'run_id', var_name='policy', value_name='outcome' )
    
    save_df(df_long, 'df_long' , run_dir)

    summary_df = get_summary(df_long)
    save_df(summary_df, 'summary_df' , run_dir)
    
    df_modal_winner = policy_rank_modal(df_long)
    df_modal_winner_minus_treat_all =    policy_rank_modal(df_long, exclude = ['treat_all'])

    save_df(df_modal_winner, 'df_modal_winner' , run_dir)
    save_df(df_modal_winner_minus_treat_all, 'df_modal_winner_minus_treat_all' , run_dir)

    fig1 = plot_policy_distribution(df_long, metric='outcome', order = None)
    fig2 = plot_policy_ecdf(df_long, metric='outcome', order = None)
    


    save_plot(fig1, "policy_distribution", run_dir)
    save_plot(fig2, "policy_ecdf", run_dir)


if __name__ == "__main__":
    main()