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


def main(config_name: str, num_simulations=1000):

    config = load_config(config_name, verbose=False)

    validate_config(config)

    run_dir = create_run_dir(config_name, author="nitesh", base_dir="results")

    save_config(config, run_dir)

    df_base = generate_environment(config)

    df_base["oracle_churn_score"] = 1 - df_base["oracle_p0"]

    print(
        "spearman correlation between oracle_churn_score and oracle_tau is:",
        df_base[["oracle_churn_score", "oracle_tau"]]
        .corr(method="spearman")["oracle_tau"]
        .iloc[0]
        .round(3),
    )

    save_df(df_base, "0_df_base", run_dir)

    fig1 = plot_feature_space(
        df_base,
        x_col="engagement_score",
        y_col="price_sensitivity",
        color_col="oracle_tau",
    )
    fig2 = plot_feature_space(
        df_base,
        x_col="engagement_score",
        y_col="price_sensitivity",
        color_col="oracle_churn_score",
    )

    save_plot(fig1, "1_tau_distribution", run_dir)
    save_plot(fig2, "2_churn_distribution", run_dir)

    policies = config.get("policies")

    result = run_simulations(df_base, policies, num_simulations)
    final_result_df = make_result_df(result)

    save_df(final_result_df, "1_final_result_df", run_dir)

    df_long = get_df_long(
        final_result_df, id_vars="run_id", var_name="policy", value_name="outcome"
    )

    save_df(df_long, "2_df_long", run_dir)

    summary_df = get_summary(df_long).reset_index()
    save_df(summary_df, "3_summary_df", run_dir)

    df_modal_winner = policy_rank_modal(df_long, exclude=["treat_all"])

    save_df(df_modal_winner, "4_df_modal_winner", run_dir)

    fig3 = plot_policy_distribution(df_long, metric="outcome", order=None)
    fig4 = plot_policy_ecdf(df_long, metric="outcome", order=None)

    save_plot(fig3, "3_policy_distribution", run_dir)
    save_plot(fig4, "4_policy_ecdf", run_dir)


if __name__ == "__main__":
    main()
