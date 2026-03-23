import sys
from src.config_utils import load_config, validate_config, save_config
from src.run_utils import create_run_dir
from src.environment import generate_environment
from src.potential_outcomes import *
from src.outcomes import *
from src.policy import *
from src.plotting import generate_plots


def main(config_name="retention"):

    config = load_config(config_name)

    validate_config(config)

    run_dir = create_run_dir(config_name)

    save_config(config, run_dir)

    # df_env = generate_environment(config)
    
    df_env = (
    generate_environment(config)
    .pipe(draw_potential_outcomes)
    .pipe(assign_policy, policy='treat_none')
    .pipe(realised_outcomes, 'D_treat_none')
    )



    generate_plots(df_env, run_dir, config)

    print("Run directory:", run_dir)
    print("Environment shape:", df_env.shape)


if __name__ == "__main__":
    main()