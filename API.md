# API Documentation


## `__init__.py`


## `environment.py`

### `generate_environment(config)`
generates a dataframe containing user_id, their features, baseline and treatemnt churn indices, baseline and treatment churn probabilities, treatment effect.


## `evaluation.py`

### `run_simulations(df, policies, num_simulations)`

### `make_result_df(lst)`

### `get_df_long(df, id_vars, var_name, value_name)`

### `get_summary(df)`
assert df in long format

### `policy_rank_modal(df, exclude)`
tells how many times each policy wins. 


## `features.py`

### `generate_features(config)`
generates features following specification in config 


## `generate_doc.py`

### `generate_doc()`


## `outcome_runner.py`

### `apply_realised_outcomes(df, policy_cols, Y0_col, Y1_col)`
wrapper 


## `outcomes.py`

### `realised_outcome(df, policy_col, y0_col, y1_col)`
Generate realized outcomes for a given policy.


## `plotting.py`

### `plot_policy_distribution(df_long, metric, order)`
Boxplot of policy performance with mean markers.

### `plot_policy_ecdf(df_long, metric, order)`
ECDF plot to compare distribution dominance across policies.


## `policy.py`

### `assign_policy(df, policy, K, score_col, seed)`
Assign treatment according to specified policy.


## `policy_runner.py`

### `apply_policies(df, policies, seed)`
wrapper


## `potential_outcomes.py`

### `draw_potential_outcomes(df, seed)`
Draw potential outcomes using a single  uniform shock.


## `utils.py`

### `load_config(config_name, verbose)`
read config from configs folder. prints config if verbose is not False.

### `save_config(config, run_dir)`
saves this config in the run directory for results folder.

### `validate_config(config)`
validates if config is valid. keep adding validation as config grows complex.

### `create_run_dir(name, author, base_dir)`
creates a directory of the form {config_name}_{author}_{timestamp} in the results folder. All artifacts from this run will be stored in this directory.

### `clear_directory(path)`
clears all run directories inside results folder. Use it before pushing to git.

### `sigmoid(t)`
returns logistic output

### `linear_index(df, coeffs, features)`
returns the dot product of features and coefficients

### `get_columns(policies)`

### `save_df(df, name, run_dir)`

### `save_plot(fig, name, run_dir)`
Save a matplotlib Figure object to run_dir/figures as a JPG.

API.md last updated at 2026-03-26_17-17-13.