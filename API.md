# API Documentation


## `__init__.py`


## `environment.py`

### `generate_environment(config)`
generates a dataframe containing user_id, their features, baseline and treatemnt churn indices, baseline and treatment churn probabilities, treatment effect.


## `evaluation.py`

### `run_simulations(df, policies, num_simulations)`

### `make_result_df(lst)`


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

### `create_run_dir(config_name, author, base_dir)`
creates a directory of the form {config_name}_{author}_{timestamp} in the results folder. All artifacts from this run will be stored in this directory.

### `clear_directory(path)`
clears all run directories inside results folder. Use it before oushing to git.

### `sigmoid(t)`
returns logistic output

### `linear_index(df, coeffs, features)`
returns the dot product of features and coefficients

### `get_columns(policies)`

API.md last updated at 2026-03-23_15-42-23.