import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




## Config utils


def load_config(config_name, verbose = True):
    '''
    read config from configs folder. prints config if verbose is not False.
    '''
    
    config_path = Path("..") / "configs" / f"{config_name}.json"

    with open(config_path) as f:
        config = json.load(f)

   

    if verbose:
        print('Config loaded:')
        print(json.dumps(config, indent=2))

    return config


def save_config(config, run_dir):

    '''
    saves this config in the run directory for results folder.
    '''

    config_path = os.path.join(run_dir, 'arguments', 'config.json')

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def validate_config(config):

    '''
    validates if config is valid. keep adding validation as config grows complex.
    '''

    assert 'I' in config, "Missing key: I"
    assert 'features' in config, "Missing key: features"
    assert 'outcome_model' in config, "Missing key: outcome_model"

    features = config['features']
    outcome_model = config['outcome_model']
    objective = config.get('objective', [])

    M = len(features)

    # Validate feature distributions
    for name, spec in features.items():

        dist = spec['dist']

        if dist == 'normal':
            assert 'mean' in spec and 'std' in spec, \
                f"Feature '{name}' missing mean/std"

        elif dist == 'uniform':
            assert 'low' in spec and 'high' in spec, \
                f"Feature '{name}' missing low/high"

        elif dist == 'bernoulli':
            assert 'p' in spec, \
                f"Feature '{name}' missing p"

        else:
            raise ValueError(f"Unknown distribution: {dist}")

    baseline_coeff = outcome_model.get('baseline_coeff')
    treatment_coeff = outcome_model.get('treatment_coeff')

    assert baseline_coeff is not None, "baseline_coeff missing"
    assert treatment_coeff is not None, "treatment_coeff missing"

    assert len(baseline_coeff) == M + 1, \
        "baseline_coeff length must be M+1 (including intercept)"

    assert len(treatment_coeff) == M + 1, \
        "treatment_coeff length must be M+1 (including intercept)"

    if 'profit' in objective:
        assert len(config.get('profit_coeff', [])) == M + 1, \
            "profit_coeff must have length M+1"

    if 'cost' in objective:
        assert len(config.get('cost_coeff', [])) == M + 1, \
            "cost_coeff must have length M+1"

    return "config is valid"

    assert 'I' in config
    assert 'features' in config
    assert 'outcome_model' in config

    features = config['features']
    outcome_model = config['outcome_model']
    objective = config.get('objective', [])

    M = len(features)

    for name, spec in features.items():

        dist = spec['dist']

        if dist == 'normal':
            assert 'mean' in spec and 'std' in spec

        elif dist == 'uniform':
            assert 'low' in spec and 'high' in spec

        elif dist == 'bernoulli':
            assert 'p' in spec

        else:
            raise ValueError(f'Unknown distribution: {dist}')

    baseline_coeff = outcome_model.get('baseline_coeff')
    treatment_coeff = outcome_model.get('treatment_coeff')

    assert len(baseline_coeff) == M + 1
    assert len(treatment_coeff) == M + 1

    if 'profit' in objective:
        assert len(config.get('profit_coeff', [])) == M + 1

    if 'cost' in objective:
        assert len(config.get('cost_coeff', [])) == M + 1

    return ("config is valid")


# run utils

def create_run_dir(name : str, author='nitesh', base_dir='results'):

    '''
    creates a directory of the form {config_name}_{author}_{timestamp} in the results folder. All artifacts from this run will be stored in this directory.
    '''

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'{name}_{author}_{timestamp}'
    run_dir = os.path.join("..", base_dir, run_name)

    os.makedirs(os.path.join(run_dir, 'arguments'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)

    return run_dir


def clear_directory(path):

    '''
    clears all run directories inside results folder. Use it before pushing to git.
    '''
    

    for name in os.listdir(path):
        full_path = os.path.join(path, name)

        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)




# math utils

def sigmoid(t):
    '''
    returns logistic output
    '''
    return 1 / (1 + np.exp(-t))


def linear_index(df, coeffs, features):
    '''
    returns the dot product of features and coefficients
    '''
    intercept = coeffs[0]
    betas = coeffs[1:]
    return intercept + df[features].dot(betas)


# etc

def get_columns(policies):
    policy_cols = [f"D_{p['name']}" for p in policies]
    outcome_cols = [f"Y_{p['name']}" for p in policies]

    return [policy_cols, outcome_cols]


def save_df(df: pd.DataFrame, name: str, run_dir: str):
    df_dir = Path(run_dir) / "data"
    df_dir.mkdir(parents=True, exist_ok=True)

    file_path = df_dir / f"{name}.csv"
    df.to_csv(file_path, index=False)

    print(f"Saved DataFrame → {file_path}")

    return file_path

from pathlib import Path
import matplotlib.pyplot as plt

def save_plot(fig, name: str, run_dir: str):
    """
    Save a matplotlib Figure object to run_dir/figures as a JPG.
    """

    fig_dir = Path(run_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    file_path = fig_dir / f"{name}.jpg"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot → {file_path}")
    return file_path

