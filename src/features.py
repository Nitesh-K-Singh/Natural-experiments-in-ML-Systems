import numpy as np


def generate_features(config):
    '''
    generates features following specification in config 
    '''

    I = config['I']
    feature_specs = config['features']

    features = {}

    for name, spec in feature_specs.items():

        dist = spec['dist']

        if dist == 'normal':
            features[name] = np.random.normal(spec['mean'], spec['std'], I)

        elif dist == 'uniform':
            features[name] = np.random.uniform(spec['low'], spec['high'], I)

        elif dist == 'bernoulli':
            features[name] = np.random.binomial(1, spec['p'], I)

        elif dist == 'lognormal':
            features[name] = np.random.lognormal(spec['mean'], spec['sigma'], I)

        else:
            raise ValueError(f'Unknown distribution: {dist}')

    return features