import seaborn as sns
import matplotlib.pyplot as plt

def plot_policy_distribution(df_long, metric='outcome', order=None):
    '''
    Boxplot of policy performance with mean markers.
    Returns: matplotlib Figure
    '''

    fig, ax = plt.subplots(figsize=(8, 5))

    if order is None:
        order = (
            df_long.groupby('policy')[metric]
            .mean()
            .sort_values(ascending=False)
            .index
        )

    sns.boxplot(
        data=df_long,
        x='policy',
        y=metric,
        order=order,
        ax=ax
    )

    sns.pointplot(
        data=df_long,
        x='policy',
        y=metric,
        estimator='mean',
        color='red',
        markers='D',
        linestyles='',
        order=order,
        ax=ax
    )

    ax.set_title(f'Policy {metric.capitalize()} Distribution')
    ax.set_xlabel('Policy')
    ax.set_ylabel(metric.capitalize())
    ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()

    return fig

def plot_policy_ecdf(df_long, metric='outcome', order=None):
    '''
    ECDF plot to compare distribution dominance across policies.
    Returns: matplotlib Figure
    '''

    fig, ax = plt.subplots(figsize=(8, 5))

    if order is None:
        order = (
            df_long.groupby('policy')[metric]
            .mean()
            .sort_values(ascending=False)
            .index
        )

    for policy in order:
        subset = df_long[df_long['policy'] == policy]
        sns.ecdfplot(subset[metric], label=policy, ax=ax)

    ax.set_title(f'CDF of Policy {metric.capitalize()}')
    ax.set_xlabel(metric.capitalize())
    ax.set_ylabel('Probability')

    ax.legend()
    fig.tight_layout()

    return fig


'''to do: currently retunring duplicate figures, fix later '''