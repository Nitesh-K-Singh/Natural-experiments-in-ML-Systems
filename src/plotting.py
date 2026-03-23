import seaborn as sns
import matplotlib.pyplot as plt




def plot_policy_distribution(df_long, metric='outcome', order = None):
    '''
    Boxplot of policy performance with mean markers.
    '''

    plt.figure(figsize=(8, 5))

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
        order = order
    )

    sns.pointplot(
        data=df_long,
        x='policy',
        y=metric,
        estimator='mean',
        color='red',
        markers='D',
        linestyles='',
        order = order
    )

    plt.title(f"Policy {metric.capitalize()} Distribution")
    plt.xlabel("Policy")
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()

def plot_policy_ecdf(df_long, metric='outcome', order = None):
    '''
    ECDF plot to compare distribution dominance across policies.
    '''

    plt.figure(figsize=(8, 5))

    if order is None:
        order = (
            df_long.groupby('policy')[metric]
            .mean()
            .sort_values(ascending=False)
            .index
        )

    for policy in order:
        subset = df_long[df_long['policy'] == policy]
        sns.ecdfplot(subset[metric], label=policy)

    plt.title(f"CDF of Policy {metric.capitalize()}")
    plt.xlabel(metric.capitalize())
    plt.ylabel("Probability")

    plt.legend()
    plt.tight_layout()
    plt.show()


