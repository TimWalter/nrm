import numpy as np

from scipy.stats import bootstrap


def ci_95(data):
    if len(data) < 2:
        return (-1, -1)
    res = bootstrap((np.array(data),), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    return res.confidence_interval.low.astype(np.float64), res.confidence_interval.high.astype(np.float64)


def bootstrap_mean_ci(trajectories, n_bootstraps=1000, ci=95):
    """
    Calculates the mean and confidence interval for a set of trajectories.

    Args:
        trajectories (np.ndarray): A 2D numpy array where each row is a trajectory.
                                   Shape: (n_trajectories, n_timepoints).
        n_bootstraps (int): The number of bootstrap samples to generate.
        ci (int): The desired confidence interval in percent.

    Returns:
        tuple: A tuple containing:
            - mean_trajectory (np.ndarray): The mean trajectory.
            - ci_lower (np.ndarray): The lower bound of the confidence interval.
            - ci_upper (np.ndarray): The upper bound of the confidence interval.
    """
    n_trajectories, n_timepoints = trajectories.shape
    bootstrap_means = np.zeros((n_bootstraps, n_timepoints))

    for i in range(n_bootstraps):
        indices = np.random.choice(n_trajectories, size=n_trajectories, replace=True)
        bootstrap_sample = trajectories[indices, :]
        bootstrap_means[i, :] = np.mean(bootstrap_sample, axis=0)

    mean_trajectory = np.mean(trajectories, axis=0)

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower, ci_upper = np.percentile(
        bootstrap_means, [lower_percentile, upper_percentile], axis=0
    )

    return mean_trajectory, ci_lower, ci_upper
