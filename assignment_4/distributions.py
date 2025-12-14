# ===============================
# DISTRIBUTION SAMPLERS
# ===============================
import random
from config import MEAN_PREP, MEAN_RECOVERY, MEAN_SURGERY


def sample_exponential(mean):
    """Sample from exponential distribution with given mean."""
    return random.expovariate(1.0 / mean)


def sample_uniform(low, high):
    """Sample from uniform distribution U(low, high)."""
    return random.uniform(low, high)


def sample_interarrival(dist_type="exp", rate_level="low"):
    """
    Sample interarrival time based on distribution type and rate level.

    Parameters:
    - dist_type: 'exp' for exponential, 'unif' for uniform
    - rate_level: 'low' (mean=25) or 'high' (mean=22.5)

    Returns: sampled interarrival time
    """
    if rate_level == "low":
        if dist_type == "exp":
            return sample_exponential(25)  # exp(25)
        else:
            return sample_uniform(20, 30)  # Unif(20,30), mean=25
    else:  # high rate
        if dist_type == "exp":
            return sample_exponential(22.5)  # exp(22.5)
        else:
            return sample_uniform(20, 25)  # Unif(20,25), mean=22.5


def sample_prep_time(dist_type="exp"):
    """
    Sample preparation time based on distribution type.

    Parameters:
    - dist_type: 'exp' for exponential, 'unif' for uniform

    Returns: sampled preparation time
    """
    if dist_type == "exp":
        return sample_exponential(MEAN_PREP)  # exp(40)
    else:
        return sample_uniform(30, 50)  # Unif(30,50), mean=40


def sample_recovery_time(dist_type="exp"):
    """
    Sample recovery time based on distribution type.

    Parameters:
    - dist_type: 'exp' for exponential, 'unif' for uniform

    Returns: sampled recovery time
    """
    if dist_type == "exp":
        return sample_exponential(MEAN_RECOVERY)  # exp(40)
    else:
        return sample_uniform(30, 50)  # Unif(30,50), mean=40


def sample_surgery_time():
    """Sample surgery time - always exp(20) as per task requirements."""
    return sample_exponential(MEAN_SURGERY)
