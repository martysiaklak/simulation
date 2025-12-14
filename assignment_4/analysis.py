# ===============================
# STATISTICAL ANALYSIS
# ===============================
import math
import statistics

from config import CORRELATION_MAX_LAG, CORRELATION_SAMPLE_INTERVALS
from simulation import run_correlation_replication


# ===============================
# SERIAL CORRELATION ANALYSIS
# ===============================
def compute_autocorrelation(series_list, max_lag=CORRELATION_MAX_LAG):
    """
    Compute autocorrelation from multiple independent time series.

    Parameters:
    - series_list: List of time series (each from independent replication)
    - max_lag: Maximum lag to compute

    Returns: Dictionary with autocorrelation coefficients for each lag
    """
    series_length = len(series_list[0]) if series_list else 0

    if series_length < max_lag + 1:
        max_lag = series_length - 1

    correlations = {}

    for lag in range(1, max_lag + 1):
        # Collect pairs (X_t, X_{t+lag}) from all series
        pairs_x = []
        pairs_y = []

        for series in series_list:
            for t in range(len(series) - lag):
                pairs_x.append(series[t])
                pairs_y.append(series[t + lag])

        if len(pairs_x) < 2:
            correlations[lag] = 0
            continue

        # Compute Pearson correlation
        mean_x = statistics.mean(pairs_x)
        mean_y = statistics.mean(pairs_y)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(pairs_x, pairs_y))

        var_x = sum((x - mean_x) ** 2 for x in pairs_x)
        var_y = sum((y - mean_y) ** 2 for y in pairs_y)

        denominator = math.sqrt(var_x * var_y)

        if denominator > 0:
            correlations[lag] = numerator / denominator
        else:
            correlations[lag] = 0

    return correlations


def analyze_serial_correlation(
    config, num_runs=10, num_samples=10, sample_intervals=CORRELATION_SAMPLE_INTERVALS
):
    """
    Analyze serial correlation for a given configuration.

    Tests different sampling intervals to find one that reduces correlation.

    Returns: Dictionary with correlation results for each interval
    """
    results = {}

    for interval in sample_intervals:
        print(f"  Testing sample interval = {interval}...")

        # Collect time series from multiple independent runs
        all_series = []
        for run in range(num_runs):
            series = run_correlation_replication(
                config,
                seed=1000 + run,
                num_samples=num_samples,
                sample_interval=interval,
            )
            all_series.append(series)

        # Compute autocorrelations
        correlations = compute_autocorrelation(all_series, max_lag=CORRELATION_MAX_LAG)

        # Compute average queue length across all samples
        all_samples = [s for series in all_series for s in series]
        avg_queue = statistics.mean(all_samples) if all_samples else 0

        results[interval] = {
            "correlations": correlations,
            "avg_queue": avg_queue,
            "series": all_series,
        }

    return results


# ===============================
# REGRESSION ANALYSIS
# ===============================
def matrix_inverse(matrix):
    """Compute inverse of a square matrix using Gaussian elimination."""
    n = len(matrix)

    # Create augmented matrix [A | I]
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = matrix[i][j]
        aug[i][n + i] = 1.0

    # Forward elimination
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row

        # Swap rows
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Check for singular matrix
        if abs(aug[col][col]) < 1e-10:
            continue

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # Extract inverse from augmented matrix
    inverse = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    return inverse


def t_distribution_cdf(t, df):
    """
    Approximate CDF of t-distribution.
    Uses approximation for degrees of freedom > 30.
    """
    if df > 30:
        # Use normal approximation
        return normal_cdf(t)
    else:
        # Use a simple approximation based on beta function relationship
        x = df / (df + t**2)
        # Regularized incomplete beta function approximation
        return 0.5 + 0.5 * math.copysign(1, t) * (1 - incomplete_beta(df / 2, 0.5, x))


def normal_cdf(x):
    """Standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def incomplete_beta(a, b, x):
    """Regularized incomplete beta function approximation."""
    # Simple numerical approximation
    if x <= 0:
        return 0
    if x >= 1:
        return 1

    # Use continued fraction approximation
    # This is a simplified version
    result = 0
    n_terms = 100
    dx = x / n_terms

    for i in range(n_terms):
        t = (i + 0.5) * dx
        result += (t ** (a - 1)) * ((1 - t) ** (b - 1)) * dx

    # Normalize by beta function
    beta_val = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return result / beta_val


def build_regression_model(experiment_results):
    """
    Build a regression model for average queue length.

    Model: Y = b0 + b1*A + b2*B + b3*C + b12*AB + b13*AC + b23*BC + b123*ABC + e

    Uses ordinary least squares with matrix operations.

    Returns: Dictionary with model coefficients and statistics
    """
    # Collect data from all experiments
    Y = []  # Response: average queue on arrival
    X = []  # Design matrix

    for config_name, results in experiment_results.items():
        for r in results:
            y = r["avg_queue_on_arrival"]
            A, B, C = r["A"], r["B"], r["C"]

            # Build design matrix row: [1, A, B, C, AB, AC, BC, ABC]
            row = [1, A, B, C, A * B, A * C, B * C, A * B * C]

            Y.append(y)
            X.append(row)

    n = len(Y)
    p = len(X[0])  # Number of parameters

    # Matrix calculations for OLS: beta = (X'X)^(-1) X'Y

    # Compute X'X
    XtX = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            for k in range(n):
                XtX[i][j] += X[k][i] * X[k][j]

    # Compute X'Y
    XtY = [0.0] * p
    for i in range(p):
        for k in range(n):
            XtY[i] += X[k][i] * Y[k]

    # Invert X'X using Gaussian elimination
    XtX_inv = matrix_inverse(XtX)

    # Compute beta = (X'X)^(-1) X'Y
    beta = [0.0] * p
    for i in range(p):
        for j in range(p):
            beta[i] += XtX_inv[i][j] * XtY[j]

    # Compute fitted values and residuals
    Y_hat = []
    residuals = []
    for i in range(n):
        y_hat = sum(beta[j] * X[i][j] for j in range(p))
        Y_hat.append(y_hat)
        residuals.append(Y[i] - y_hat)

    # Compute R-squared
    Y_mean = statistics.mean(Y)
    SS_tot = sum((y - Y_mean) ** 2 for y in Y)
    SS_res = sum(r**2 for r in residuals)
    R_squared = 1 - (SS_res / SS_tot) if SS_tot > 0 else 0

    # Adjusted R-squared
    R_squared_adj = 1 - ((1 - R_squared) * (n - 1) / (n - p)) if n > p else R_squared

    # Estimate variance of residuals
    MSE = SS_res / (n - p) if n > p else 0

    # Standard errors of coefficients
    se_beta = []
    for i in range(p):
        se = math.sqrt(MSE * XtX_inv[i][i]) if XtX_inv[i][i] > 0 else 0
        se_beta.append(se)

    # T-statistics
    t_stats = []
    for i in range(p):
        t = beta[i] / se_beta[i] if se_beta[i] > 0 else 0
        t_stats.append(t)

    # P-values (using approximation for large samples)
    p_values = []
    df = n - p
    for t in t_stats:
        # Two-tailed p-value approximation
        p_val = 2 * (1 - t_distribution_cdf(abs(t), df))
        p_values.append(p_val)

    # Factor names
    factor_names = [
        "Intercept",
        "A (Rate)",
        "B (Prep)",
        "C (Recovery)",
        "AB",
        "AC",
        "BC",
        "ABC",
    ]

    return {
        "coefficients": dict(zip(factor_names, beta)),
        "std_errors": dict(zip(factor_names, se_beta)),
        "t_statistics": dict(zip(factor_names, t_stats)),
        "p_values": dict(zip(factor_names, p_values)),
        "R_squared": R_squared,
        "R_squared_adj": R_squared_adj,
        "MSE": MSE,
        "n_observations": n,
        "residuals": residuals,
        "fitted": Y_hat,
        "observed": Y,
    }


def analyze_factor_significance(model, alpha=0.05):
    """
    Analyze which factors are statistically significant.

    Returns: Dictionary with significance analysis
    """
    significant_factors = []
    insignificant_factors = []

    for factor, p_val in model["p_values"].items():
        if factor == "Intercept":
            continue
        if p_val < alpha:
            significant_factors.append((factor, model["coefficients"][factor], p_val))
        else:
            insignificant_factors.append((factor, model["coefficients"][factor], p_val))

    # Sort by absolute effect size
    significant_factors.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "significant": significant_factors,
        "insignificant": insignificant_factors,
        "alpha": alpha,
    }
