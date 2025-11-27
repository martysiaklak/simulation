import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind

# ============================================================
# LOAD SIMULATION RESULTS
# The CSV file contains 20 replications for each configuration.
# Columns: Config, Rep, AvgPrepQueue, IdlePrep, Blocking, TheatreUtil, RecoveryFull
# ============================================================
df = pd.read_csv("experiment_results.csv")

# ============================================================
# FUNCTION: Compute point estimate and 95% confidence interval
# Uses sample mean, sample SD, and the 97.5% t-quantile.
# This is needed for PART 1 of the assignment.
# ============================================================
def get_stats(series):
    n = len(series)
    mean = series.mean()
    sd = series.std(ddof=1)         # sample standard deviation
    tval = t.ppf(0.975, n - 1)      # critical value for 95% CI
    se = sd / np.sqrt(n)            # standard error
    ci_low = mean - tval * se
    ci_high = mean + tval * se
    return mean, sd, ci_low, ci_high

# Metrics we compute CIs for
metrics = ["AvgPrepQueue", "IdlePrep", "Blocking", "TheatreUtil", "RecoveryFull"]

# Store results by configuration
results = {}

# ============================================================
# PART 1: Compute point estimates & 95% CI for each configuration
# This answers:
# "Compute the point estimates and 95% confidence intervals. 
#  Do any metrics differ significantly when treated as independent samples?"
# ============================================================
for cfg in df["Config"].unique():
    results[cfg] = {}
    sub = df[df["Config"] == cfg]  # 20 replications for this config
    
    for metric in metrics:
        mean, sd, ci_low, ci_high = get_stats(sub[metric])
        results[cfg][metric] = {
            "mean": mean,
            "sd": sd,
            "CI_low": ci_low,
            "CI_high": ci_high
        }

# Print results for each configuration
for cfg, stats in results.items():
    print(f"\n=== Results for Configuration {cfg} ===")
    for metric, vals in stats.items():
        print(f"{metric}:")
        print(f"  Mean     = {vals['mean']:.6f}")
        print(f"  SD       = {vals['sd']:.6f}")
        print(f"  95% CI   = ({vals['CI_low']:.6f}, {vals['CI_high']:.6f})")


# ============================================================
# PART 2: Independent comparison (two-sample t-tests)
# This compares the configurations as *independent* experiments.
# Answering:
# "Do any observed values differ significantly between 3p5r and 4p5r 
#  or between 3p5r and 3p4r?"
# ============================================================
configs = {
    "3p4r": df[df["Config"] == "3p4r"],
    "3p5r": df[df["Config"] == "3p5r"],
    "4p5r": df[df["Config"] == "4p5r"]
}

comparisons = [
    ("3p5r", "4p5r"),
    ("3p5r", "3p4r")
]

for (cfg1, cfg2) in comparisons:
    print(f"\n=== Independent Sample Comparison: {cfg1} vs {cfg2} ===")

    for m in metrics:
        x = configs[cfg1][m]
        y = configs[cfg2][m]

        # Welch’s t-test (unequal variances)
        tstat, pval = ttest_ind(x, y, equal_var=False)

        print(f"{m}:")
        print(f"  Mean {cfg1}: {x.mean():.6f}")
        print(f"  Mean {cfg2}: {y.mean():.6f}")
        print(f"  p-value: {pval:.6f}")

        if pval < 0.05:
            print("  → SIGNIFICANT difference")
        else:
            print("  → NOT significant")


# ============================================================
# PART 3: Paired comparison using Common Random Numbers (CRN)
# This answers:
# "Arrange the simulation so that observations are pairwise using 
#  the same seeds. Compute interval estimates for the differences."
#
# Each replication index corresponds to the same seed across configs.
# ============================================================

# Sort so that replication numbers align
df_3p5r = df[df["Config"] == "3p5r"].sort_values("Rep")
df_4p5r = df[df["Config"] == "4p5r"].sort_values("Rep")
df_3p4r = df[df["Config"] == "3p4r"].sort_values("Rep")

# Compute paired CI for differences X - Y
def paired_ci(x, y):
    d = x.values - y.values    # paired differences
    mean = d.mean()
    sd = d.std(ddof=1)
    tval = t.ppf(0.975, len(d)-1)
    ci_low = mean - tval * sd / np.sqrt(len(d))
    ci_high = mean + tval * sd / np.sqrt(len(d))
    return mean, sd, ci_low, ci_high

# ---------- Paired comparison: 3p5r vs 4p5r ----------
print("\n=== Paired comparison: 3p5r - 4p5r ===")
for m in metrics:
    mean, sd, lo, hi = paired_ci(df_3p5r[m], df_4p5r[m])
    print(f"{m}: mean diff={mean:.6f}, 95% CI=({lo:.6f}, {hi:.6f})")

# ---------- Paired comparison: 3p5r vs 3p4r ----------
print("\n=== Paired comparison: 3p5r - 3p4r ===")
for m in metrics:
    mean, sd, lo, hi = paired_ci(df_3p5r[m], df_3p4r[m])
    print(f"{m}: mean diff={mean:.6f}, 95% CI=({lo:.6f}, {hi:.6f})")
