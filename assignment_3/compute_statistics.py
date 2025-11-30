import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind

# ============================================================
# LOAD SIMULATION RESULTS
# The CSV file contains 20 replications for each (Scenario, Config).
# Columns: Scenario, Config, Rep, AvgPrepQueue, IdlePrep, Blocking, TheatreUtil, RecoveryFull
# Scenarios: "Original" (baseline) and "Twisted" (priority-based)
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

# Compute paired CI for differences X - Y
def paired_ci(x, y):
    d = x.values - y.values    # paired differences
    mean = d.mean()
    sd = d.std(ddof=1)
    tval = t.ppf(0.975, len(d)-1)
    ci_low = mean - tval * sd / np.sqrt(len(d))
    ci_high = mean + tval * sd / np.sqrt(len(d))
    # Check if CI excludes zero (significant)
    significant = (ci_low > 0) or (ci_high < 0)
    return mean, sd, ci_low, ci_high, significant

# Metrics we compute CIs for
metrics = ["AvgPrepQueue", "IdlePrep", "Blocking", "TheatreUtil", "RecoveryFull"]

# Store results by (scenario, configuration)
results = {}

# ============================================================
# PART 1: Compute point estimates & 95% CI for each (Scenario, Config)
# This answers:
# "Compute the point estimates and 95% confidence intervals. 
#  Do any metrics differ significantly when treated as independent samples?"
# ============================================================
print("=" * 70)
print("PART 1: Point Estimates and 95% Confidence Intervals")
print("=" * 70)

for scenario in df["Scenario"].unique():
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*70}")
    
    for cfg in df["Config"].unique():
        results[(scenario, cfg)] = {}
        sub = df[(df["Scenario"] == scenario) & (df["Config"] == cfg)]
        
        print(f"\n--- Configuration {cfg} ({scenario}) ---")
        for metric in metrics:
            mean, sd, ci_low, ci_high = get_stats(sub[metric])
            results[(scenario, cfg)][metric] = {
                "mean": mean,
                "sd": sd,
                "CI_low": ci_low,
                "CI_high": ci_high
            }
            print(f"{metric}:")
            print(f"  Mean     = {mean:.6f}")
            print(f"  SD       = {sd:.6f}")
            print(f"  95% CI   = ({ci_low:.6f}, {ci_high:.6f})")


# ============================================================
# PART 2: Independent comparison (two-sample t-tests)
# This compares the configurations as *independent* experiments
# within each scenario.
# Answering:
# "Do any observed values differ significantly between 3p5r and 4p5r 
#  or between 3p5r and 3p4r?"
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Independent Sample Comparisons (Two-Sample t-tests)")
print("=" * 70)

comparisons = [
    ("3p5r", "4p5r"),
    ("3p5r", "3p4r")
]

for scenario in df["Scenario"].unique():
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*70}")
    
    # Filter data for this scenario
    df_scenario = df[df["Scenario"] == scenario]
    configs = {
        "3p4r": df_scenario[df_scenario["Config"] == "3p4r"],
        "3p5r": df_scenario[df_scenario["Config"] == "3p5r"],
        "4p5r": df_scenario[df_scenario["Config"] == "4p5r"]
    }
    
    for (cfg1, cfg2) in comparisons:
        print(f"\n--- {cfg1} vs {cfg2} ({scenario}) ---")
        
        for m in metrics:
            x = configs[cfg1][m]
            y = configs[cfg2][m]
            
            # Welch's t-test (unequal variances)
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
print("\n" + "=" * 70)
print("PART 3: Paired Comparisons Between Configurations (CRN)")
print("=" * 70)

for scenario in df["Scenario"].unique():
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*70}")
    
    df_scenario = df[df["Scenario"] == scenario]
    
    # Sort so that replication numbers align
    df_3p5r = df_scenario[df_scenario["Config"] == "3p5r"].sort_values("Rep")
    df_4p5r = df_scenario[df_scenario["Config"] == "4p5r"].sort_values("Rep")
    df_3p4r = df_scenario[df_scenario["Config"] == "3p4r"].sort_values("Rep")
    
    # Paired comparison: 3p5r vs 4p5r
    print(f"\n--- Paired: 3p5r - 4p5r ({scenario}) ---")
    for m in metrics:
        mean, sd, lo, hi, sig = paired_ci(df_3p5r[m], df_4p5r[m])
        sig_str = "SIGNIFICANT" if sig else "not significant"
        print(f"{m}: mean diff={mean:+.6f}, 95% CI=({lo:.6f}, {hi:.6f}) [{sig_str}]")
    
    # Paired comparison: 3p5r vs 3p4r
    print(f"\n--- Paired: 3p5r - 3p4r ({scenario}) ---")
    for m in metrics:
        mean, sd, lo, hi, sig = paired_ci(df_3p5r[m], df_3p4r[m])
        sig_str = "SIGNIFICANT" if sig else "not significant"
        print(f"{m}: mean diff={mean:+.6f}, 95% CI=({lo:.6f}, {hi:.6f}) [{sig_str}]")


# ============================================================
# PART 4: Original vs Twisted Comparison (Personal Twist Analysis)
# This compares the baseline simulation with the priority-based
# twisted version using paired samples (same seeds).
#
# Key question: Does the priority system lead to significantly
# different performance compared to the original FIFO system?
# ============================================================
print("\n" + "=" * 70)
print("PART 4: Original vs Twisted Comparison (Personal Twist)")
print("=" * 70)
print("\nComparing Original (FIFO) vs Twisted (Priority-based) scenarios")
print("Using paired samples with Common Random Numbers (same seeds)")

config_list = ["3p4r", "3p5r", "4p5r"]

for cfg in config_list:
    print(f"\n{'='*70}")
    print(f"Configuration: {cfg}")
    print(f"{'='*70}")
    
    # Get Original and Twisted data for this config, sorted by Rep
    df_original = df[(df["Scenario"] == "Original") & (df["Config"] == cfg)].sort_values("Rep")
    df_twisted = df[(df["Scenario"] == "Twisted") & (df["Config"] == cfg)].sort_values("Rep")
    
    print("\n--- Paired Comparison: Original - Twisted ---")
    print("(Positive diff means Original > Twisted, Negative means Twisted > Original)")
    
    for m in metrics:
        mean, sd, lo, hi, sig = paired_ci(df_original[m], df_twisted[m])
        sig_str = "SIGNIFICANT" if sig else "not significant"
        
        # Interpretation
        if sig:
            if mean > 0:
                interp = "Original HIGHER"
            else:
                interp = "Twisted HIGHER"
        else:
            interp = "No significant difference"
        
        print(f"{m}:")
        print(f"  Original mean: {df_original[m].mean():.6f}")
        print(f"  Twisted mean:  {df_twisted[m].mean():.6f}")
        print(f"  Difference:    {mean:+.6f}")
        print(f"  95% CI:        ({lo:.6f}, {hi:.6f})")
        print(f"  Result:        {sig_str} - {interp}")


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Original vs Twisted Key Metrics")
print("=" * 70)
print(f"\n{'Config':<10} {'Metric':<16} {'Original':>12} {'Twisted':>12} {'Diff':>12} {'Significant?':<15}")
print("-" * 75)

for cfg in config_list:
    df_orig = df[(df["Scenario"] == "Original") & (df["Config"] == cfg)]
    df_twist = df[(df["Scenario"] == "Twisted") & (df["Config"] == cfg)]
    
    for m in ["AvgPrepQueue", "Blocking", "TheatreUtil", "RecoveryFull"]:
        orig_mean = df_orig[m].mean()
        twist_mean = df_twist[m].mean()
        
        # Get paired CI for significance
        df_o = df_orig.sort_values("Rep")
        df_t = df_twist.sort_values("Rep")
        _, _, lo, hi, sig = paired_ci(df_o[m], df_t[m])
        
        sig_str = "YES" if sig else "no"
        print(f"{cfg:<10} {m:<16} {orig_mean:>12.4f} {twist_mean:>12.4f} {twist_mean - orig_mean:>+12.4f} {sig_str:<15}")
    print()
