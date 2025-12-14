# ===============================
# OUTPUT FUNCTIONS
# ===============================
import csv
import os

from config import OUTPUT_DIR


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_correlation_results(
    correlation_results, filename="serial_correlation_results.csv"
):
    """Save serial correlation results to CSV."""
    ensure_output_dir()
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SampleInterval", "Lag", "Autocorrelation", "AvgQueueLength"])

        for interval, data in sorted(correlation_results.items()):
            avg_queue = data["avg_queue"]
            for lag, corr in sorted(data["correlations"].items()):
                w.writerow([interval, lag, f"{corr:.6f}", f"{avg_queue:.4f}"])

    print(f"  Saved: {filepath}")


def save_factorial_results(
    experiment_results, filename="factorial_experiment_results.csv"
):
    """Save factorial experiment results to CSV."""
    ensure_output_dir()
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Experiment",
                "Replication",
                "A",
                "B",
                "C",
                "AvgQueueOnArrival",
                "AvgPrepQueue",
                "AvgWaitTime",
                "PrepUtil",
                "TheatreUtil",
                "RecoveryUtil",
                "BlockingProb",
                "RecoveryFullProb",
            ]
        )

        for config_name, results in experiment_results.items():
            for r in results:
                w.writerow(
                    [
                        config_name,
                        r["rep"],
                        r["A"],
                        r["B"],
                        r["C"],
                        f"{r['avg_queue_on_arrival']:.4f}",
                        f"{r['avg_prep_queue']:.4f}",
                        f"{r['avg_wait_time']:.4f}",
                        f"{r['avg_prep_util']:.4f}",
                        f"{r['avg_theatre_util']:.4f}",
                        f"{r['avg_recovery_util']:.4f}",
                        f"{r['blocking_prob']:.6f}",
                        f"{r['recovery_full_prob']:.4f}",
                    ]
                )

    print(f"  Saved: {filepath}")


def save_regression_summary(model, significance, filename="regression_summary.txt"):
    """Save regression summary to text file."""
    ensure_output_dir()
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("REGRESSION ANALYSIS SUMMARY\n")
        f.write("Model: AvgQueueOnArrival ~ A + B + C + AB + AC + BC + ABC\n")
        f.write("=" * 70 + "\n\n")

        f.write("FACTOR CODING:\n")
        f.write("-" * 40 + "\n")
        f.write("  A (Arrival Rate): -1 = exp(25), +1 = exp(22.5)\n")
        f.write("  B (Prep Units):   -1 = 4 units, +1 = 5 units\n")
        f.write("  C (Recovery):     -1 = 4 units, +1 = 5 units\n\n")

        f.write("MODEL FIT STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  R-squared:           {model['R_squared']:.6f}\n")
        f.write(f"  Adjusted R-squared:  {model['R_squared_adj']:.6f}\n")
        f.write(f"  Mean Squared Error:  {model['MSE']:.6f}\n")
        f.write(f"  N observations:      {model['n_observations']}\n\n")

        f.write("COEFFICIENT ESTIMATES:\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Factor':<15} {'Estimate':>12} {'Std.Error':>12} {'t-stat':>10} {'p-value':>12}\n"
        )
        f.write("-" * 70 + "\n")

        for factor in model["coefficients"].keys():
            coef = model["coefficients"][factor]
            se = model["std_errors"][factor]
            t = model["t_statistics"][factor]
            p = model["p_values"][factor]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            f.write(
                f"{factor:<15} {coef:>12.6f} {se:>12.6f} {t:>10.3f} {p:>10.6f} {sig}\n"
            )

        f.write("-" * 70 + "\n")
        f.write("Significance codes: *** p<0.001, ** p<0.01, * p<0.05\n\n")

        f.write("SIGNIFICANT FACTORS (alpha = 0.05):\n")
        f.write("-" * 40 + "\n")
        if significance["significant"]:
            for factor, coef, p_val in significance["significant"]:
                f.write(f"  {factor}: coefficient = {coef:.4f}, p = {p_val:.6f}\n")
        else:
            f.write("  None\n")

        f.write("\nNON-SIGNIFICANT FACTORS:\n")
        f.write("-" * 40 + "\n")
        if significance["insignificant"]:
            for factor, coef, p_val in significance["insignificant"]:
                f.write(f"  {factor}: coefficient = {coef:.4f}, p = {p_val:.6f}\n")
        else:
            f.write("  None\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("=" * 70 + "\n")

        # Interpret main effects
        f.write("\nMain Effects:\n")
        for factor in ["A (Rate)", "B (Prep)", "C (Recovery)"]:
            coef = model["coefficients"][factor]
            p = model["p_values"][factor]

            if p < 0.05:
                if factor == "A (Rate)":
                    direction = "increases" if coef > 0 else "decreases"
                    f.write(
                        f"  - Higher arrival rate significantly {direction} queue length (effect: {coef:.4f})\n"
                    )
                elif factor == "B (Prep)":
                    direction = "decreases" if coef < 0 else "increases"
                    f.write(
                        f"  - More prep units significantly {direction} queue length (effect: {coef:.4f})\n"
                    )
                elif factor == "C (Recovery)":
                    direction = "decreases" if coef < 0 else "increases"
                    f.write(
                        f"  - More recovery units {direction} queue length (effect: {coef:.4f})\n"
                    )
            else:
                f.write(f"  - {factor} has no significant effect (p = {p:.4f})\n")

        # Check for important interactions
        f.write("\nInteraction Effects:\n")
        interactions = ["AB", "AC", "BC", "ABC"]
        for inter in interactions:
            coef = model["coefficients"][inter]
            p = model["p_values"][inter]
            if p < 0.05:
                f.write(
                    f"  - {inter} interaction is significant (effect: {coef:.4f}, p = {p:.4f})\n"
                )

        # Model adequacy
        f.write("\nModel Adequacy:\n")
        if model["R_squared"] > 0.8:
            f.write(
                f"  - The model explains {model['R_squared'] * 100:.1f}% of variance (excellent fit)\n"
            )
        elif model["R_squared"] > 0.6:
            f.write(
                f"  - The model explains {model['R_squared'] * 100:.1f}% of variance (good fit)\n"
            )
        else:
            f.write(
                f"  - The model explains {model['R_squared'] * 100:.1f}% of variance (may need more factors)\n"
            )

    print(f"  Saved: {filepath}")
