# ===============================
# VISUALIZATION
# ===============================
import os
import statistics

from config import PLOTS_DIR


def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)


def plot_correlation_analysis(correlation_results, output_file=None):
    if output_file is None:
        output_file = os.path.join(PLOTS_DIR, "serial_correlation.png")
    """Plot autocorrelation analysis results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping correlation plot")
        return

    create_plots_directory()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Autocorrelation by lag for different intervals
    ax1 = axes[0]
    intervals = sorted(correlation_results.keys())
    lags = list(range(1, 6))

    for interval in intervals:
        corrs = correlation_results[interval]["correlations"]
        corr_values = [corrs.get(lag, 0) for lag in lags]
        ax1.plot(
            lags,
            corr_values,
            "o-",
            label=f"Interval={interval}",
            linewidth=2,
            markersize=8,
        )

    ax1.axhline(y=0.1, color="r", linestyle="--", label="Threshold (0.1)")
    ax1.axhline(y=-0.1, color="r", linestyle="--")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax1.set_xlabel("Lag", fontsize=12)
    ax1.set_ylabel("Autocorrelation", fontsize=12)
    ax1.set_title("Autocorrelation by Lag and Sample Interval", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.5, 1.0)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average queue length by interval
    ax2 = axes[1]
    avg_queues = [correlation_results[i]["avg_queue"] for i in intervals]
    ax2.bar(range(len(intervals)), avg_queues, color="steelblue", edgecolor="navy")
    ax2.set_xticks(range(len(intervals)))
    ax2.set_xticklabels([str(i) for i in intervals])
    ax2.set_xlabel("Sample Interval", fontsize=12)
    ax2.set_ylabel("Average Queue Length", fontsize=12)
    ax2.set_title("Average Queue Length by Sample Interval", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


def plot_main_effects(experiment_results, output_file=None):
    if output_file is None:
        output_file = os.path.join(PLOTS_DIR, "main_effects.png")
    """
    Plot main effects for each factor in the 2^3 factorial design.

    Shows how the average queue length changes when each factor is varied,
    while averaging over the levels of the other factors.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping main effects plot")
        return

    create_plots_directory()

    # Collect data
    all_data = []
    for config_name, results in experiment_results.items():
        for r in results:
            all_data.append(r)

    # Factor definitions
    factor_info = {
        "A": {
            "title": "Arrival Rate",
            "low_label": "exp(25)",
            "high_label": "exp(22.5)",
        },
        "B": {
            "title": "Preparation Units",
            "low_label": "4 units",
            "high_label": "5 units",
        },
        "C": {
            "title": "Recovery Units",
            "low_label": "4 units",
            "high_label": "5 units",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    factors = ["A", "B", "C"]

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        info = factor_info[factor]

        # Split by factor level
        low_values = [r["avg_queue_on_arrival"] for r in all_data if r[factor] == -1]
        high_values = [r["avg_queue_on_arrival"] for r in all_data if r[factor] == 1]

        # Calculate means
        low_mean = statistics.mean(low_values)
        high_mean = statistics.mean(high_values)

        # Box plots
        positions = [1, 2]
        bp = ax.boxplot(
            [low_values, high_values],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(
                marker="o", markerfacecolor="gray", markersize=4, alpha=0.5
            ),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1),
        )

        # Color boxes
        for patch in bp["boxes"]:
            patch.set_facecolor("lightgray")
            patch.set_edgecolor("black")
            patch.set_linewidth(1)

        # Line connecting means
        ax.plot(
            positions,
            [low_mean, high_mean],
            "ko-",
            linewidth=2,
            markersize=8,
            markerfacecolor="black",
            zorder=5,
        )

        # Axes labels
        ax.set_xticks(positions)
        ax.set_xticklabels([info["low_label"], info["high_label"]], fontsize=11)
        ax.set_ylabel("Average queue length at arrival", fontsize=11)
        ax.set_title(info["title"], fontsize=12, fontweight="bold")

        # Grid
        ax.grid(True, alpha=0.3, axis="y", linestyle="-")
        ax.set_axisbelow(True)

        # Y-axis starts at 0
        ax.set_ylim(bottom=0)

        # Set x-axis limits for better spacing
        ax.set_xlim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


def plot_interaction_effects(experiment_results, output_file=None):
    if output_file is None:
        output_file = os.path.join(PLOTS_DIR, "interactions.png")
    """
    Plot two-way interaction effects for the 2^3 factorial design.

    Shows whether the effect of one factor depends on the level of another factor.
    Each point represents the mean queue length averaged over all replications
    and all levels of the third factor not involved in the interaction.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping interaction plot")
        return

    create_plots_directory()

    # Collect data
    all_data = []
    for config_name, results in experiment_results.items():
        for r in results:
            all_data.append(r)

    # Interaction definitions
    interactions = [
        {
            "panel": "A x B Interaction",
            "f1": "A",
            "f2": "B",
            "f1_labels": ["exp(25)", "exp(22.5)"],
            "f2_labels": ["4 prep units", "5 prep units"],
            "xlabel": "Arrival rate",
        },
        {
            "panel": "A x C Interaction",
            "f1": "A",
            "f2": "C",
            "f1_labels": ["exp(25)", "exp(22.5)"],
            "f2_labels": ["4 recovery units", "5 recovery units"],
            "xlabel": "Arrival rate",
        },
        {
            "panel": "B x C Interaction",
            "f1": "B",
            "f2": "C",
            "f1_labels": ["4 units", "5 units"],
            "f2_labels": ["4 recovery units", "5 recovery units"],
            "xlabel": "Preparation units",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for idx, inter in enumerate(interactions):
        ax = axes[idx]
        f1 = inter["f1"]
        f2 = inter["f2"]

        # Calculate means for each combination (averaged over third factor)
        means = {}
        for level1 in [-1, 1]:
            for level2 in [-1, 1]:
                values = [
                    r["avg_queue_on_arrival"]
                    for r in all_data
                    if r[f1] == level1 and r[f2] == level2
                ]
                means[(level1, level2)] = statistics.mean(values) if values else 0

        # X positions
        x = [1, 2]

        # Line for f2 = -1 (low level of second factor)
        y_f2_low = [means[(-1, -1)], means[(1, -1)]]
        # Line for f2 = +1 (high level of second factor)
        y_f2_high = [means[(-1, 1)], means[(1, 1)]]

        # Plot lines
        ax.plot(
            x,
            y_f2_low,
            "o-",
            color="black",
            linewidth=2,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            label=inter["f2_labels"][0],
        )
        ax.plot(
            x,
            y_f2_high,
            "s--",
            color="black",
            linewidth=2,
            markersize=8,
            markerfacecolor="black",
            label=inter["f2_labels"][1],
        )

        # Axes
        ax.set_xticks(x)
        ax.set_xticklabels(inter["f1_labels"], fontsize=11)
        ax.set_xlabel(inter["xlabel"], fontsize=11)
        ax.set_ylabel("Average queue length at arrival", fontsize=11)
        ax.set_title(inter["panel"], fontsize=12, fontweight="bold")

        # Legend
        ax.legend(loc="best", fontsize=9)

        # Grid
        ax.grid(True, alpha=0.3, axis="y", linestyle="-")
        ax.set_axisbelow(True)

        # Y-axis starts at 0
        ax.set_ylim(bottom=0)

        # X-axis limits
        ax.set_xlim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


def inverse_normal_cdf(p):
    """Approximate inverse normal CDF (quantile function)."""
    import math

    if p <= 0:
        return -5
    if p >= 1:
        return 5

    # Rational approximation
    a = [
        0,
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        0,
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / (
            (((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6])
            * q
            / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]
        ) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)


def plot_regression_diagnostics(model, output_file=None):
    if output_file is None:
        output_file = os.path.join(PLOTS_DIR, "regression_diagnostics.png")
    """Plot regression diagnostics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping diagnostics plot")
        return

    create_plots_directory()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    residuals = model["residuals"]
    fitted = model["fitted"]
    observed = model["observed"]

    # Plot 1: Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(fitted, residuals, alpha=0.5, edgecolors="navy", facecolors="steelblue")
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_xlabel("Fitted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs Fitted Values", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-Q Plot (Normal probability plot)
    ax2 = axes[0, 1]
    sorted_residuals = sorted(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = [inverse_normal_cdf((i + 0.5) / n) for i in range(n)]
    ax2.scatter(
        theoretical_quantiles,
        sorted_residuals,
        alpha=0.5,
        edgecolors="navy",
        facecolors="steelblue",
    )

    # Add reference line
    min_q = min(theoretical_quantiles)
    max_q = max(theoretical_quantiles)
    ax2.plot(
        [min_q, max_q],
        [min_q * statistics.stdev(residuals), max_q * statistics.stdev(residuals)],
        "r--",
    )
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.set_title("Normal Q-Q Plot", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Histogram of Residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, edgecolor="navy", facecolor="steelblue", alpha=0.7)
    ax3.set_xlabel("Residuals", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Histogram of Residuals", fontsize=12)
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Observed vs Fitted
    ax4 = axes[1, 1]
    ax4.scatter(fitted, observed, alpha=0.5, edgecolors="navy", facecolors="steelblue")
    min_val = min(min(fitted), min(observed))
    max_val = max(max(fitted), max(observed))
    ax4.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
    ax4.set_xlabel("Fitted Values", fontsize=11)
    ax4.set_ylabel("Observed Values", fontsize=11)
    ax4.set_title("Observed vs Fitted Values", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"Regression Diagnostics (R^2 = {model['R_squared']:.4f})", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
