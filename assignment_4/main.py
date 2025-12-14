#!/usr/bin/env python3
"""
Simulation Experiment Analysis - Task 4

This script performs a systematic analysis of a hospital simulation model
using a 2^3 factorial design with serial correlation analysis.

Modules:
- config.py: Configuration parameters
- distributions.py: Distribution sampling functions
- simulation.py: Core simulation logic
- experiment.py: Experiment configuration and factorial design
- analysis.py: Serial correlation and regression analysis
- visualization.py: Plotting functions
- output.py: CSV and text file output
"""
import statistics

from config import NUM_REPS
from experiment import ExperimentConfig, create_factorial_design, run_factorial_experiments
from analysis import (
    analyze_serial_correlation,
    build_regression_model,
    analyze_factor_significance,
)
from visualization import (
    plot_correlation_analysis,
    plot_main_effects,
    plot_interaction_effects,
    plot_regression_diagnostics,
)
from output import (
    save_correlation_results,
    save_factorial_results,
    save_regression_summary,
)


def main():
    print("=" * 70)
    print("SIMULATION EXPERIMENT ANALYSIS - TASK 4")
    print("=" * 70)

    # -----------------------------------------------
    # PART 1: SERIAL CORRELATION ANALYSIS
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: SERIAL CORRELATION ANALYSIS")
    print("=" * 70)

    # Select high-memory configuration: high arrival rate, few units
    high_memory_config = ExperimentConfig(
        "HighMemory",
        interarrival_rate="high",  # exp(22.5) - more arrivals
        prep_units=4,  # Fewer prep units
        recovery_units=4,  # Fewer recovery units
    )

    print(f"\nConfiguration: {high_memory_config}")
    print("Testing serial correlation with different sample intervals...")

    correlation_results = analyze_serial_correlation(
        high_memory_config,
        num_runs=10,
        num_samples=10,
        sample_intervals=[20, 50, 100, 150, 200],
    )

    # Print correlation summary
    print("\nSerial Correlation Results:")
    print("-" * 60)
    print(
        f"{'Interval':>10} {'Lag-1':>10} {'Lag-2':>10} {'Lag-3':>10} {'Avg Queue':>12}"
    )
    print("-" * 60)

    for interval in sorted(correlation_results.keys()):
        data = correlation_results[interval]
        corrs = data["correlations"]
        print(
            f"{interval:>10} {corrs.get(1, 0):>10.4f} {corrs.get(2, 0):>10.4f} "
            f"{corrs.get(3, 0):>10.4f} {data['avg_queue']:>12.4f}"
        )

    # Find recommended interval
    recommended_interval = None
    for interval in sorted(correlation_results.keys()):
        if abs(correlation_results[interval]["correlations"].get(1, 1)) < 0.15:
            recommended_interval = interval
            break

    if recommended_interval:
        print(
            f"\nRecommended sample interval: {recommended_interval} "
            f"(lag-1 correlation < 0.15)"
        )
    else:
        print("\nNote: All tested intervals show notable serial correlation.")
        print("Consider longer intervals or longer simulation time.")

    # Save correlation results
    print("\nSaving correlation results...")
    save_correlation_results(correlation_results)

    # -----------------------------------------------
    # PART 2: 2^3 FACTORIAL DESIGN EXPERIMENTS
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: 2^3 FACTORIAL DESIGN EXPERIMENTS")
    print("=" * 70)

    design = create_factorial_design()

    print("\nExperimental Design Matrix:")
    print("-" * 60)
    print(f"{'Experiment':<12} {'A (Rate)':<12} {'B (Prep)':<12} {'C (Recovery)':<12}")
    print("-" * 60)
    for config in design:
        A, B, C = config.get_coded_levels()
        rate = "22.5" if config.interarrival_rate == "high" else "25"
        print(
            f"{config.name:<12} {A:>3} ({rate:>4})   {B:>3} ({config.prep_units})      "
            f"{C:>3} ({config.recovery_units})"
        )

    print(f"\nRunning {len(design)} experiments with {NUM_REPS} replications each...")
    experiment_results = run_factorial_experiments(design, num_reps=NUM_REPS)

    # Print summary statistics
    print("\nExperiment Summary (Mean values across replications):")
    print("-" * 80)
    print(
        f"{'Experiment':<12} {'A':>4} {'B':>4} {'C':>4} {'AvgQueue':>10} {'WaitTime':>10} "
        f"{'TheatreUtil':>12}"
    )
    print("-" * 80)

    for config in design:
        results = experiment_results[config.name]
        A, B, C = config.get_coded_levels()
        avg_q = statistics.mean([r["avg_queue_on_arrival"] for r in results])
        avg_wait = statistics.mean([r["avg_wait_time"] for r in results])
        avg_theatre = statistics.mean([r["avg_theatre_util"] for r in results])
        print(
            f"{config.name:<12} {A:>4} {B:>4} {C:>4} {avg_q:>10.4f} {avg_wait:>10.4f} "
            f"{avg_theatre:>12.4f}"
        )

    # Save factorial results
    print("\nSaving factorial experiment results...")
    save_factorial_results(experiment_results)

    # -----------------------------------------------
    # PART 3: REGRESSION ANALYSIS
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: REGRESSION ANALYSIS")
    print("=" * 70)

    print("\nBuilding regression model...")
    model = build_regression_model(experiment_results)
    significance = analyze_factor_significance(model)

    print("\nRegression Model Results:")
    print("-" * 70)
    print(f"R-squared: {model['R_squared']:.6f}")
    print(f"Adjusted R-squared: {model['R_squared_adj']:.6f}")
    print()
    print(f"{'Factor':<15} {'Estimate':>12} {'Std.Error':>12} {'p-value':>12}")
    print("-" * 70)

    for factor in model["coefficients"].keys():
        coef = model["coefficients"][factor]
        se = model["std_errors"][factor]
        p = model["p_values"][factor]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{factor:<15} {coef:>12.6f} {se:>12.6f} {p:>10.6f} {sig}")

    print("\nSignificant Factors (alpha = 0.05):")
    if significance["significant"]:
        for factor, coef, p_val in significance["significant"]:
            print(f"  - {factor}: effect = {coef:.4f}")
    else:
        print("  None")

    # Save regression summary
    print("\nSaving regression summary...")
    save_regression_summary(model, significance)

    # -----------------------------------------------
    # PART 4: VISUALIZATION
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("PART 4: GENERATING PLOTS")
    print("=" * 70)

    print("\nGenerating analysis plots...")
    plot_correlation_analysis(correlation_results)
    plot_main_effects(experiment_results)
    plot_interaction_effects(experiment_results)
    plot_regression_diagnostics(model)

    # -----------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nOutput files generated:")
    print("  - out/serial_correlation_results.csv")
    print("  - out/factorial_experiment_results.csv")
    print("  - out/regression_summary.txt")
    print("  - out/plots/serial_correlation.png")
    print("  - out/plots/main_effects.png")
    print("  - out/plots/interactions.png")
    print("  - out/plots/regression_diagnostics.png")

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    # Summarize findings
    print("\n1. Serial Correlation:")
    print("   - High-memory configuration (exp(22.5), 4P, 4R) shows queue buildup")
    if recommended_interval:
        print(
            f"   - Sample interval of {recommended_interval} time units reduces lag-1 correlation"
        )

    print("\n2. Factor Effects on Queue Length:")
    for factor, coef, p_val in significance["significant"]:
        if "Rate" in factor:
            print(f"   - Arrival rate has the largest effect ({coef:+.4f})")
        elif "Prep" in factor:
            print(f"   - Prep units significantly affect queue ({coef:+.4f})")
        elif "Recovery" in factor:
            print(f"   - Recovery units affect queue ({coef:+.4f})")

    print("\n3. Model Fit:")
    print(
        f"   - R^2 = {model['R_squared']:.4f} ({model['R_squared'] * 100:.1f}% of variance explained)"
    )

    # Check if interactions are significant
    significant_interactions = [
        f for f, c, p in significance["significant"] if f in ["AB", "AC", "BC", "ABC"]
    ]
    if significant_interactions:
        print(f"   - Significant interactions: {', '.join(significant_interactions)}")
    else:
        print("   - No significant interaction effects detected")


if __name__ == "__main__":
    main()

