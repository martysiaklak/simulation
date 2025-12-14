# ===============================
# EXPERIMENT CONFIGURATION
# ===============================
from config import NUM_REPS
from simulation import run_single_replication


class ExperimentConfig:
    """Configuration class for a single experiment."""

    def __init__(
        self,
        name,
        interarrival_rate,
        prep_units,
        recovery_units,
        interarrival_dist="exp",
        prep_dist="exp",
        recovery_dist="exp",
    ):
        self.name = name
        self.interarrival_rate = interarrival_rate  # 'low' or 'high'
        self.prep_units = prep_units
        self.recovery_units = recovery_units
        self.interarrival_dist = interarrival_dist
        self.prep_dist = prep_dist
        self.recovery_dist = recovery_dist

    def get_coded_levels(self):
        """Return coded levels (-1, +1) for regression analysis."""
        A = 1 if self.interarrival_rate == "high" else -1
        B = 1 if self.prep_units == 5 else -1
        C = 1 if self.recovery_units == 5 else -1
        return A, B, C

    def __repr__(self):
        return (
            f"Config({self.name}: rate={self.interarrival_rate}, "
            f"prep={self.prep_units}, recovery={self.recovery_units})"
        )


def create_factorial_design():
    """
    Create the 2^3 full factorial design matrix.

    Factors:
    - A: Interarrival Rate (low=25, high=22.5)
    - B: Prep Units (4, 5)
    - C: Recovery Units (4, 5)

    Returns: List of ExperimentConfig objects
    """
    design = [
        # Exp 1: A=-1, B=-1, C=-1
        ExperimentConfig("Exp1_---", "low", 4, 4),
        # Exp 2: A=+1, B=-1, C=-1
        ExperimentConfig("Exp2_+--", "high", 4, 4),
        # Exp 3: A=-1, B=+1, C=-1
        ExperimentConfig("Exp3_-+-", "low", 5, 4),
        # Exp 4: A=+1, B=+1, C=-1
        ExperimentConfig("Exp4_++-", "high", 5, 4),
        # Exp 5: A=-1, B=-1, C=+1
        ExperimentConfig("Exp5_--+", "low", 4, 5),
        # Exp 6: A=+1, B=-1, C=+1
        ExperimentConfig("Exp6_+-+", "high", 4, 5),
        # Exp 7: A=-1, B=+1, C=+1
        ExperimentConfig("Exp7_-++", "low", 5, 5),
        # Exp 8: A=+1, B=+1, C=+1
        ExperimentConfig("Exp8_+++", "high", 5, 5),
    ]
    return design


def run_factorial_experiments(design, num_reps=NUM_REPS, seed_base=500):
    """
    Run all experiments in the factorial design.

    Parameters:
    - design: List of ExperimentConfig objects
    - num_reps: Number of replications per experiment
    - seed_base: Base seed for reproducibility

    Returns: Dictionary mapping config names to list of results
    """
    all_results = {}

    for exp_idx, config in enumerate(design):
        print(f"Running {config.name}...")
        exp_results = []

        for rep in range(num_reps):
            seed = seed_base + exp_idx * 100 + rep
            result = run_single_replication(config, seed)
            result["config"] = config.name
            result["rep"] = rep
            A, B, C = config.get_coded_levels()
            result["A"] = A
            result["B"] = B
            result["C"] = C
            exp_results.append(result)

        all_results[config.name] = exp_results

    return all_results
