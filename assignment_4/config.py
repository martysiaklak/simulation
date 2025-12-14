# ===============================
# CONFIGURATION PARAMETERS
# ===============================

# Output directory
OUTPUT_DIR = "out"
PLOTS_DIR = "out/plots"

# Simulation parameters
MEAN_SURGERY = 20  # Fixed exp(20) for all scenarios
SIM_TIME = 1000
WARM_UP = 200
NUM_REPS = 20
MONITOR_INTERVAL = 5

# Factor levels for 2^3 design
# Factor A: Interarrival Rate
INTERARRIVAL_LOW = 25  # Level -1: Lower arrival rate (less congestion)
INTERARRIVAL_HIGH = 22.5  # Level +1: Higher arrival rate (more congestion)

# Factor B: Prep Units
PREP_UNITS_LOW = 4  # Level -1
PREP_UNITS_HIGH = 5  # Level +1

# Factor C: Recovery Units
RECOVERY_UNITS_LOW = 4  # Level -1
RECOVERY_UNITS_HIGH = 5  # Level +1

# Default distribution parameters (held constant in 2^3 design)
MEAN_PREP = 40
MEAN_RECOVERY = 40

# Serial correlation analysis parameters
CORRELATION_NUM_RUNS = 10
CORRELATION_NUM_SAMPLES = 10
CORRELATION_SAMPLE_INTERVALS = [20, 50, 100, 150, 200]
CORRELATION_MAX_LAG = 5
