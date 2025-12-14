# ===============================
# CORE SIMULATION LOGIC
# ===============================
import simpy
import random
import statistics

from config import SIM_TIME, WARM_UP, MONITOR_INTERVAL
from distributions import (
    sample_interarrival,
    sample_prep_time,
    sample_recovery_time,
    sample_surgery_time,
)


def patient_flexible(env, pid, prep, theatre, recovery, config, metrics):
    """
    Flexible patient process with configurable distributions.

    Parameters:
    - env: SimPy environment
    - pid: Patient ID
    - prep: Preparation resource
    - theatre: Theatre resource
    - recovery: Recovery resource
    - config: ExperimentConfig object
    - metrics: Dictionary to collect metrics
    """
    arrival_time = env.now

    # Record queue length on arrival (before requesting prep)
    if env.now >= WARM_UP:
        metrics["queue_on_arrival"].append(len(prep.queue))

    # PREPARATION - request and hold prep room
    req_p = prep.request()
    yield req_p

    if env.now >= WARM_UP:
        metrics["wait_times"].append(env.now - arrival_time)

    yield env.timeout(sample_prep_time(config.prep_dist))

    # Wait for theatre (blocking: still holding prep until theatre available)
    req_t = theatre.request()
    yield req_t
    prep.release(req_p)

    # SURGERY (always exp(20))
    yield env.timeout(sample_surgery_time())

    # Wait for recovery (blocking: still holding theatre until recovery available)
    block_start = env.now
    req_r = recovery.request()
    yield req_r
    block_end = env.now

    if env.now >= WARM_UP:
        metrics["blocking_times"].append(block_end - block_start)

    theatre.release(req_t)

    # RECOVERY
    yield env.timeout(sample_recovery_time(config.recovery_dist))
    recovery.release(req_r)


def generator_flexible(env, prep, theatre, recovery, config, metrics):
    """Generate patients with configurable interarrival times."""
    pid = 0
    while True:
        pid += 1
        env.process(
            patient_flexible(env, pid, prep, theatre, recovery, config, metrics)
        )
        yield env.timeout(
            sample_interarrival(config.interarrival_dist, config.interarrival_rate)
        )


def monitor(env, prep, theatre, recovery, metrics):
    """Monitor system state at regular intervals."""
    while True:
        if env.now >= WARM_UP:
            metrics["prep_queue"].append(len(prep.queue))
            metrics["prep_util"].append(prep.count / prep.capacity)
            metrics["theatre_util"].append(theatre.count / theatre.capacity)
            metrics["recovery_util"].append(recovery.count / recovery.capacity)
            metrics["recovery_full"].append(
                1 if recovery.count == recovery.capacity else 0
            )

        yield env.timeout(MONITOR_INTERVAL)


def run_single_replication(config, seed):
    """Run a single simulation replication with given configuration."""
    random.seed(seed)

    env = simpy.Environment()
    prep = simpy.Resource(env, capacity=config.prep_units)
    theatre = simpy.Resource(env, capacity=1)
    recovery = simpy.Resource(env, capacity=config.recovery_units)

    # Initialize metrics collection
    metrics = {
        "queue_on_arrival": [],
        "wait_times": [],
        "blocking_times": [],
        "prep_queue": [],
        "prep_util": [],
        "theatre_util": [],
        "recovery_util": [],
        "recovery_full": [],
    }

    env.process(generator_flexible(env, prep, theatre, recovery, config, metrics))
    env.process(monitor(env, prep, theatre, recovery, metrics))

    env.run(until=SIM_TIME)

    # Compute summary statistics
    results = {
        "avg_queue_on_arrival": statistics.mean(metrics["queue_on_arrival"])
        if metrics["queue_on_arrival"]
        else 0,
        "avg_prep_queue": statistics.mean(metrics["prep_queue"])
        if metrics["prep_queue"]
        else 0,
        "avg_wait_time": statistics.mean(metrics["wait_times"])
        if metrics["wait_times"]
        else 0,
        "avg_prep_util": statistics.mean(metrics["prep_util"])
        if metrics["prep_util"]
        else 0,
        "avg_theatre_util": statistics.mean(metrics["theatre_util"])
        if metrics["theatre_util"]
        else 0,
        "avg_recovery_util": statistics.mean(metrics["recovery_util"])
        if metrics["recovery_util"]
        else 0,
        "blocking_prob": sum(metrics["blocking_times"]) / (SIM_TIME - WARM_UP)
        if metrics["blocking_times"]
        else 0,
        "recovery_full_prob": statistics.mean(metrics["recovery_full"])
        if metrics["recovery_full"]
        else 0,
    }

    return results


def run_correlation_replication(config, seed, num_samples=10, sample_interval=50):
    """
    Run a single replication collecting queue samples at fixed intervals
    for serial correlation analysis.

    Returns: list of queue length samples
    """
    random.seed(seed)

    env = simpy.Environment()
    prep = simpy.Resource(env, capacity=config.prep_units)
    theatre = simpy.Resource(env, capacity=1)
    recovery = simpy.Resource(env, capacity=config.recovery_units)

    metrics = {
        "queue_on_arrival": [],
        "wait_times": [],
        "blocking_times": [],
        "prep_queue": [],
        "prep_util": [],
        "theatre_util": [],
        "recovery_util": [],
        "recovery_full": [],
    }

    # Samples collected at specific intervals
    queue_samples = []

    def sample_collector(env, prep, samples, num_samples, sample_interval):
        """Collect queue samples at regular intervals after warm-up."""
        # Wait for warm-up
        yield env.timeout(WARM_UP)

        # Collect samples
        for _ in range(num_samples):
            samples.append(len(prep.queue))
            yield env.timeout(sample_interval)

    env.process(generator_flexible(env, prep, theatre, recovery, config, metrics))
    env.process(
        sample_collector(env, prep, queue_samples, num_samples, sample_interval)
    )

    # Extend simulation time to collect all samples
    extended_time = WARM_UP + (num_samples + 1) * sample_interval
    env.run(until=extended_time)

    return queue_samples
