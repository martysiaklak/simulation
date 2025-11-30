import simpy
import random
import statistics
import csv

# ===============================
# ASSIGNMENT PARAMETERS
# ===============================
MEAN_INTERARRIVAL = 25
MEAN_PREP = 40
MEAN_SURGERY = 20
MEAN_RECOVERY = 40

SIM_TIME = 1000
WARM_UP = 200
NUM_REPS = 20
MONITOR_INTERVAL = 5

# ===============================
# TWISTED SCENARIO PARAMETERS
# ===============================
# Priority levels (lower number = higher priority)
PRIORITY_CRITICAL = 0
PRIORITY_URGENT = 1
PRIORITY_ROUTINE = 2

# Priority distribution (realistic hospital)
PROB_CRITICAL = 0.10
PROB_URGENT = 0.30
PROB_ROUTINE = 0.60

# Surgery times by priority
SURGERY_CRITICAL = 30  # More complex cases
SURGERY_URGENT = 20    # Same as original
SURGERY_ROUTINE = 15   # Elective, simpler

# Adjusted interarrival to maintain 80% utilization
# Weighted avg surgery = 0.10*30 + 0.30*20 + 0.60*15 = 18
# New interarrival = 18 / 0.80 = 22.5
MEAN_INTERARRIVAL_TWISTED = 22.5


# ===============================
# EXPONENTIAL TIME SAMPLERS
# ===============================
def exp_time(mean):
    return random.expovariate(1.0 / mean)


# ===============================
# PATIENT PROCESS (CORRECT BLOCKING)
# ===============================
def patient(env, pid, prep, theatre, recovery):
    # PREPARATION
    with prep.request() as req_p:
        yield req_p
        yield env.timeout(exp_time(MEAN_PREP))

        # BLOCK on theatre WITHOUT releasing prep
        with theatre.request() as req_t:
            yield req_t  # blocks here if theatre full

            # SURGERY
            yield env.timeout(exp_time(MEAN_SURGERY))

            # BLOCK on recovery WITHOUT releasing theatre
            with recovery.request() as req_r:
                yield req_r  # blocks here if recovery full

                # RECOVERY
                yield env.timeout(exp_time(MEAN_RECOVERY))


# ===============================
# ASSIGN PATIENT PRIORITY
# ===============================
def assign_priority():
    """Assign priority based on probability distribution."""
    r = random.random()
    if r < PROB_CRITICAL:
        return PRIORITY_CRITICAL, SURGERY_CRITICAL
    elif r < PROB_CRITICAL + PROB_URGENT:
        return PRIORITY_URGENT, SURGERY_URGENT
    else:
        return PRIORITY_ROUTINE, SURGERY_ROUTINE


# ===============================
# PRIORITY PATIENT PROCESS (TWISTED)
# ===============================
def patient_twisted(env, pid, prep, theatre, recovery):
    # Assign priority at arrival
    priority, surgery_time = assign_priority()
    
    # PREPARATION (with priority)
    with prep.request(priority=priority) as req_p:
        yield req_p
        yield env.timeout(exp_time(MEAN_PREP))

        # BLOCK on theatre WITHOUT releasing prep (with priority)
        with theatre.request(priority=priority) as req_t:
            yield req_t  # blocks here if theatre full

            # SURGERY (priority-specific duration)
            yield env.timeout(exp_time(surgery_time))

            # BLOCK on recovery WITHOUT releasing theatre
            # Recovery uses regular Resource (no priority needed)
            with recovery.request() as req_r:
                yield req_r  # blocks here if recovery full

                # RECOVERY
                yield env.timeout(exp_time(MEAN_RECOVERY))


# ===============================
# PATIENT GENERATOR
# ===============================
def generator(env, prep, theatre, recovery):
    pid = 0
    while True:
        pid += 1
        env.process(patient(env, pid, prep, theatre, recovery))
        yield env.timeout(exp_time(MEAN_INTERARRIVAL))


# ===============================
# PATIENT GENERATOR (TWISTED)
# ===============================
def generator_twisted(env, prep, theatre, recovery):
    pid = 0
    while True:
        pid += 1
        env.process(patient_twisted(env, pid, prep, theatre, recovery))
        yield env.timeout(exp_time(MEAN_INTERARRIVAL_TWISTED))


# ===============================
# MONITORING
# ===============================
def monitor(env, prep, theatre, recovery,
            prep_q, theatre_util, recovery_util, recovery_full):

    while True:
        if env.now >= WARM_UP:
            prep_q.append(len(prep.queue))
            theatre_util.append(theatre.count / theatre.capacity)
            recovery_util.append(recovery.count / recovery.capacity)
            recovery_full.append(1 if recovery.count == recovery.capacity else 0)

        yield env.timeout(MONITOR_INTERVAL)


# ===============================
# RUN ONE REPLICATION
# ===============================
def run_once(P, R, seed):
    random.seed(seed)

    env = simpy.Environment()
    prep = simpy.Resource(env, capacity=P)
    theatre = simpy.Resource(env, capacity=1)
    recovery = simpy.Resource(env, capacity=R)

    # monitoring lists
    prep_q = []
    theatre_util = []
    recovery_util = []
    recovery_full = []

    env.process(generator(env, prep, theatre, recovery))
    env.process(monitor(env, prep, theatre, recovery,
                        prep_q, theatre_util, recovery_util, recovery_full))

    env.run(until=SIM_TIME)

    # METRICS
    avg_prep_q = statistics.mean(prep_q) if prep_q else 0
    avg_theatre_u = statistics.mean(theatre_util) if theatre_util else 0
    avg_recovery_full = statistics.mean(recovery_full) if recovery_full else 0
    blocking_prob = 1 - avg_theatre_u
    idle_prep = 1 - (prep.count / prep.capacity)

    return {
        "avg_prep_queue": avg_prep_q,
        "idle_prep": idle_prep,
        "blocking": blocking_prob,
        "theatre_util": avg_theatre_u,
        "recovery_full": avg_recovery_full
    }


# ===============================
# RUN ONE REPLICATION (TWISTED)
# ===============================
def run_once_twisted(P, R, seed):
    random.seed(seed)

    env = simpy.Environment()
    # Use PriorityResource for prep and theatre
    prep = simpy.PriorityResource(env, capacity=P)
    theatre = simpy.PriorityResource(env, capacity=1)
    recovery = simpy.Resource(env, capacity=R)

    # monitoring lists
    prep_q = []
    theatre_util = []
    recovery_util = []
    recovery_full = []

    env.process(generator_twisted(env, prep, theatre, recovery))
    env.process(monitor(env, prep, theatre, recovery,
                        prep_q, theatre_util, recovery_util, recovery_full))

    env.run(until=SIM_TIME)

    # METRICS
    avg_prep_q = statistics.mean(prep_q) if prep_q else 0
    avg_theatre_u = statistics.mean(theatre_util) if theatre_util else 0
    avg_recovery_full = statistics.mean(recovery_full) if recovery_full else 0
    blocking_prob = 1 - avg_theatre_u
    idle_prep = 1 - (prep.count / prep.capacity)

    return {
        "avg_prep_queue": avg_prep_q,
        "idle_prep": idle_prep,
        "blocking": blocking_prob,
        "theatre_util": avg_theatre_u,
        "recovery_full": avg_recovery_full
    }


# ===============================
# RUN FULL EXPERIMENT
# ===============================
def run_experiment(P, R, seed_base=500):
    results = []
    for rep in range(NUM_REPS):
        res = run_once(P, R, seed=seed_base + rep)
        results.append(res)
    return results


# ===============================
# RUN FULL EXPERIMENT (TWISTED)
# ===============================
def run_experiment_twisted(P, R, seed_base=500):
    results = []
    for rep in range(NUM_REPS):
        res = run_once_twisted(P, R, seed=seed_base + rep)
        results.append(res)
    return results


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    configs = [(3,4), (3,5), (4,5)]
    all_res_original = {}
    all_res_twisted = {}

    # Run original scenarios
    for (P, R) in configs:
        print(f"Running ORIGINAL config {P}p{R}r...")
        all_res_original[(P, R)] = run_experiment(P, R)

    # Run twisted scenarios (same seeds for paired comparison)
    for (P, R) in configs:
        print(f"Running TWISTED config {P}p{R}r...")
        all_res_twisted[(P, R)] = run_experiment_twisted(P, R)

    # Save results to CSV
    with open("experiment_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Scenario", "Config", "Rep",
            "AvgPrepQueue", "IdlePrep",
            "Blocking", "TheatreUtil",
            "RecoveryFull"
        ])

        # Write original results
        for (P,R), reps in all_res_original.items():
            for i, r in enumerate(reps):
                w.writerow([
                    "Original", f"{P}p{R}r", i,
                    r["avg_prep_queue"],
                    r["idle_prep"],
                    r["blocking"],
                    r["theatre_util"],
                    r["recovery_full"]
                ])

        # Write twisted results
        for (P,R), reps in all_res_twisted.items():
            for i, r in enumerate(reps):
                w.writerow([
                    "Twisted", f"{P}p{R}r", i,
                    r["avg_prep_queue"],
                    r["idle_prep"],
                    r["blocking"],
                    r["theatre_util"],
                    r["recovery_full"]
                ])

    print("DONE. CSV saved.")

    # Print summary statistics for quick comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON (Mean values across 20 replications)")
    print("="*60)
    
    for (P, R) in configs:
        orig = all_res_original[(P, R)]
        twist = all_res_twisted[(P, R)]
        
        print(f"\nConfig {P}p{R}r:")
        print("-" * 40)
        
        metrics = ["avg_prep_queue", "blocking", "theatre_util", "recovery_full"]
        labels = ["Avg Prep Queue", "Blocking Prob", "Theatre Util", "Recovery Full"]
        
        for metric, label in zip(metrics, labels):
            orig_mean = statistics.mean([r[metric] for r in orig])
            twist_mean = statistics.mean([r[metric] for r in twist])
            diff = twist_mean - orig_mean
            print(f"  {label:16s}: Original={orig_mean:.4f}, Twisted={twist_mean:.4f}, Diff={diff:+.4f}")
