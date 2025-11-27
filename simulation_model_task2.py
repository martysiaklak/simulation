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
# PATIENT GENERATOR
# ===============================
def generator(env, prep, theatre, recovery):
    pid = 0
    while True:
        pid += 1
        env.process(patient(env, pid, prep, theatre, recovery))
        yield env.timeout(exp_time(MEAN_INTERARRIVAL))


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
# RUN FULL EXPERIMENT
# ===============================
def run_experiment(P, R, seed_base=500):
    results = []
    for rep in range(NUM_REPS):
        res = run_once(P, R, seed=seed_base + rep)
        results.append(res)
    return results


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    configs = [(3,4), (3,5), (4,5)]
    all_res = {}

    for (P, R) in configs:
        print(f"Running config {P}p{R}r...")
        all_res[(P, R)] = run_experiment(P, R)

    with open("experiment_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Config", "Rep",
            "AvgPrepQueue", "IdlePrep",
            "Blocking", "TheatreUtil",
            "RecoveryFull"
        ])

        for (P,R), reps in all_res.items():
            for i, r in enumerate(reps):
                w.writerow([
                    f"{P}p{R}r", i,
                    r["avg_prep_queue"],
                    r["idle_prep"],
                    r["blocking"],
                    r["theatre_util"],
                    r["recovery_full"]
                ])

    print("DONE. CSV saved.")
