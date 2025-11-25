#!/usr/bin/env python
# coding: utf-8

# In[5]:


import simpy
import random
import statistics
import csv

# -----------------------------
# PARAMETERS
# -----------------------------
SIM_TIME = 1000  # total simulation time
WARM_UP = 100    # warm-up period
NUM_REPS = 20    # number of replications

# -----------------------------
# PATIENT PROCESS
# -----------------------------
def patient_process(env, patient_id, prep, theatre, recovery):
    # Preparation
    with prep.request() as req1:
        yield req1
        prep_time = random.expovariate(1 / 20)  # mean 20
        yield env.timeout(prep_time)

    # Operating Theatre
    with theatre.request() as req2:
        yield req2
        surgery_time = random.expovariate(1 / 60)  # mean 60
        yield env.timeout(surgery_time)

    # Recovery
    with recovery.request() as req3:
        yield req3
        recovery_time = random.expovariate(1 / 30)  # mean 30
        yield env.timeout(recovery_time)

# -----------------------------
# PATIENT GENERATOR
# -----------------------------
def patient_generator(env, prep, theatre, recovery):
    i = 0
    while True:
        i += 1
        env.process(patient_process(env, i, prep, theatre, recovery))
        interarrival = random.expovariate(1 / 10)  # mean 10
        yield env.timeout(interarrival)

# -----------------------------
# MONITORING
# -----------------------------
def monitor(env, prep, theatre, recovery, prep_queue_lengths, theatre_util, recovery_util, recovery_full):
    while True:
        if env.now > WARM_UP:
            prep_queue_lengths.append(len(prep.queue))
            theatre_util.append(theatre.count / theatre.capacity)
            recovery_util.append(recovery.count / recovery.capacity)
            recovery_full.append(recovery.count == recovery.capacity)
        yield env.timeout(2)  # check every 2 time units

# -----------------------------
# RUN ONE SIMULATION
# -----------------------------
def run_simulation_once(P_value, R_value):
    env = simpy.Environment()
    prep = simpy.Resource(env, capacity=P_value)
    theatre = simpy.Resource(env, capacity=1)
    recovery = simpy.Resource(env, capacity=R_value)

    # Lists for monitoring
    prep_queue_lengths = []
    theatre_util = []
    recovery_util = []
    recovery_full = []

    env.process(patient_generator(env, prep, theatre, recovery))
    env.process(monitor(env, prep, theatre, recovery, prep_queue_lengths, theatre_util, recovery_util, recovery_full))

    env.run(until=SIM_TIME)

    # Compute metrics
    avg_prep_queue = statistics.mean(prep_queue_lengths) if prep_queue_lengths else 0
    avg_theatre_util = statistics.mean(theatre_util) if theatre_util else 0
    idle_prep = 1 - (sum([prep.count / prep.capacity for prep in [prep]]) / 1)  # approximate idle
    blocking = 1 - avg_theatre_util
    freq_recovery_full = statistics.mean(recovery_full) if recovery_full else 0

    return {
        "avg_prep_queue": avg_prep_queue,
        "idle_prep": idle_prep,
        "blocking": blocking,
        "theatre_util": avg_theatre_util,
        "recovery_full": freq_recovery_full
    }

# -----------------------------
# RUN EXPERIMENT (20 REP)
# -----------------------------
def run_experiment(P_value, R_value, seed_base=1000):
    results = []
    for rep in range(NUM_REPS):
        random.seed(seed_base + rep)
        result = run_simulation_once(P_value, R_value)
        results.append(result)
    return results

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    configurations = [(3,4), (3,5), (4,5)]
    all_results = {}

    for (P_val, R_val) in configurations:
        print(f"\n=== Running configuration {P_val} prep, {R_val} recovery ===")
        results = run_experiment(P_val, R_val)
        all_results[(P_val, R_val)] = results

    # Save to CSV
    with open("experiment_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Rep", "AvgPrepQueue", "IdlePrep", "Blocking", "TheatreUtil", "RecoveryFull"])
        for config, values in all_results.items():
            P_val, R_val = config
            for i, row in enumerate(values):
                writer.writerow([
                    f"{P_val}p{R_val}r",
                    i,
                    row["avg_prep_queue"],
                    row["idle_prep"],
                    row["blocking"],
                    row["theatre_util"],
                    row["recovery_full"]
                ])
    print("\nDone! Results saved to experiment_results.csv")


# In[2]:


get_ipython().system('pip install simpy')


# In[ ]:




