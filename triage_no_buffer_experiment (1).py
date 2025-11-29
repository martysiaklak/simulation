#!/usr/bin/env python
# coding: utf-8

# In[2]:


import simpy
import random
import statistics
import csv

MEAN_INTERARRIVAL = 25.0
MEAN_PREP_BASE = 40.0
MEAN_SURGERY_BASE = 20.0
MEAN_RECOVERY_BASE = 40.0

SIM_TIME = 1000.0
WARM_UP = 200.0
NUM_REPS = 20
MONITOR_INTERVAL = 5.0


CONFIGS = [(3,4), (3,5), (4,5)]  # (P, R)


PRIORITIES = {'CRITICAL': 0, 'URGENT': 1, 'ROUTINE': 2}
PRIORITY_PROBS = [('CRITICAL', 0.10), ('URGENT', 0.30), ('ROUTINE', 0.60)]


PREP_MULT = {'CRITICAL': 1.2, 'URGENT': 1.1, 'ROUTINE': 1.0}
SURGERY_MULT = {'CRITICAL': 1.5, 'URGENT': 1.2, 'ROUTINE': 1.0}
RECOVERY_MULT = {'CRITICAL': 1.5, 'URGENT': 1.2, 'ROUTINE': 1.0}


def exp_time(mean):
    return random.expovariate(1.0 / mean)

def sample_priority():
    r = random.random()
    acc = 0.0
    for name, p in PRIORITY_PROBS:
        acc += p
        if r <= acc:
            return name
    return PRIORITY_PROBS[-1][0]


def patient(env, pid, prep_res, theatre_res, recovery_res, stats):
    arrival = env.now
    pr_name = sample_priority()
    pr = PRIORITIES[pr_name]

    # --- REQUEST PREPARATION  ---
    req_p = prep_res.request(priority=pr)
    q_enter_prep = None
    if prep_res.count >= prep_res.capacity:
        q_enter_prep = env.now
    yield req_p
    if q_enter_prep is not None:
        stats['wait_prep_by_pr'][pr_name].append(env.now - q_enter_prep)

    # PREPARATION 
    yield env.timeout(exp_time(MEAN_PREP_BASE * PREP_MULT[pr_name]))
    stats['prep_serv_count_by_pr'][pr_name] += 1

    # --- REQUEST THEATRE 
    req_t = theatre_res.request(priority=pr)
    q_enter_theatre = None
    if theatre_res.count >= theatre_res.capacity:
        q_enter_theatre = env.now
    yield req_t
    if q_enter_theatre is not None:
        stats['wait_theatre_by_pr'][pr_name].append(env.now - q_enter_theatre)

    
    prep_res.release(req_p)

    # SURGERY
    yield env.timeout(exp_time(MEAN_SURGERY_BASE * SURGERY_MULT[pr_name]))
    stats['surgery_serv_count_by_pr'][pr_name] += 1

    # --- REQUEST RECOVERY ---
    block_start = None
    if recovery_res.count >= recovery_res.capacity:
        block_start = env.now  

    req_r = recovery_res.request(priority=pr)
    q_enter_recovery = None
    if recovery_res.count >= recovery_res.capacity:
        q_enter_recovery = env.now
    yield req_r
    if q_enter_recovery is not None:
        stats['wait_recovery_by_pr'][pr_name].append(env.now - q_enter_recovery)

  
    if block_start is not None:
        stats['total_blocked_time'] += (env.now - block_start)
        stats['blocking_events'] += 1

 
    theatre_res.release(req_t)

    # RECOVERY
    yield env.timeout(exp_time(MEAN_RECOVERY_BASE * RECOVERY_MULT[pr_name]))
    stats['recovery_serv_count_by_pr'][pr_name] += 1

    # Release recovery
    recovery_res.release(req_r)

    # throughput
    stats['throughput_by_pr'][pr_name].append(env.now - arrival)
    stats['arrival_count_by_pr'][pr_name] += 1

def generator(env, prep_res, theatre_res, recovery_res, stats):
    pid = 0
    while True:
        pid += 1
        env.process(patient(env, pid, prep_res, theatre_res, recovery_res, stats))
        yield env.timeout(exp_time(MEAN_INTERARRIVAL))

def monitor(env, prep_res, theatre_res, recovery_res, stats):
    while True:
        if env.now >= WARM_UP:
            # черга перед prep = len(prep.queue)
            stats['sampled_prep_queue'].append(len(prep_res.queue))
            stats['sampled_theatre_util'].append(theatre_res.count / float(theatre_res.capacity))
            stats['sampled_prep_util'].append(prep_res.count / float(prep_res.capacity))
            stats['sampled_recovery_full'].append(1 if recovery_res.count == recovery_res.capacity else 0)
        yield env.timeout(MONITOR_INTERVAL)


def run_once(P, R, seed):
    random.seed(seed)
    env = simpy.Environment()

    prep = simpy.PriorityResource(env, capacity=P)
    theatre = simpy.PriorityResource(env, capacity=1)
    recovery = simpy.PriorityResource(env, capacity=R)

    stats = {
        'sampled_prep_queue': [],
        'sampled_prep_util': [],
        'sampled_theatre_util': [],
        'sampled_recovery_full': [],

        'wait_prep_by_pr': {k: [] for k in PRIORITIES},
        'wait_theatre_by_pr': {k: [] for k in PRIORITIES},
        'wait_recovery_by_pr': {k: [] for k in PRIORITIES},

        'throughput_by_pr': {k: [] for k in PRIORITIES},

        'arrival_count_by_pr': {k: 0 for k in PRIORITIES},
        'prep_serv_count_by_pr': {k: 0 for k in PRIORITIES},
        'surgery_serv_count_by_pr': {k: 0 for k in PRIORITIES},
        'recovery_serv_count_by_pr': {k: 0 for k in PRIORITIES},

        'total_blocked_time': 0.0,
        'blocking_events': 0
    }

    env.process(generator(env, prep, theatre, recovery, stats))
    env.process(monitor(env, prep, theatre, recovery, stats))

    env.run(until=SIM_TIME)

    samples = len(stats['sampled_prep_queue'])
    avg_prep_q = statistics.mean(stats['sampled_prep_queue']) if samples>0 else 0.0
    avg_theatre_util = statistics.mean(stats['sampled_theatre_util']) if samples>0 else 0.0
    avg_prep_util = statistics.mean(stats['sampled_prep_util']) if samples>0 else 0.0
    recovery_full_frac = statistics.mean(stats['sampled_recovery_full']) if samples>0 else 0.0

    blocking_prob = stats['total_blocked_time'] / max(1.0, (SIM_TIME - WARM_UP))
    idle_prep = 1.0 - avg_prep_util

    res = {
        'AvgPrepQueue': avg_prep_q,
        'IdlePrep': idle_prep,
        'Blocking': blocking_prob,
        'TheatreUtil': avg_theatre_util,
        'RecoveryFull': recovery_full_frac,
        'BlockingEvents': stats['blocking_events']
    }

    return res


def run_all_experiments(configs=CONFIGS, num_reps=NUM_REPS, seed_base=1000):
    rows = []
    for (P,R) in configs:
        print(f"Running config {P}p{R}r ...")
        for rep in range(num_reps):
            seed = seed_base + rep  # same seed index across configs -> CRN
            r = run_once(P, R, seed)
            r['Config'] = f"{P}p{R}r"
            r['Rep'] = rep
            rows.append(r)
            print(f"  rep {rep} done (seed {seed})")
   
    fname = "triage_no_buffer_results.csv"
    fieldnames = ['Config','Rep','AvgPrepQueue','IdlePrep','Blocking','TheatreUtil','RecoveryFull','BlockingEvents']
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k,0) for k in fieldnames})
    print(f"All done. Results saved to {fname}")

if __name__ == "__main__":
    run_all_experiments()


# In[ ]:




