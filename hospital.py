"""
Hospital simulation with blocking at both preparation and operating stages.
"""

import simpy
import random
import statistics
import csv


# ---------- Parameters ----------
P = 3      # number of preparation rooms
R = 3      # number of recovery beds
MEAN_INTERARRIVAL = 25
MEAN_PREP = 40
MEAN_SURGERY = 20
MEAN_RECOVERY = 40
SIM_TIME = 500
MONITOR_INTERVAL = 10

random.seed(42)


# ---------- Random time generators ----------
def interarrival_time():
    return random.expovariate(1.0 / MEAN_INTERARRIVAL)

def prep_time():
    return random.expovariate(1.0 / MEAN_PREP)

def surgery_time():
    return random.expovariate(1.0 / MEAN_SURGERY)

def recovery_time():
    return random.expovariate(1.0 / MEAN_RECOVERY)

# ---------- Global statistics containers ----------
prep_queue_lengths, theatre_queue_lengths, recovery_queue_lengths = [], [], []
prep_util, theatre_util, recovery_util = [], [], []


# ---------- Patient process ----------
def patient(env, name, prep_res, theatre_res, recovery_res):
    """Full patient lifecycle with blocking between stages."""
    arrival_time = env.now
    print(f"{env.now:.1f}: {name} arrives")

    # --- Preparation phase (blocking if theatre full) ---
    with prep_res.request() as req_prep:
        yield req_prep
        print(f"{env.now:.1f}: {name} starts preparation")
        yield env.timeout(prep_time())
        print(f"{env.now:.1f}: {name} finished prep, waiting for theatre")

        # Wait for available theatre BEFORE releasing prep room (blocking)
        with theatre_res.request() as req_theatre:
            yield req_theatre     # blocks here if theatre busy
            print(f"{env.now:.1f}: {name} enters theatre (prep still occupied)")
            # Now prep can be released
            # (Leaving 'with' block releases prep automatically)

            # --- Surgery phase (blocking if recovery full) ---
            yield env.timeout(surgery_time())
            print(f"{env.now:.1f}: {name} finished surgery, waiting for recovery")

            # Wait for available recovery bed BEFORE leaving theatre (blocking)
            with recovery_res.request() as req_recovery:
                yield req_recovery   # blocks here if recovery full
                print(f"{env.now:.1f}: {name} enters recovery (theatre still occupied)")
                # leaving 'with' block releases theatre automatically

                # --- Recovery phase ---
                yield env.timeout(recovery_time())
                print(f"{env.now:.1f}: {name} finishes recovery and leaves hospital")

    total_time = env.now - arrival_time
    print(f"{env.now:.1f}: {name} total time in system = {total_time:.1f}")


# ---------- Patient generator ----------
def patient_generator(env, prep_res, theatre_res, recovery_res):
    i = 0
    while True:
        i += 1
        env.process(patient(env,
                            f"Patient {i}",
                            prep_res,
                            theatre_res,
                            recovery_res))
        yield env.timeout(interarrival_time())

# ---------- Monitoring process ----------
def monitor(env, prep, theatre, recovery):
    """Takes snapshots every MONITOR_INTERVAL time units."""
    while True:
        prep_queue_lengths.append(len(prep.queue))
        theatre_queue_lengths.append(len(theatre.queue))
        recovery_queue_lengths.append(len(recovery.queue))
        prep_util.append(prep.count / prep.capacity)
        theatre_util.append(theatre.count / theatre.capacity)
        recovery_util.append(recovery.count / recovery.capacity)
        yield env.timeout(MONITOR_INTERVAL)


# ---------- Simulation runner ----------
def run_simulation():
    env = simpy.Environment()
    prep = simpy.Resource(env, capacity=P)
    theatre = simpy.Resource(env, capacity=1)
    recovery = simpy.Resource(env, capacity=R)

    # --- Start generator and monitoring processes ---
    env.process(patient_generator(env, prep, theatre, recovery))
    env.process(monitor(env, prep, theatre, recovery))

    env.run(until=SIM_TIME)

    print("\n=== MONITORING RESULTS ===")
    print(f"Average preparation queue length: {statistics.mean(prep_queue_lengths):.2f}")
    print(f"Average theatre queue length: {statistics.mean(theatre_queue_lengths):.2f}")
    print(f"Average recovery queue length: {statistics.mean(recovery_queue_lengths):.2f}")
    print(f"Average utilisation of preparation rooms: {100*statistics.mean(prep_util):.1f}%")
    print(f"Average utilisation of operating theatre: {100*statistics.mean(theatre_util):.1f}%")
    print(f"Average utilisation of recovery rooms: {100*statistics.mean(recovery_util):.1f}%")

    save_monitor_results()
    save_summary_results()

def save_monitor_results():
    with open("monitor_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "PrepQueue", "TheatreQueue", "RecoveryQueue",
                         "PrepUtil", "TheatreUtil", "RecoveryUtil"])
        for i in range(len(prep_queue_lengths)):
            writer.writerow([
                i * MONITOR_INTERVAL,
                prep_queue_lengths[i],
                theatre_queue_lengths[i],
                recovery_queue_lengths[i],
                prep_util[i],
                theatre_util[i],
                recovery_util[i],
            ])

# ---------- Save summary statistics ----------
def save_summary_results():
    with open("summary_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Avg prep queue", round(statistics.mean(prep_queue_lengths), 2)])
        writer.writerow(["Avg theatre queue", round(statistics.mean(theatre_queue_lengths), 2)])
        writer.writerow(["Avg recovery queue", round(statistics.mean(recovery_queue_lengths), 2)])
        writer.writerow(["Prep utilisation (%)", round(100 * statistics.mean(prep_util), 1)])
        writer.writerow(["Theatre utilisation (%)", round(100 * statistics.mean(theatre_util), 1)])
        writer.writerow(["Recovery utilisation (%)", round(100 * statistics.mean(recovery_util), 1)])

if __name__ == "__main__":
    run_simulation()
