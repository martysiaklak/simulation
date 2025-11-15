import simpy
from parameters import prep_time, surgery_time, recovery_time

def patient(env, name, resources):
    """Lifecycle of one patient through preparation, surgery, recovery."""

    # --- Preparation phase ---
    arrival_time = env.now
    print(f"{env.now:.1f}: {name} arrives")

    with resources["prep"].request() as req:
        yield req
        print(f"{env.now:.1f}: {name} starts preparation")
        yield env.timeout(prep_time())
        print(f"{env.now:.1f}: {name} ends preparation")

    # --- Surgery phase ---
    with resources["theatre"].request() as req:
        yield req
        print(f"{env.now:.1f}: {name} enters operating theatre")
        yield env.timeout(surgery_time())
        print(f"{env.now:.1f}: {name} ends surgery")

    # --- Recovery phase ---
    with resources["recovery"].request() as req:
        yield req
        print(f"{env.now:.1f}: {name} starts recovery")
        yield env.timeout(recovery_time())
        print(f"{env.now:.1f}: {name} ends recovery and leaves hospital")

    total_time = env.now - arrival_time
    print(f"{env.now:.1f}: {name} total time in system = {total_time:.1f}")
