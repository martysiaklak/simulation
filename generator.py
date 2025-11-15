import simpy
from parameters import interarrival_time
from patient_process import patient

def patient_generator(env, resources):
    i = 0
    while True:
        i += 1
        env.process(patient(env, f"Patient {i}", resources))
        yield env.timeout(interarrival_time())
