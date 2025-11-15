import random

# Fixed parameters
P = 3   # number of preparation rooms
R = 3   # number of recovery beds

# Time parameters (mean values)
MEAN_INTERARRIVAL = 25
MEAN_PREP = 40
MEAN_SURGERY = 20
MEAN_RECOVERY = 40

# Random time generators
def interarrival_time():
    return random.expovariate(1.0 / MEAN_INTERARRIVAL)

def prep_time():
    return random.expovariate(1.0 / MEAN_PREP)

def surgery_time():
    return random.expovariate(1.0 / MEAN_SURGERY)

def recovery_time():
    return random.expovariate(1.0 / MEAN_RECOVERY)
