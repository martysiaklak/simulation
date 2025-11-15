import simpy
from parameters import *
from resources import create_resources
from generator import patient_generator

SIM_TIME = 500  # total simulation time

def run_simulation():
    env = simpy.Environment()
    resources = create_resources(env)

    # Start the patient arrival process
    env.process(patient_generator(env, resources))

    print("Starting simulation...")
    env.run(until=SIM_TIME)
    print("Simulation completed.")

if __name__ == "__main__":
    run_simulation()
