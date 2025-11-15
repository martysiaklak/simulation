import simpy

def create_resources(env):
    return {
        "prep": simpy.Resource(env, capacity=3),
        "theatre": simpy.Resource(env, capacity=1),
        "recovery": simpy.Resource(env, capacity=3)
    }
