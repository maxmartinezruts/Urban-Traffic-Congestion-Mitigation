# Author:   Max Martinez Ruts
# Creation: 2019

from environment import env
import numpy as np

# Pick a progenitor
def pick_progenitor(vehicles):
    index = 0
    r = np.random.uniform(0, 1)
    while r > 0:
        r -= vehicles[index].fitness
        index += 1
    index -= 1
    return index

# Create new generation of vehicles
def next_generation(controllers):
    p1s = []
    p2s = []
    for controller in env.controllers:


        # Select first progenitor
        # print(controllers)
        #
        # print(pick_progenitor(controllers))
        p1s.append(controllers[pick_progenitor(controllers)].brain.model.get_weights())

        # Select second progenitor
        p2s.append(controllers[pick_progenitor(controllers)].brain.model.get_weights())

    for v in range(len(controllers)):
        # Redefine vehicle to the new generated vehicle from its progenitors
        child = controllers[v]
        child.reset()
        child.brain.crossover(p1s[v],p2s[v])
        child.brain.mutate()
        child.brain.create()
        controllers[v] = child