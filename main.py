import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation



import car as cr
import line as ln
import road as rd
import traffic_light as tl
import zebra as zb
import random
from environment import env
import visual as vs
import connections as cn

import controller as ct
import simulation as sm
import genetic_algorithm as ga
import ga_plain as gp
import ppo
import torch
periods = np.random.randint(0, 3000, (100000, 4))



# Create 4 roads, 16 lanes,

road1 = rd.Road(np.array([0,-2]),np.array([0,-1]))
road2 = rd.Road(np.array([2,0]),np.array([1,0]))
road3 = rd.Road(np.array([0,2]),np.array([0,1]))
road4 = rd.Road(np.array([-2,0]),np.array([-1,0]))


road5 = rd.Road(np.array([5,-2]),np.array([5,-1]))
road6 = rd.Road(np.array([7,0]),np.array([6,0]))
road7 = rd.Road(np.array([5,2]),np.array([5,1]))
road8 = rd.Road(np.array([3,0]),np.array([4,0]))

road9 = rd.Road(np.array([5,-7]),np.array([5,-6]))
road10 = rd.Road(np.array([7,-5]),np.array([6,-5]))
road11 = rd.Road(np.array([5,-3]),np.array([5,-4]))
road12 = rd.Road(np.array([3,-5]),np.array([4,-5]))

road13 = rd.Road(np.array([0,-7]),np.array([0,-6]))
road14 = rd.Road(np.array([2,-5]),np.array([1,-5]))
road15 = rd.Road(np.array([0,-3]),np.array([0,-4]))
road16 = rd.Road(np.array([-2,-5]),np.array([-1,-5]))

road17 = rd.Road(np.array([-5,-7]),np.array([-5,-6]))
road18 = rd.Road(np.array([-3,-5]),np.array([-4,-5]))
road19 = rd.Road(np.array([-5,-3]),np.array([-5,-4]))
road20 = rd.Road(np.array([-7,-5]),np.array([-6,-5]))

road21 = rd.Road(np.array([-5,-2]),np.array([-5,-1]))
road22 = rd.Road(np.array([-3,0]),np.array([-4,0]))
road23 = rd.Road(np.array([-5,2]),np.array([-5,1]))
road24 = rd.Road(np.array([-7,0]),np.array([-6,0]))

road25 = rd.Road(np.array([-5,3]),np.array([-5,4]))
road26 = rd.Road(np.array([-3,5]),np.array([-4,5]))
road27 = rd.Road(np.array([-5,7]),np.array([-5,6]))
road28 = rd.Road(np.array([-7,5]),np.array([-6,5]))

road29 = rd.Road(np.array([0,3]),np.array([0,4]))
road30 = rd.Road(np.array([2,5]),np.array([1,5]))
road31 = rd.Road(np.array([0,7]),np.array([0,6]))
road32 = rd.Road(np.array([-2,5]),np.array([-1,5]))

road33 = rd.Road(np.array([5,3]),np.array([5,4]))
road34 = rd.Road(np.array([7,5]),np.array([6,5]))
road35 = rd.Road(np.array([5,7]),np.array([5,6]))
road36 = rd.Road(np.array([3,5]),np.array([4,5]))

env.roads = {0:road1,1:road2,2:road3,3:road4,4:road5,5:road6,6:road7,7:road8,8:road9,9:road10,10:road11,11:road12,12:road13,13:road14,14:road15,15:road16, 16:road17,17:road18,18:road19,19:road20, 20:road21, 21:road22,22:road23,23:road24, 24:road25, 25:road26, 26:road27, 27:road28,28:road29,29:road30,30:road31,31:road32,32:road33,33:road34,34:road35,35:road36}


model, controller = ppo.train()
model.actor.layer[0].weight
state = controller.get_state()

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

for t in range(100000):
    vs.draw(controller)
    state = torch.FloatTensor([state]).to(device)
    print(state)
    dist, value = model(state)
    action = dist.sample()
    print(action)
    state, _, _, _ = controller.step(action)


for i in range(0,10):
    controller = ct.Controller()
    controller.brain.mutate()
    env.controllers.append(controller)

t=0

for generation in range(1,100):
    print('Generation: ',generation)
    # Simulation loop
    sm.simulate(env.controllers, 20000)

    # Dramatize scores and determine total score of generation
    collective_score = 0
    scores_raw = []
    for controller in env.controllers:
        scores_raw.append(controller.score)
    print('Min:',min(scores_raw))
    print('Avg:',np.mean(np.array(scores_raw)))

    scores = []

    for controller in env.controllers:
        # Dramatize scores
        controller.score = max(scores_raw)-controller.score

        # Add score of vehicle to collective score
        collective_score += controller.score
        scores.append(controller.score)

    # # Save histograms and progression over generations
    # pl.save(scores, generation, tmax, population_size, start_time)

    # Determine fitness scores
    for controller in env.controllers:
        controller.fitness = controller.score/collective_score

    # Create next generation
    ga.next_generation(env.controllers)

# Close simulation
pygame.quit()
#
# while True:
#
#     t+=1
#     if t%60==0:
#         vs.draw(controller)
#         # print(t)
#     if t%11 ==0:
#
#         controller.cars.append(cr.Car(controller))
#
#     if t%100 == 0:
#         controller.evaluate()
#     # for traffic_light in controller.traffic_lights:
#     #     traffic_light.check(t)