# Author:   Max Martinez Ruts
# Creation: 2019

import pygame
import numpy as np
import car as cr
import visual as vs


# Simulation loop
def simulate(controllers, tmax):
    # for controller in controllers:
        # controller.brain.model.load_weights('weights')

    # Initialize clock
    t= 0

    while t < tmax:

        scores = []



        for controller in controllers:

            scores.append(controller.score)
            if t==len(controller.cars)*9:
                controller.cars.append(cr.Car(controller))
            if t%50 == 0:

                controller.update()
            if t % 1 == 0:
                for tl in controller.traffic_lights:
                    tl.check()
                    for connection in tl.phase:
                        connection.check_traffic()

            if t>tmax/3*2:
                controller.update()
        if t%200==0:
            vs.draw(controllers[scores.index(min(scores))])

        t += 1
