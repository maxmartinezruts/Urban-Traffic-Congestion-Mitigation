# Author:   Max Martinez Ruts
# Creation: 2019

import pygame
import genetic_algorithm as ga
import vehicle as vh
import simulation as sm
import plotter as pl
import time
import os

start_time = str(int(time.time()))

# Crate folders for restults
os.makedirs('results/'+start_time+'/progress_charts')
os.makedirs('results/'+start_time+'/histogram_charts')

tmax = 500                  # Simulation time
population_size = 50       # Number of vehicles
vehicles = []               # List containing all vehicles

# Create all vehicles
for n in range(population_size):
    vehicles.append(vh.Vehicle())
    vehicles[-1].brain.mutate()


for generation in range(1,100):

    # Simulation loop
    sm.simulate(vehicles, tmax, generation)

    # Dramatize scores and determine total score of generation
    collective_score = 0
    scores = []
    for vehicle in vehicles:
        # Dramatize scores
        vehicle.score = vehicle.score**1.3

        # Add score of vehicle to collective score
        collective_score += vehicle.score
        scores.append(vehicle.score)

    # Save histograms and progression over generations
    pl.save(scores, generation, tmax, population_size, start_time)

    # Determine fitness scores
    for vehicle in vehicles:
        vehicle.fitness = vehicle.score/collective_score

    # Create next generation
    ga.next_generation(vehicles)

# Close simulation
pygame.quit()

