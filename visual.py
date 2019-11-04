import pygame
import numpy as np
from environment import env

render = 0
# Screen parameters
width = 800
height = 800
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255, 0)
fpsClock = pygame.time.Clock()
fps = 400



# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 0.021
    screen_pos = np.array([center[0] * factor + car_pos[0], center[1] * factor - car_pos[1]]) / factor
    screen_pos = screen_pos.astype(int)
    return screen_pos

# Convert coordinates form screen to cartesian  (used to draw in pygame screen)
def screen_to_cartesian(screen_pos):
    factor = 0.021
    car_pos = np.array([screen_pos[0] - center[0], center[1] - screen_pos[1]]) * factor
    car_pos = car_pos.astype(float)
    return car_pos

# Drawing Board
def draw(controller):
    global render
    render +=1
    screen.fill((0, 0, 0))

    pygame.event.get()

    for line in controller.lanes:
        # if line.direction == 1:
        # pygame.draw.line(screen, green, cartesian_to_screen(line.start),cartesian_to_screen(line.end),  1)
        pygame.draw.line(screen, red, cartesian_to_screen(line.end),cartesian_to_screen(line.end + (line.start-line.end)*1/30*len(line.cars)),  5)

        pygame.draw.circle(screen, red, cartesian_to_screen(line.start), 4)

    for connection in controller.connections:
        color = green
        if connection.state:
            pygame.draw.line(screen, blue, cartesian_to_screen(connection.start.end), cartesian_to_screen(connection.end.start), 1)

    pygame.display.flip()


