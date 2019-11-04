import numpy as np
import random
class Line:
    def __init__(self, road, direction, layer):
        self.left = []
        self.right = []
        self.straight = []
        self.layer = layer
        self.direction = direction
        if direction == 1:
            p1 = road.start
            p2 = road.end
        else:
            p1 = road.end
            p2 = road.start
        self.vector = ((p1-p2))
        self.heading = self.vector/np.linalg.norm(p2-p1)
        R = np.array([[0,-1],[1,0]])
        perp = np.matmul(R,self.heading)
        self.start = p1 + perp*(layer*0.5+0.25)
        self.end = p2 + perp*(layer*0.5+0.25)
        self.cars = []
        self.entries = []
        self.exits = []



        # print(p1,self.start, self.heading,perp)

    def add_car(self, car):
        if len(self.exits) >0:
            self.cars.append(car)
            if len(self.exits)>0:
                car.next_connection = random.choice(self.exits)
    def remove_car(self,car):
        self.cars.remove(car)
    def add_entry(self,connection):
        self.entries.append(connection)
    def add_exit(self, connection):
        self.exits.append(connection)

    # def flow_car(self,car):
    #     for exit in self.exits:
    #         if car.next_connection == exit and len(exit.start.cars) < 30:
    #             print('flow')
    #             self.remove_car(car)
    #             exit.end.add_car(car)
