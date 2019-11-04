from environment import env
import random
class Car:
    def __init__(self,controller):
        r = random.randint(0,16*9-1)
        p = random.randint(0,2)
        if p==0:
            self.intention = 'left'
        elif p==1: self.intention = 'straight'
        else: self.intention = 'right'

        self.v = 1
        self.rt = 1
        self.line = controller.lanes[r]
        # if len(self.line.cars) <10:
        self.line.add_car(self)
        if len(self.line.exits) >0 :
            self.next_connection = random.choice(self.line.exits)

