from environment import env
import numpy as np
class Traffic_Light:
    def __init__(self,types,period,delay,controller):
        self.i = 0
        self.phase = []
        self.types = types
        self.period = period
        self.delay = delay
        phases = {0:[],1:[],2:[],3:[]}
        self.i = -1
        self.ct = controller
        self.timers = [np.random.rand()*100,0,0,0]
        self.t_last = 0
        self.vel = 1

        for connection in self.ct.connections:
            for i in range(len(types)):
                for type in types[i]:
                    if connection.type == type:
                        phases[i].append(connection)

        self.phases = phases
        self.phase = phases[0]
        self.update_state(True)

    # def check(self):
    #     self.timers[self.i] +=1


    def update_state(self,allow = False):
        if (self.t_last>2000 or allow):
            self.i +=1
            self.i = self.i%4
            self.t_last = 0
            for connection in self.phase:
                connection.change_state(False)
            self.phase = self.phases[self.i]
            for connection in self.phase:
                connection.change_state(True)
        # if allow:
        #     for connection in self.phase:
        #         connection.check_traffic()




