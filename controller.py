from environment import env
import numpy as np
import neural_network as nn
import connections as cn
import line as ln
import traffic_light as tl
import random
import visual as vs
import car as cr

class Controller:
    def __init__(self, model=0):
        # Initialize scores
        self.score = 0
        self.fitness = 0
        self.lanes = []
        self.cars = []
        self.connections = []
        self.traffic_lights = []

        print(len(env.roads))
        for i in env.roads:
            line1 = ln.Line(env.roads[i], -1, 1)
            line2 = ln.Line(env.roads[i], -1, 0)
            line3 = ln.Line(env.roads[i], 1, 0)
            line4 = ln.Line(env.roads[i], 1, 1)
            self.lanes.append(line1)
            self.lanes.append(line2)
            self.lanes.append(line3)
            self.lanes.append(line4)

        for i in range(len(self.lanes)):
            if int((i % 16) / 4) == 0 or int((i % 16) / 4) == 2:
                type = 'v'
            else:
                type = 'h'
            if self.lanes[i].layer == 0 and self.lanes[i].direction == 1:
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 10) % 16],
                                                     str(1 + int(i / 16)) + 'left' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 11) % 16],
                                                     str(1 + int(i / 16)) + 'left' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 6) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 7) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))

            if self.lanes[i].layer == 1 and self.lanes[i].direction == 1:
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 1) % 16],
                                                     str(1 + int(i / 16)) + 'right' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 2) % 16],
                                                     str(1 + int(i / 16)) + 'right' + type, False))

                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 5) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 6) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))

        self.connections.append(cn.Connection(self.lanes[4], self.lanes[31], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[5], self.lanes[30], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[29], self.lanes[6], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[28], self.lanes[7], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[16], self.lanes[43], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[17], self.lanes[42], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[41], self.lanes[18], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[40], self.lanes[19], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[44], self.lanes[55], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[45], self.lanes[54], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[53], self.lanes[46], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[52], self.lanes[47], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[56], self.lanes[3], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[57], self.lanes[2], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[1], self.lanes[58], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[0], self.lanes[59], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[63], self.lanes[68], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[62], self.lanes[69], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[70], self.lanes[61], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[71], self.lanes[60], 'parallel', True))

        self.traffic_lights.append(
            tl.Traffic_Light([['1lefth', '1rightv'], ['1straightv'], ['1leftv', '1righth'], ['1straighth']], 2000, 0,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['2lefth', '2rightv'], ['2straightv'], ['2leftv', '2righth'], ['2straighth']], 2000,
                             4000,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['3lefth', '3rightv'], ['3straightv'], ['3leftv', '3righth'], ['3straighth']], 2000,
                             0000,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['4lefth', '4rightv'], ['4straightv'], ['4leftv', '4righth'], ['4straighth']], 2000,
                             4000,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['5lefth', '5rightv'], ['5straightv'], ['5leftv', '4righth'], ['5straighth']], 2000,
                             4000, self))

        self.traffic_lights.append(tl.Traffic_Light([['parallel']], 200, 0,self))
        # Generate ANN
        self.brain = nn.Brain(model)

    def get_state(self):
        input = np.zeros(4*9)
        for i in range(0, 9):
            # print(self.traffic_lights[i].i,i*4+self.traffic_lights[i].i)
            input[i*4+self.traffic_lights[i].i] =(max(min(self.traffic_lights[i].t_last / 3000, 1.0), 0.1))

        # input = np.random.rand(16)
        input = list(input)
        for lane in self.lanes:
            input.append(len(lane.cars)/30)
        return list(input)

    def evaluate_nn(self):


        input = self.get_state()
        input  =np.array([input], )

        output = self.brain.model.predict(input)[0]
        # print(output)
        # output = [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0]
        tl1 = output[:4]
        i = np.argmax( tl1)
        self.traffic_lights[0].update_state(i)

        tl2 = output[4:8]
        i = np.argmax(tl2)
        self.traffic_lights[1].update_state(i)

        tl3 = output[8:12]
        i = np.argmax(tl3)
        self.traffic_lights[2].update_state(i)

        tl4 = output[12:]
        i = np.argmax(tl4)
        self.traffic_lights[3].update_state(i)

    def update(self):


        for lane in self.lanes:

            # Add score contribution
            self.score += (len(lane.cars)/30)**2

    def step(self, action):
        for i in range(5000):
            if i %100 == 0:
                vs.draw(self)
            if i % 10 == 0:

                self.cars.append(cr.Car(self))

            vels = []
            for i in range(len(self.traffic_lights)-1):
                # print(action, 'action')
                # self.traffic_lights[i].vel = min(max(0.1,action[0,i]/1000+self.traffic_lights[i].vel),0)
                self.traffic_lights[i].vel = max(1+action[0,0],0)
                # print('Velocity:',self.traffic_lights[i].vel)
                self.traffic_lights[i].t_last += self.traffic_lights[i].vel
                # print(action[0,i])
                vels.append(self.traffic_lights[i].vel)

            reward  = 0
            for i in range(len(self.traffic_lights)-1):
                self.traffic_lights[i].update_state()

            # print(self.traffic_lights[4].phases[0])
            for tl in self.traffic_lights:
                for connection in tl.phase:
                    # print(connection, 'connection')
                    connection.check_traffic()

            for lane in self.lanes:
                reward -= len(lane.cars)**2
            # print(reward)
        print('Velocities', vels)

        return self.get_state(), [reward], False, {}

    def reset(self):

        # Reset parameters (needed to recycle vehicles during different generations)
        self.cars = []
        self.lanes = []
        self.traffic_lights = []
        self.connections = []
        self.score = 0
        self.fitness = 0

        for i in env.roads:
            line1 = ln.Line(env.roads[i], -1, 1)
            line2 = ln.Line(env.roads[i], -1, 0)
            line3 = ln.Line(env.roads[i], 1, 0)
            line4 = ln.Line(env.roads[i], 1, 1)
            self.lanes.append(line1)
            self.lanes.append(line2)
            self.lanes.append(line3)
            self.lanes.append(line4)

        for i in range(len(self.lanes)):
            # print('i:',int(i/16)*16,i)
            if int((i % 16) / 4) == 0 or int((i % 16) / 4) == 2:
                type = 'v'
            else:
                type = 'h'
            if self.lanes[i].layer == 0 and self.lanes[i].direction == 1:
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 10) % 16],
                                                     str(1 + int(i / 16)) + 'left' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 11) % 16],
                                                     str(1 + int(i / 16)) + 'left' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 6) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 7) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))

            if self.lanes[i].layer == 1 and self.lanes[i].direction == 1:
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 1) % 16],
                                                     str(1 + int(i / 16)) + 'right' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 2) % 16],
                                                     str(1 + int(i / 16)) + 'right' + type, False))

                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 5) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))
                self.connections.append(cn.Connection(self.lanes[i], self.lanes[int(i / 16) * 16 + (i + 6) % 16],
                                                     str(1 + int(i / 16)) + 'straight' + type, False))

        self.connections.append(cn.Connection(self.lanes[4], self.lanes[31], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[5], self.lanes[30], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[29], self.lanes[6], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[28], self.lanes[7], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[16], self.lanes[43], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[17], self.lanes[42], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[41], self.lanes[18], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[40], self.lanes[19], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[44], self.lanes[55], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[45], self.lanes[54], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[53], self.lanes[46], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[52], self.lanes[47], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[56], self.lanes[3], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[57], self.lanes[2], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[1], self.lanes[58], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[0], self.lanes[59], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[60], self.lanes[71], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[61], self.lanes[70], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[69], self.lanes[62], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[68], self.lanes[63], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[72], self.lanes[83], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[73], self.lanes[82], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[81], self.lanes[74], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[80], self.lanes[75], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[88], self.lanes[99], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[89], self.lanes[98], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[97], self.lanes[90], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[96], self.lanes[91], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[12], self.lanes[87], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[13], self.lanes[86], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[85], self.lanes[14], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[84], self.lanes[15], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[124], self.lanes[103], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[125], self.lanes[102], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[101], self.lanes[126], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[100], self.lanes[127], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[140], self.lanes[119], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[141], self.lanes[118], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[117], self.lanes[142], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[116], self.lanes[143], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[8], self.lanes[115], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[9], self.lanes[114], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[113], self.lanes[10], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[112], self.lanes[11], 'parallel', True))

        self.connections.append(cn.Connection(self.lanes[24], self.lanes[131], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[25], self.lanes[130], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[129], self.lanes[26], 'parallel', True))
        self.connections.append(cn.Connection(self.lanes[128], self.lanes[27], 'parallel', True))

        self.traffic_lights.append(
            tl.Traffic_Light([['1lefth', '1rightv'], ['1straightv'], ['1leftv', '1righth'], ['1straighth']], 2000, 0,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['2lefth', '2rightv'], ['2straightv'], ['2leftv', '2righth'], ['2straighth']], 2000,
                             4000,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['3lefth', '3rightv'], ['3straightv'], ['3leftv', '3righth'], ['3straighth']], 2000,
                             0000,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['4lefth', '4rightv'], ['4straightv'], ['4leftv', '4righth'], ['4straighth']], 2000,
                             4000,self))
        self.traffic_lights.append(
            tl.Traffic_Light([['5lefth', '5rightv'], ['5straightv'], ['5leftv', '5righth'], ['5straighth']], 2000,
                             4000, self))
        self.traffic_lights.append(
            tl.Traffic_Light([['6lefth', '6rightv'], ['6straightv'], ['6leftv', '6righth'], ['6straighth']], 2000,
                             4000, self))
        self.traffic_lights.append(
            tl.Traffic_Light([['7lefth', '7rightv'], ['7straightv'], ['7leftv', '7righth'], ['7straighth']], 2000,
                             4000, self))
        self.traffic_lights.append(
            tl.Traffic_Light([['8lefth', '8rightv'], ['8straightv'], ['8leftv', '8righth'], ['8straighth']], 2000,
                             4000, self))
        self.traffic_lights.append(
            tl.Traffic_Light([['9lefth', '9rightv'], ['9straightv'], ['9leftv', '9righth'], ['9straighth']], 2000,
                             4000, self))

        self.traffic_lights.append(tl.Traffic_Light([['parallel']], 200, 0,self))

        return self.get_state()