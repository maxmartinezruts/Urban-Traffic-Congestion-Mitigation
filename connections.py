

class Connection:
    def __init__(self,start, end, type, linked):
        self.type = type
        self.start = start
        self.end = end
        self.linked = linked
        self.start.add_exit(self)
        self.end.add_entry(self)
        self.timer = 0
        count = 0
        current = 50
        rest = 9
        self.flow = []
        self.separation = 50
        for i in range(200):
            rest = max(rest-1,0)
            current-=rest
            count+=current
            self.flow.append(count)
        # print(self.flow)
        self.state = linked

    def change_state(self, state):
        self.state = state
        self.timer = -200
        self.separation = 150

    def check_traffic(self):

        if self.linked and len(self.start.cars)>0:
            self.car_flow()
        elif self.state:
            self.timer += 1
            if ((self.timer >= self.separation) and len(self.start.cars)>0):
                self.timer = 0
                self.separation = max(50,self.separation-3)
                self.car_flow()

    def car_flow(self):
        car = self.start.cars[0]
        next_line = car.next_connection.end
        if len(next_line.cars) < 30:
            self.start.remove_car(car)
            next_line.add_car(car)