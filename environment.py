"""Not used just was testing with this"""


import random
import math



class Environment:
    def __init__(self, ):
        pass



class Simple_Navigation_Env:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.actions = [0,1,2,3,4]#N,E,S,W, do nothing
        self.goal = [random.randint(0,self.width),random.randint(0,self.height)]
        self.position = [0,0]
        self.observations = self.goal + self.position
        self.farthest_distance = math.sqrt(((self.width/2)**2) + ((self.height/2)**2))
        self.current_fitness = None

    def reset(self):
        if self.current_fitness == 1:#resets goal location if someone finds the goal
            self.goal = [random.randint(0,self.width),random.randint(0,self.height)]
            self.farthest_distance = math.sqrt(((self.width/2)**2) + ((self.height/2)**2))
        else:
            self.position = [0,0]

    def _distance_to(self, pos_x2, pos_y2):
        """generates the distance while accounting for the screen wrapping around"""
        dx = abs(self.position[0] - pos_x2)
        dy = abs(self.position[1] - pos_y2)
        if dx>self.height/2:
            dx = self.height - dx
        if dy>self.width/2:
            dy = self.width - dy
        return math.sqrt((dx**2)+dy**2)

    def gen_fitness(self, must_touch_goal = False):
        """Generates fitness 1 being perfect"""
        if must_touch_goal is True: #experimental on only being rewarded for finding the location
            if self._distance_to(self.goal[0],self.goal[1]) <= 3:# within 3 points
                fitness = 1
            else:
                fitness=0.01
        else:
            fitness = (self.farthest_distance-self._distance_to(self.goal[0],self.goal[1]))/self.farthest_distance
            if fitness <= 0:
                fitness = .001
            elif self.position == self.goal:
                fitness = 1
        self.current_fitness = fitness
        return fitness

    def move(self, step, prediction):
        """Calculates a move based on a softmax output"""
        i = prediction
        if i == 0:
            self.position[1] += step#North
            if self.position[1] > self.height:
                self.position[1] = self.position[1] - self.height -1#set for screen wrapping
        elif i == 1:
            self.position[0] += step#East
            if self.position[0] > self.width:
                self.position[0] = self.position[0] - self.width -1
        elif i == 2:
            self.position[1] -= step#South
            if self.position[1] < 0:
                self.position[1] = self.position[1] + self.height+1
        elif i == 3:
            self.position[0] -= step#West
            if self.position[0] < 0:
                self.position[0] = self.position[0] + self.width+1
        elif i == 4:
            pass#dont move
