"""File for storing various activation functions.
They don't need to be classes but I started with that so I kept it like that"""

import math
import numpy as np

class Relu:
    """Creates a Relu activation object that can be passed into nodes"""
    def __init__(self):
        self.output = None
    def forward(self, inputs):
        """calculates output"""
        self.output = np.maximum(0, inputs)
        return self.output

class Softmax:
    """Creates softmax activation function"""
    def __init__(self):
        self.output = None
    def forward(self, inputs):
        """Calculates output"""
        exp_values = np.exp(inputs- np.max(inputs))
        probabilities = exp_values/np.sum(exp_values)
        self.output = probabilities
        return self.output

class tanh:
    def __init__(self):
        self.output = None
    
    def forward(self, inputs):
        return math.tanh(inputs)

class Identity:
    def __init__(self):
        self.output = None
    def forward(self,inputs):
        self.output = inputs
        return self.output

class Sigmoid:
    pass