import numpy as np

"""File for storing various activation functions"""

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
        exp_values = np.exp(inputs- np.max(inputs, axis= 1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class TestAct:
    def forward(self, inputs):
        output = inputs +1
        return output