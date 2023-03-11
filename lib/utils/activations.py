import numpy as np
def u(t): return np.heaviside(t,0)


def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return sigmoid(-x)*sigmoid(x)

def relu(x): return x * u(x)
def drelu(x): return u(x)