import numpy as np
def u(t): return np.heaviside(t,0)

np.seterr("raise")

def sigmoid(x):
    try:
        v = 1/(1+np.exp(-x))
    except FloatingPointError:
        raise RuntimeError
    return v
def dsigmoid(x): 
    return sigmoid(-x)*sigmoid(x)

def relu(x): 
    return x * u(x)
def drelu(x): 
    return u(x)