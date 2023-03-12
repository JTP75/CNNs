import numpy as np

def MSE(ya,yp):
    ya = ya.reshape((ya.shape[0],1)) if len(ya.shape) == 1 else ya
    if ya.shape != yp.shape: yp = yp.T
    assert ya.shape == yp.shape, f"Shape mismatch: ya: ({ya.shape[0]},{ya.shape[1]}), yp: ({yp.shape[0]},{yp.shape[1]})"
    return np.sum((ya-yp)**2)/ya.size