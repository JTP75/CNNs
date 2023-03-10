import numpy as np

def MSE(ya,yp):
    ya = ya.reshape((ya.shape[0],1)) if len(ya.shape) == 1 else ya
    if ya.shape != yp.shape:
        raise Exception("Array shape mismatch")
    return np.sum((ya-yp)**2)/ya.size