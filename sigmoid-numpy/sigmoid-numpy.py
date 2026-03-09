import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    val = 1/(1 + 1/np.exp(x))
    return val
    