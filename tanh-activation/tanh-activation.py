import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    val = (np.exp(x) - (1/np.exp(x)))/(np.exp(x) + (1/np.exp(x)))
    return val