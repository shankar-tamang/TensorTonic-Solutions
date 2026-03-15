import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here

    y = np.dot(x, W1) + b1

    z = np.maximum(0, y)

    z1 = np.dot(z, W2) + b2

    return z1