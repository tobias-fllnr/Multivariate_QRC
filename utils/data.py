import numpy as np

def generate_random_sequence(length: int, dimension: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random sequence of integers from 0 to 9.

    Parameters:
    length (int): Length of the random sequence.
    dimension (int): Dimension of the random sequence.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    np.ndarray: A numpy array containing the random sequence.
    """

    np.random.seed(seed)
    return np.random.uniform(0, 1, (length, dimension))