import numpy as np
import decimal
from scipy.stats import entropy

# TODO (St John): Test matrix normalisation functions. #testing

# TODO (St John): Add documentation to all functions in this file. #documentation

def normalise_matrix_columns(m):
    """
    Normalize the columns of a matrix so that each sums to 1.
    
    This function divides each element of a matrix by the sum of its column,
    effectively normalizing each column. If a column sum is zero, the column 
    remains unchanged to prevent division by zero.

    Parameters:
    m (numpy.ndarray): A 2D NumPy array (matrix).

    Returns:
    numpy.ndarray: A 2D NumPy array with normalized columns.
    """
    column_sums = m.sum(axis=0, keepdims=True)
    # Using np.divide to avoid division by zero and handle with 'where'
    normalized_m = np.divide(m, column_sums, where=column_sums!=0)

    return normalized_m

def normalise_vector(vector):
    """
    Normalize a 1D numeric vector to make its elements sum to 1.

    This function converts a vector into a probability distribution. If the sum of the vector 
    is zero, it returns a uniform distribution. Otherwise, each element is divided by the 
    total sum of the vector.

    Parameters:
    vector (np.ndarray): A 1D NumPy array containing numerical values.

    Returns:
    np.ndarray: A 1D NumPy array representing a normalized probability distribution.
    """

    if vector.sum() == 0:
        normalised_v = np.ones_like(vector) / np.size(vector)
    else:
        normalised_v = vector / vector.sum()

    return normalised_v

def kl_divergence(A, B):
    """
    Compute the Kullback-Leibler (KL) divergence between two matrices.

    Flattens and normalizes matrices A and B into vectors, and then calculates 
    the KL divergence between these vectors.

    Parameters:
    A, B (numpy.ndarray): Input matrices to compare.

    Returns:
    float: KL divergence between the normalized vectors of A and B.
    """
    # TODO (St John): Ensure that deep copying is not required if A and B are passed by reference. #testing
    A_norm = normalise_vector(A.ravel(order='F'))
    B_norm = normalise_vector(B.ravel(order='F'))
    e = entropy(A_norm, B_norm)
    return e

def maximum_entropy(A):
    """
    Calculate the maximum entropy of a given state distribution matrix.

    The function computes the entropy across all states in the matrix A. 
    Each column in A represents a different state, and the rows represent 
    different instances of state distributions. The entropy is calculated 
    using the formula for Shannon entropy, considering the information content 
    of each state.

    Parameters:
    A (numpy.ndarray): A 2D array where each column represents a state and 
                       each row represents a state distribution.

    Returns:
    float: The calculated maximum entropy value for the state distribution.
    """
    entropy = 0
    states = A.shape[1]
    # This function is unused currently,
    # but the line below might be problematic given row- versus column-major order
    L = A.reshape(A.shape[0], -1) 
    for state in range(states):
        state_dist = L[:, state]
        information = np.log2(state_dist + np.exp(-16))
        entropy += -np.dot(state_dist, information)
    return entropy

def softmax(x):
    """
    Compute the softmax of a vector or matrix.

    The softmax function converts each element of the input to a probability,
    with larger values having higher probabilities. The function is often used
    in machine learning, especially for classification tasks.

    Parameters:
    x (numpy.ndarray): A 1D or 2D array of numerical values.

    Returns:
    numpy.ndarray: The softmax-transformed array, with the same shape as the input.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def nat_log(x):
    """
    Compute the natural logarithm of an array, adding a small value to prevent log(0).

    This function calculates the natural logarithm (log) while avoiding the issue of log(0)
    by adding a very small value to the input array.

    Parameters:
    x (numpy.ndarray): A NumPy array of numerical values.

    Returns:
    numpy.ndarray: An array where the natural logarithm has been applied to each element.
    """
    return np.log(x + np.exp(-500))


def round_half_up(n, decimals=0, epsilon=1e-9):
    """
    Round a number, NumPy array, list of numbers, or list of NumPy arrays half up to a specified number of decimal places.
    If the rounded value is a whole number, it is returned as an integer.
    
    This function rounds values using the "round half up" rule, with an adjustment for floating-point representation errors.
    It handles single numbers, NumPy arrays, lists of numbers, and lists of NumPy arrays.

    Parameters:
    n (int, float, numpy.ndarray, list): The number, NumPy array, or list of numbers/NumPy arrays to round.
    decimals (int, optional): The number of decimal places to round to. Default is 0.
    epsilon (float, optional): A small value to adjust for floating-point errors. Default is 1e-9.

    Returns:
    int, float, numpy.ndarray, list: The rounded number(s) with the same type as the input.
    """
    def round_single_value(value):
        # Adjust the value slightly to account for floating-point representation error
        adjusted_value = decimal.Decimal(value) + decimal.Decimal(epsilon)
        # Round the value
        rounded_value = adjusted_value.quantize(decimal.Decimal('1.' + '0'*decimals), rounding=decimal.ROUND_HALF_UP)
        # Convert to integer if the result is a whole number
        return int(rounded_value) if rounded_value == rounded_value.to_integral_value() else float(rounded_value)

    # Check the type of the input and apply the rounding accordingly
    if isinstance(n, np.ndarray):
        vectorized_round = np.vectorize(round_single_value)
        return vectorized_round(n)
    elif isinstance(n, (int, float)):
        return round_single_value(n)
    elif isinstance(n, list):
        if all(isinstance(item, (int, float)) for item in n):
            return [round_single_value(item) for item in n]
        elif all(isinstance(item, np.ndarray) for item in n):
            return [round_half_up(arr, decimals, epsilon) for arr in n]
        else:
            raise ValueError("All items in the list must be numbers or NumPy arrays.")
    else:
        raise TypeError("Input must be an int, float, numpy.ndarray, or list of numpy.ndarrays or numbers.")