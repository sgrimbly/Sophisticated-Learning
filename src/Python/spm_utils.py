import numpy as np 

def spm_backwards(O, Q, A, B, t, T):    
    # Backwards smoothing to evaluate posterior over initial states (from t-6 to t)
    
    # Create a copy of the context component of the posterior (2nd dimension/factor) 
    L = np.array(Q[t][1], copy=True)  
    p = np.eye(N=B[1][:,:,0].shape[0]) # = np.ones(shape=(4,4))  # Initialize p to 1
    
    # Iterate through the timesteps from t+1 to T
    for timestep in range(t+1, T+1):
        
        # Take the action dimension of the context transition matrix
        # Multiply p by itself to track the probability over time of getting to the current state
        p = B[1][:, :, 0] @ p 
        
        # Loop over index tuples in L. e.g. (0,0) for the first element in L
        for state in range(len(L[0])):
            # Calculate which context observation was made at the current timestep by sampling from the observation distribution. This is done by calculating the CDF of the observation distribution (O) and sampling from a uniform distribution (using np.random). The index of the first element in the CDF that is greater than the sampled value is the context observation that was made.
            cumulative_sum = np.cumsum(O[timestep][2])
            obs = np.searchsorted(cumulative_sum, np.random.rand())
            obs = min(obs, len(cumulative_sum) - 1)
            
            # Temp computations using A, Q, and obs
            # TODO: Is using capital A in this spm_backward smoothing a mistake? Isn't capital A the likelihood of the agent's observations given the true state? Shouldn't this be a lower case a?
            temp = A[2][obs, :, :]
            temp = np.transpose(temp, (1, 0))
            temp = temp @ Q[timestep][0].T
            aaa = temp.T @ p[:, state]
            
            # Update L
            L[0][state] = L[0][state] * aaa

    # Normalize the columns of L
    sum_L = np.sum(L)
    if sum_L != 0:
        L = L / sum_L
        # Handle NaN values (replace NaNs with 1/number of columns)
        L[np.isnan(L)] = 1 / L.shape[1]
    else:
        L[:] = 1 / L.shape[1]
    
    return L

# NOTE: This implementation is based on version from PyMDP
def spm_cross(x, y=None, *args):
    """ Multi-dimensional outer product
    
    Parameters
    ----------
    - `x` [np.ndarray] || [Categorical] (optional)
        The values to perform the outer-product with. If empty, then the outer-product 
        is taken between x and itself. If y is not empty, then outer product is taken 
        between x and the various dimensions of y.
    - `args` [np.ndarray] || [Categorical] (optional)
        Remaining arrays to perform outer-product with. These extra arrays are recursively 
        multiplied with the 'initial' outer product (that between X and x).
    
    Returns
    -------
    - `z` [np.ndarray] || [Categorical]
          The result of the outer-product
    """
    
    # Helper function to check if the input is a numpy array with dtype "object"
    def is_object_array(array):
        return isinstance(array, np.ndarray) and array.dtype == "object"

    # Check if X is a list and handle it recursively like a MATLAB cell array
    if isinstance(x, list):
        # If X is a list with more than one element, unpack the list and call spm_cross
        if len(x) > 1:
            return spm_cross(*x)
        # If X is a list with one element, just return that element
        elif len(x) == 1:
            return x[0]
        # If X is an empty list, return an empty numpy array
        else:
            return np.array([])

    if y is None and len(args) == 0:
        if is_object_array(x):
            z = spm_cross(*list(x))
        elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
            z = x
        else:
            raise ValueError(f"Invalid input to spm_cross ({x})")
        return z

    if is_object_array(x):
        x = spm_cross(*list(x))

    if y is not None and is_object_array(y):
        y = spm_cross(*list(y))

    # Ensure x and y are numpy arrays before reshaping
    if not isinstance(x, np.ndarray) or (y is not None and not isinstance(y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    reshape_dims = tuple(list(x.shape) + list(np.ones(y.ndim, dtype=int)))
    A = x.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(x.ndim, dtype=int)) + list(y.shape))
    B = y.reshape(reshape_dims)
    z = np.squeeze(A * B)

    for next_arg in args:
        z = spm_cross(z, next_arg)

    return z