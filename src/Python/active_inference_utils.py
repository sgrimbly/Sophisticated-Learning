import numpy as np
import copy
from spm_utils import spm_cross
from math_utils import nat_log, kl_divergence, normalise_vector

def calculate_posterior(P, A, O):
    """
    Calculate the posterior over the current state based on the given prior, agent likelihood, and observation.
    
    The intuition of this function is that we take the current posterior and use it as a prior over the current state. We then use the agent likelihood to calculate the probability of the observation given the current state. Using both the likelihood and the prior, we then compute 
    our (Bayesian) posterior over the current state. This essentially means determining the probability of being in each state given the current observation and our prior beliefs about that state.
    
    Note:
    Rowan's implementation of 'calculate_posterior()' in Matlab is much more intricate for this simple agent. The complexity might be needed when the number of modalities and factors increase. For now, this Python version is kept relatively simple.
    
    Parameters:
    - P (array-like): The posterior used as a 'prior' for the next prediction.
    - A (array-like): The agent likelihood.
    - O (array-like): The actual observation received.
    
    Returns:
    array-like: The calculated posterior.
    
    Examples:
    >>> calculate_posterior([...], [...], [...])
    [...]
    
    """
    # The second modality -> the resource modality
    resource_observation = np.argmax(np.cumsum(O[1]) >= np.random.rand())
    resource_likelihood = A[1][resource_observation, :, :]
    
    # The third modality -> the context modality
    context_observation = np.argmax(np.cumsum(O[2]) >= np.random.rand())
    context_likelihood = A[2][context_observation, :, :]
    
    # Shape: (100,4)
    joint_likelihood = resource_likelihood * context_likelihood
    
    # This is the probability over the true resource in this position given that we have seen this observation and given our prior beliefs about the true state of resource locations. This is NOT context dependent, rather it is a general probability about being in this state with this resource (or no resource) given what we know about resources.
    # -> shape of resource_posterior: (4,1) since (no resource, sleep, food, water) (or whatever order...)
    resource_posterior = P[0] @ joint_likelihood
    
    # We can calculate a positional_posterior for EACH context. This is not matrix multiplication, 
    # rather it is element-wise multiplication.
    y = resource_posterior * P[1] # Shape: (1,4)
    posterior = normalise_vector(y) # Shape: (1,4)
    P[1] = posterior # Shape: (1,4)
    return P

def calculate_novelty(P, a, imagined_O, num_factors, weights):
    """
    Calculate the novelty based on the posterior, resource likelihood/preference, 
    and imagined observations using Kullback-Leibler (KL) divergence.
    
    Parameters:
    - P: List of posterior distributions for each factor.
    - a: List containing the resource likelihood/preference matrices.
    - imagined_O: List of imagined observations.
    - num_factors: The number of factors in the model.
    - weights: Dictionary of weight hyperparameters.
    
    Returns:
    - novelty: The calculated novelty based on KL divergence.
    """
    
    # NOTE (St John): In sophisticated learning there is a whole step to check when to do backward smoothing in the tree search, this isn't required for sophisticated inference.
    
    # Deep copy the posteriors and resource likelihood/preference to avoid modifying the originals
    L = copy.deepcopy(P)  # Posterior 
    a_prior = copy.deepcopy(a[1])  # Resource likelihood/preference
    a_learning = copy.deepcopy(imagined_O[1]).T  # Shape (1,4)
    
    # Perform a cross product of the learning array with each posterior distribution
    # The resulting array will be used for calculating the update in learning
    for factor in range(num_factors):
        # Shape (4,100) after first cross then (4,100,4) after second cross
        a_learning = spm_cross(a_learning, L[factor])

    # Element-wise multiplication to zero-out elements where a[1] is zero
    a_learning = a_learning * (a[1] > 0)  # Shape (4,100,4)

    # Prepare the weighted update for learning
    # This operation scales the learning update by the learning weight, but keeps the first row unchanged
    a_learning_weighted = np.array(a_learning)
    a_learning_weighted[1:, :] = weights["Learning"] * a_learning[1:, :]
    
    # Combine the prior with the weighted learning update
    a_temp = a_prior + a_learning_weighted

    # Calculate the novelty as the KL divergence between the updated likelihood and the prior
    novelty = kl_divergence(a_temp, a_prior)
    
    return novelty

def G_epistemic_value(A,s):
    '''Auxiliary function for Bayesian surprise or mutual information.
    A - likelihood array (probability of outcomes given causes)
    s - probability density of causes (prior)
    '''

    # qx holds the result of the outer product of the probability distribution over hidden states with itself.
    # This represents all possible combinations of states, indicating how likely each combination is.
    qx = spm_cross(s)  

    # G will accumulate the expected entropy or Bayesian surprise over all outcomes and state combinations.
    # qo will accumulate the probability distribution over all outcomes, weighted by the probability of the 
    # state combinations that could produce them.
    G = 0
    qo = 0

    # Identify the indices of state combinations in qx that have non-negligible probability.
    non_zero_indices = np.nonzero(qx.ravel(order='F') > np.exp(-16))[0]  
    
    for i in non_zero_indices:
        # Initialize po, which will hold the probability distribution over outcomes for the current state combination.
        po = np.array([1])

        # Calculate the probability distribution over outcomes for the current state combination by taking the 
        # outer product of the likelihoods of outcomes given this state combination across all dimensions of the model.
        for g in range(len(A)):
            a_temp = copy.deepcopy(A[g])
            a_temp = a_temp.reshape(a_temp.shape[0],-1, order='F')[:,i]
            po = spm_cross(po, a_temp)
            
        # Flatten po to make it a 1D array, for ease of the subsequent computations.
        po = po.ravel(order='F')

        # Update qo to include the probabilities of outcomes given the current state combination, weighted by 
        # the probability of the state combination itself.
        qo = qo + qx.ravel(order='F')[i] * po
        
        # Update G to include the expected entropy contributed by the current state combination and its outcomes.
        # This is weighted by the probability of the state combination and the natural log of the probabilities 
        # of the outcomes, indicating the "surprise" or information gain the outcomes represent.
        G = G + qx.ravel(order='F')[i][..., np.newaxis] @ np.dot(po.T, nat_log(po))[np.newaxis, ...]
        
    # Subtract the entropy of the overall expected outcomes from the accumulated expected entropy. This gives the 
    # Bayesian surprise or mutual information represented by the current distribution over states and outcomes, 
    # indicating how much uncertainty reduction or learning the model expects from them.
    G = G - np.dot(qo.T, nat_log(qo))

    # Return the final calculation of Bayesian surprise or mutual information.
    return G

def determine_observation_preference(time_since_resource, resource_constraints, weights):
    """
    Determine the weighted preference for each resource based on the time since each resource
    was last consumed, the constraints on resource consumption, and weight hyperparameters.
    
    This function calculates a weighted preference score for each resource. If the time since a resource
    was last consumed exceeds a specified constraint, all other resources (including an 'empty' category) 
    are set to a high negative value (representing a very low preference), and the exceeded resource 
    maintains its original value. The preference scores are then scaled by the weight hyperparameter.
    
    Parameters:
    - time_since_resource: Dictionary specifying the timesteps since each resource was last consumed.
    - resource_constraints: Dictionary specifying the maximum time the agent can go without consuming each resource.
    - weights: Dictionary specifying the weight hyperparameters.
    
    Returns:
    - np.array: An array containing the scaled preferences in the order: empty, resource1, resource2, etc.
    """
    
    # Initialize the preference array with zeros; the first element represents 'empty'
    C = np.zeros(len(resource_constraints) + 1)
    C[0] = -1  # Set default preference for 'empty' category

    constraint_met = False
    for constraint_resource, constraint in resource_constraints.items():
        # Check if the constraint for any resource is met
        if time_since_resource[constraint_resource] >= constraint:
            constraint_met = True
            # If constraint is met, set all other resources to -500
            time_since_resource = {k: (-500 if k != constraint_resource else time_since_resource[k])
                                   for k in time_since_resource}
            break

    # If any constraint is met, set 'empty' preference to -500
    if constraint_met:
        C[0] = -500

    # Populate the preference array using updated time_since_resource values
    for index, resource in enumerate(resource_constraints, start=1):
        C[index] = time_since_resource[resource]

    # Scale the preference scores by the preference weight
    return C / weights['Preference']