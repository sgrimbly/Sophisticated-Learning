import numpy as np
from numpy import random
import copy

from spm_utils import spm_cross, spm_backwards
from math_utils import normalise_matrix_columns, normalise_vector, round_half_up
from active_inference_utils import calculate_posterior

# Backward smoothing the posterior Q -> L
# Masking, and then upweighting parts of the a_learning matrix

def initialise_distributions(num_states, contextual_food_locations, contextual_water_locations, contextual_sleep_locations, hill_1, start_position):
    '''
    Set up the transition dynamics for the environment (B) and agent beliefs about these dynamics (b),
    as well as initialize resource locations and chosen actions.

    Parameters:
    - num_states (int): The number of states in the environment.
    - contextual_food_locations (list or array-like): Indices of food source locations in the environment.
    - contextual_water_locations (list or array-like): Indices of water source locations in the environment.
    - contextual_sleep_locations (list or array-like): Indices of sleep area locations in the environment.
    - hill_1 (int or list): Index or indices representing a hill location in the environment.
    - B (dict, optional): The transition dynamics for the environment. If not provided, it will be initialized.
    - b (dict, optional): The agent's beliefs about the transition dynamics. If not provided, they will match the true dynamics initially.

    Returns:
    - A (numpy.ndarray): A 4D array representing certain environmental dynamics.
    - a (numpy.ndarray): A 4D array representing the agent's beliefs about environmental dynamics.
    - B (dict): The updated transition dynamics for the environment.
    - b (dict): The updated agent's beliefs about the transition dynamics.
    - D (list of numpy.ndarray): A list of 2D arrays representing some initial state distributions.
    '''

    # If B and b are not provided, initialize them
    B = [np.zeros((num_states, num_states, 5)), np.zeros((4, 4, 5))]

    # Set the values directly, no need for the loop
    B[0] = np.repeat(np.eye(num_states)[:, :, np.newaxis], 5, axis=2)
    B[1] = np.repeat(np.array([
                    [0.95, 0, 0, 0.05],
                    [0.05, 0.95, 0, 0],
                    [0, 0.05, 0.95, 0],
                    [0, 0, 0.05, 0.95]])[:, :, np.newaxis], 5, axis=2)
    #b[1] = np.full((4, 4, 5), 0.25)  # Uniform distribution

    # Shifting distributions based on actions and states
    # Define the sets of states that will not be rolled for each direction
    exclude_left = set(range(0, 100, 10))
    exclude_right = set(range(9, 100, 10))
    exclude_up = set(range(90, 100))
    exclude_down = set(range(0, 10))

    for i in range(num_states):
        if i not in exclude_left:
            B[0][:, i, 1] = np.roll(B[0][:, i, 1], -1)  # move left
        if i not in exclude_right:
            B[0][:, i, 2] = np.roll(B[0][:, i, 2], 1)  # move right
        if i not in exclude_up:
            B[0][:, i, 3] = np.roll(B[0][:, i, 3], 10)  # move up
        if i not in exclude_down:
            B[0][:, i, 4] = np.roll(B[0][:, i, 4], -10)  # move down

    # Agent just knows the true dynamics here.
    b = copy.deepcopy(B)
    
    A, a, D = set_resource_locations(
        num_states,
        contextual_food_locations,
        contextual_water_locations,
        contextual_sleep_locations,
        hill_1,
        start_position)
    
    return A, a, B, b, D

def set_resource_locations(
    num_states,
    contextual_food_locations,
    contextual_water_locations,
    contextual_sleep_locations,
    hill_1,
    start_position):
    """
    Initialize and set resource locations for an agent navigating an environment.

    This function initializes multi-dimensional arrays representing the environment and the agent's beliefs about resource locations. It sets specific values based on the provided contextual locations of food, water, sleep areas, and a hill.

    Parameters:
    - num_states (int): The number of states in the environment.
    - contextual_food_locations (list or array-like): Indices of food source locations in the environment.
    - contextual_water_locations (list or array-like): Indices of water source locations in the environment.
    - contextual_sleep_locations (list or array-like): Indices of sleep area locations in the environment.
    - hill_1 (int or list): Index or indices representing a hill location in the environment.

    Returns:
    - A (numpy.ndarray): A 4D array representing certain environmental dynamics, with dimensions representing [resource type, state, state, action].
    - a (numpy.ndarray): A 4D array representing the agent's beliefs about environmental dynamics, with the same dimensions as `A`.
    - D (list of numpy.ndarray): A list of 2D arrays representing some initial state distributions.
    - short_term_memory (numpy.ndarray): A 4D array representing the agent's short-term memory, with dimensions [time, state, action, feature].
    """
    # Initialize A and a as lists of NumPy arrays with different shapes
    A = [
        np.zeros((100, 100, 4)),  # 100x100x4
        np.zeros((4, 100, 4)),    # 4x100x4
        np.zeros((5, 100, 4))     # 5x100x4
    ]

    a = [
        np.zeros((100, 100, 4)),  # 100x100x4
        np.zeros((4, 100, 4)),    # 4x100x4
        np.zeros((5, 100, 4))     # 5x100x4
    ]

    for i in range(num_states):
        A[0][i,i,:] = 1
        a[0][i,i,:] = 1

    # Directly set the values for A[1]
    A[1][0, :, :] = 1
    for i in range(4):
        A[1][1, contextual_food_locations[i], i] = 1
        A[1][0, contextual_food_locations[i], i] = 0
        A[1][2, contextual_water_locations[i], i] = 1
        A[1][0, contextual_water_locations[i], i] = 0
        A[1][3, contextual_sleep_locations[i], i] = 1
        A[1][0, contextual_sleep_locations[i], i] = 0

    # Directly set the values for A[2]
    A[2][4, :, :] = 1
    for i in range(4):
        A[2][i, hill_1, i] = 1
        A[2][4, hill_1, i] = 0

    a[2] = copy.deepcopy(A[2])
    a[1] += 0.1  # starting concentration counts

    # Create D as a list of positional and contextual distributions. Position starts at index 50 (51st element).
    D = [np.zeros((100, 1)), np.full((4, 1), 0.25)]
    D[0][start_position] = 1

    return A, a, D

def is_alive(time_since_resource, resource_constraints):
    for resource, time_since in time_since_resource.items():
        if time_since > resource_constraints.get(resource, float('inf')):
            return False
    return True

def update_environment(b, D, t, historical_chosen_action, historical_true_states, historical_agent_posterior_Q, start_position):
    """
    Transition the state given the chosen action, the posterior, and the transition matrix. Based on the current time step and factor.
    """
    
    b[1] = normalise_matrix_columns(b[1])
    
    if t > 0:
        Q_previous = historical_agent_posterior_Q[t-1]
        true_state_previous = historical_true_states[t-1]
        chosen_action_previous = historical_chosen_action[t-1]
        
        Q = [np.zeros_like(arr) for arr in Q_previous]
        true_state = np.zeros(shape=true_state_previous.shape, dtype=int)
        for factor in range(2):            
            Q[factor] = (b[factor][:,:,chosen_action_previous] @ Q_previous[factor].T).T
            state = 0
            if factor == 0:
                # (Positional) State transitions are dependent on the previously selected action 
                state = np.argmax(np.cumsum(b[factor][:, true_state_previous[factor], chosen_action_previous]) >= np.random.rand())
            else:
                # 0 here because context does not depend on the selected action
                state = np.argmax(np.cumsum(b[factor][:, true_state_previous[factor], 0]) >= np.random.rand())
            true_state[factor] = int(state)
            
        historical_agent_posterior_Q.append(Q)  
        historical_true_states.append(true_state)
    else: 
        # Initialise (t=0)
        # Optional optimising step: calculating shape by doing a transpose seems like a waste
        Q = [D[0].T, D[1].T]
        historical_agent_posterior_Q.append(Q)
        
        sampled_context = np.argmax(np.cumsum(D[1]) >= np.random.rand())
        true_state = np.array([start_position,sampled_context])
        historical_true_states.append(true_state)
        
    return historical_agent_posterior_Q, historical_true_states

def update_needs(historical_true_states, t, contextual_food_locations, contextual_water_locations, contextual_sleep_locations, time_since_resource):
    season = historical_true_states[t][1]
    location = historical_true_states[t][0]

    if (season == 0 and location == contextual_food_locations[0]) or \
       (season == 1 and location == contextual_food_locations[1]) or \
       (season == 2 and location == contextual_food_locations[2]) or \
       (season == 3 and location == contextual_food_locations[3]):
        time_since_resource["Food"] = 0
        time_since_resource["Water"] += 1
        time_since_resource["Sleep"] += 1

    elif (season == 0 and location == contextual_water_locations[0]) or \
         (season == 1 and location == contextual_water_locations[1]) or \
         (season == 2 and location == contextual_water_locations[2]) or \
         (season == 3 and location == contextual_water_locations[3]):
        time_since_resource["Water"] = 0
        time_since_resource["Food"] += 1
        time_since_resource["Sleep"] += 1

    elif (season == 0 and location == contextual_sleep_locations[0]) or \
         (season == 1 and location == contextual_sleep_locations[1]) or \
         (season == 2 and location == contextual_sleep_locations[2]) or \
         (season == 3 and location == contextual_sleep_locations[3]):
        time_since_resource["Sleep"] = 0
        time_since_resource["Food"] += 1
        time_since_resource["Water"] += 1

    else:
        if t > 0:
            time_since_resource["Food"] += 1
            time_since_resource["Water"] += 1
            time_since_resource["Sleep"] += 1

    return time_since_resource

def get_observations(A, historical_true_states, t, num_modalities, num_states, num_resource_observations, num_context_observations):
    true_state = historical_true_states[t]
    O = [
        np.zeros(num_states), 
        np.zeros(num_resource_observations), 
        np.zeros(num_context_observations)
    ]

    for modality in range(num_modalities):
        # Cumulative distribution for the current state and context
        observations_CDF = np.cumsum(A[modality][:, true_state[0], true_state[1]])  # use true_state[1] for the context
    
        # TODO: I could use the searchsorted method for CDF sampling elsewhere in code. This is more efficient as it uses binary search O(logn) vs O(n).
        observation_index = np.searchsorted(observations_CDF, np.random.rand())  # Find the index where a random number would be inserted

        # Check if observation_index is out of bounds and correct it if necessary. In the context of a CDF, this situation can happen if the random number generated by np.random.rand() is very close to 1, leading np.searchsorted to return an index that's equal to the length of the array. The check ensures that this off-by-one error is corrected, preventing attempts to access an index that doesn't exist and thus avoiding the IndexError.
        if observation_index == O[modality].size:
            observation_index = O[modality].size - 1

        O[modality][observation_index] = 1  # Correcting the way to access elements in list 'O' and set value

    return O

def get_predicted_posterior(a, Q, O, historical_predictive_observations_posteriors):
    """
    Calculate the predicted posterior distributions based on current beliefs and observations.

    This function normalizes the agent's likelihood model, computes the cross product of the
    posterior distributions, and then calculates the predictive observation posterior for each factor.
    It updates the historical predictive observation posteriors and calculates the predicted posterior
    distributions for the current timestep.

    Parameters:
    - a: Agent's likelihood model / concentration parameters.
    - Q: List of  posterior distributions for each factor.
    - O: List of observations.
    - historical_predictive_observations_posteriors: List of historical predictive observation posteriors.

    Returns:
    - rounded_predicted_P: Rounded predicted posterior distributions.
    - historical_predictive_observations_posteriors: Updated list of historical predictive observation posteriors.
    """

    # Deep copy the agent's likelihood model to avoid modifying the original
    y = copy.deepcopy(a)

    # Normalize the columns of the resource likelihood model for a specific modality
    y[1] = normalise_matrix_columns(y[1])  # Shape: (4, 100, 4)

    # Compute the cross product of the posterior distributions and reshape for matrix multiplication
    qs = spm_cross(copy.deepcopy(Q)).ravel(order='F')[:, np.newaxis]

    # Initialize the predictive observations posterior with the current observation
    predictive_observations_posterior = [copy.deepcopy(O[0])]

    # Calculate and normalize the predictive observation posterior for each modality
    predictive_observations_posterior.append(
        normalise_vector(y[1].reshape(y[1].shape[0], -1, order='F') @ qs)
    )
    predictive_observations_posterior.append(
        normalise_vector(y[2].reshape(y[2].shape[0], -1, order='F') @ qs)
    )

    # Update the historical record of predictive observation posteriors
    historical_predictive_observations_posteriors.append(predictive_observations_posterior)

    # Calculate the predicted posterior distributions for the current timestep
    predicted_posterior = calculate_posterior(copy.deepcopy(Q), y, O=predictive_observations_posterior)

    # Round the predicted posterior for comparison and storage
    rounded_predicted_P = round_half_up(predicted_posterior[1], 1)
    
    return rounded_predicted_P, historical_predictive_observations_posteriors

def smooth_beliefs(O, Q, A, a, B, smoothing_start, smoothing_t, t):
    """
    Smooth beliefs based on observations, posterior distributions, and the model parameters.
    
    This function performs backward smoothing of the posterior distributions based on the current 
    observations and model parameters. It then checks for changes in the belief about the context. 
    If there is a significant change in belief, or if it is the last time step, the beliefs about the 
    resources are updated to reflect this change.

    Parameters:
    - O: List of observations.
    - Q: List of posterior distributions for each factor.
    - A: Observation likelihood model.
    - a: Agent's likelihood model / concentration parameters.
    - B: Transition model.
    - start: Starting time step for backward smoothing.
    - time_y: Current time step within backward smoothing.
    - t: Current time step.

    Returns:
    - a: Updated agent's likelihood model / concentration parameters.
    """
    
    # 1. Smooth the posterior (from spm_backwards logic)
    # Apply backward smoothing to evaluate the posterior over initial states based on current observations
    # TODO: Check comment in spm_backward function regarding whether or not 'A' or 'a' should be used. 
    smoothed_posterior = spm_backwards(O, Q, A, B, smoothing_t, t)
    
    # Prepare the likelihood and smoothed posterior for cross multiplication
    # Q[t][0]: Shape (1,100)
    # smoothed_posterior: Shape (4,1)
    smoothed_posterior = [Q[smoothing_t][0], smoothed_posterior]
    
    # 2. Check for changes in posterior
    # Compare the rounded posterior beliefs to determine if there's a significant change in context belief
    no_context_belief_change = np.array_equal(
        np.round(smoothed_posterior[1], 3), 
        np.round(Q[smoothing_t][1], 3)
    )
    
    # If there's a change in belief or it's the last time step within backward smoothing, update the beliefs about resources
    if (not no_context_belief_change and smoothing_t > smoothing_start) or smoothing_t == t:
        a = update_agent_likelihood(a, O, smoothed_posterior, smoothing_t)
    
    return a

def update_agent_likelihood(a, O, smoothed_posterior, smoothing_t):
    """
    Update the resource beliefs based on observations, posterior distributions, and smoothed posteriors.

    This function updates the agent's likelihood of resource locations based on the observations 
    and the smoothed posterior of the context. It involves calculating a learning update and applying 
    it to the current resource model, a.

    Parameters:
    - a: Agent's likelihood model / concentration parameters.
    - O: List of observations.
    - Q: List of current posterior distributions for each factor.
    - smoothed_posterior: Smoothed posterior distribution for the context modality.
    - smoothing_t: The timestep at which smoothing is being applied.

    Returns:
    - a: Updated agent's likelihood model / concentration parameters.

    """

    modality = 1  # Focusing on the resource location modality

    # Extract the observation for the current smoothing timestep and modality
    a_learning = np.transpose([O[smoothing_t][modality]])

    # Cross multiply the learning update with each factor's distribution
    # In MATLAB there is a redundant for loop over modalities. This might be required if we have multiple modalities that change with respect to a change in the context.
    for factor in range(2):
        a_learning = spm_cross(a_learning, smoothed_posterior[factor])

    # Apply a mask to ensure that a_learning is modified only where a[modality] is positive. Might not be necessary
    a_learning = a_learning * (a[modality] > 0)

    # Define the proportion to subtract from zero elements
    # TODO: Does this correspond to the 0.7 factor below?
    proportion = 0.3

    # TODO: What is the logic of this looping operation? It seems to penalise the first entry of each column (i.e. the first row) where the value is 0. Why is this? Also, the penalisation is some proportion of the maximum. Is this an arbitrary choice?
    # Penalize the first entry of each column where the value is zero
    for i in range(a_learning.shape[2]):
        for j in range(a_learning.shape[1]):
            max_value = max(a_learning[1:, j, i])
            amount_to_subtract = proportion * max_value
            if a_learning[0, j, i] == 0:
                a_learning[:, j, i] -= amount_to_subtract

    # Update the resource likelihood and apply a threshold
    a[modality] = a[modality] + 0.7 * a_learning
    a[modality][a[modality] <= 0.05] = 0.05
    
    return a

def get_actual_posterior(a, Q, O):
    """
    Calculate the actual posterior distributions based on the current beliefs, 
    observations, and the likelihood/preference model.

    This function normalizes the columns of the agent's likelihood/preference model 
    and then computes the actual posterior distributions using the current observations.
    The resulting posterior distributions are rounded for comparison.

    Parameters:
    - a: Agent's likelihood model / concentration parameters.
    - Q: List of posterior distributions for each factor.
    - O: List of observations.

    Returns:
    - rounded_actual_P: Rounded actual posterior distributions for the context modality.
    - P: Actual posterior distributions for both factors.
    - y: Normalized agent's likelihood/preference model.
    """

    # Deep copy the resource likelihood model to avoid modifying the original
    y = copy.deepcopy(a)

    # Normalize the columns of the resource likelihood model for a specific modality
    y[1] = normalise_matrix_columns(y[1])  # Shape: (4, 100, 4)

    # Calculate the actual posterior distributions based on the current observations
    P = calculate_posterior(copy.deepcopy(Q), y, O)

    # Round the actual posterior for the context modality for comparison
    rounded_actual_P = round_half_up(P[1], 1)
    
    return rounded_actual_P, P, y

def delete_short_term_memory():
    short_term_memory = np.zeros((35, 35, 35, 400))
    return short_term_memory

def random_position(previous_positions):
    '''Return previous_positions array with last element being a new, novel random position that has not already existed in previous_positions. Positions are limited to integer from 1 to 100, inclusive.'''
    pos = random.randint(1,101) 
    while pos in previous_positions:
        pos = random.randint(1,101)
    previous_positions.append(pos)
    return previous_positions