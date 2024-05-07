# test_update_environment.py

import unittest
import numpy as np
import sys
import copy
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sophisticated_agent_dir = os.path.join(parent_dir, 'algorithms', 'sophisticated_agent')
sys.path.append(parent_dir)
sys.path.append(sophisticated_agent_dir)

from agent_utils import initialise_distributions, update_environment
from sophisticated_agent import initialise_experiment

class TestUpdateEnvironment(unittest.TestCase):
    
    def test_actions_first_MATLAB_trial(self):
        
        # Initialise experimental/environmental hyperparams
        num_states = 100
        contextual_food_locations = [70,42,56,77]
        contextual_water_locations = [72,32,47,66]
        contextual_sleep_locations = [63,43,48,58] 
        hill_1 = 54
        start_position = 50
        
        # Initialise distributions
        A, a, B, b, D = initialise_distributions(
            num_states,
            contextual_food_locations,
            contextual_water_locations,
            contextual_sleep_locations,
            hill_1,
            start_position)
        
        # Initialise experiment
        chosen_action, short_term_memory, historical_predictive_observations_posteriors, historical_agent_observations, historical_agent_posterior_Q, historical_true_states, historical_chosen_action, time_since_resource = initialise_experiment()
        
        # Define actions for trial, taken from first MATLAB trial
        actions = np.array([3, 3, 0, 3, 3, 2, 4, 2, 3, 2, 2, 4, 4, 4, 4, 2, 4, 1, 4, 1, 1])

        # Number of time steps
        num_t = range(len(actions))
        
        # Values calculated from MATLAB code
        expected_positions = [50, 60, 70, 70, 80, 90, 91, 81, 82, 92, 93, 94, 84, 74, 64, 54, 55, 45, 44, 34, 33]
        expected_length_outputs = len(actions)
                
        # Define posterior Q for seven subsequent time steps        
        expected_historical_Q = []

        # Based solely on the update_environment() function, we don't expect the context posterior to change
        Q_one = [np.zeros((1,100)), np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q_one[0][0,expected_positions[0]] = 1
        expected_historical_Q.append(Q_one)

        # Iterate over the positions list and create subsequent states
        for i in range(1, len(expected_positions)):
            Q_new = copy.deepcopy(expected_historical_Q[i-1])
            Q_new[0][0,expected_positions[i-1]] = 0  # Reset previous index to 0
            Q_new[0][0,expected_positions[i]] = 1    # Set current index to 1
            expected_historical_Q.append(Q_new)

        # Call the update_environment function
        for t in num_t:
            historical_agent_posterior_Q, historical_true_states = update_environment(b, D, t, actions, historical_true_states, historical_agent_posterior_Q, start_position)

        calculated_length_Q = len(historical_agent_posterior_Q)
        calculated_length_true_states = len(historical_true_states)
        calculated_positions_true_states = [array[0] for array in historical_true_states]
        calculated_positions_Q = [np.where(array[0] > 0)[1][0] for array in historical_agent_posterior_Q]
        
        # Compare length of Q to length of actions provided
        np.testing.assert_equal(calculated_length_Q, expected_length_outputs)
        # Compare length of tue states to length of actions provided
        np.testing.assert_equal(calculated_length_true_states, expected_length_outputs)

        # Compare expected positions to positions calculated from true states
        np.testing.assert_array_equal(calculated_positions_true_states, expected_positions)
        # Compare positions calculated from Q to positions calculated from true states
        np.testing.assert_array_equal(calculated_positions_Q, calculated_positions_true_states)
        
        # Compare the expected and calculated historical_Q arrays        
        for arr1, arr2 in zip(historical_agent_posterior_Q, expected_historical_Q):
            for subarr1, subarr2 in zip(arr1, arr2):
                np.testing.assert_array_equal(subarr1, subarr2, "Arrays are not equal.")

# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()