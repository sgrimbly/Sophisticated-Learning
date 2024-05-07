# test_get_observations.py

import unittest
import numpy as np
import copy
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agent_utils import initialise_distributions, get_observations

class TestGetObservations(unittest.TestCase):
    
    def test_observe_hill(self):
        
        # Initialise experimental/environmental hyperparams
        num_states = 100
        contextual_food_locations = [70,42,56,77]
        contextual_water_locations = [72,32,47,66]
        contextual_sleep_locations = [63,43,48,58] 
        hill_1 = 54
        start_position = 50
        
        # Agentic hyperparams
        num_modalities = 3
        num_resource_observations = 4 # [none, food, water, sleep]
        num_context_observations = 5 # [summer, autumn, winter, spring, none]
        
        # Initialise distributions
        A, a, B, b, D = initialise_distributions(
            num_states,
            contextual_food_locations,
            contextual_water_locations,
            contextual_sleep_locations,
            hill_1,
            start_position)

        # From first trial in MATLAB with seed "1", specify equal-size vectors of actions, positions, and contexts at each time step
        actions = np.array([3, 3, 0, 3, 3, 2, 4, 2, 3, 2, 2, 4, 4, 4, 4, 2, 4, 1, 4, 1, 1])
        positions = [50, 60, 70, 70, 80, 90, 91, 81, 82, 92, 93, 94, 84, 74, 64, 54, 55, 45, 44, 34, 33]
        contexts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
        # Combine positions and contexts into true states
        true_states = [np.array([num1, num2]) for num1, num2 in zip(positions, contexts)]

        # Number of time steps
        num_t = range(len(actions))
        
        # Expected observations calculated from MATLAB code, agent observes sleep on time step 14
        expected_O = []
        # Base state
        O_one = [np.zeros(100),
                np.array([1, 0, 0, 0]),  # no resource observed
                np.array([0, 0, 0, 0, 1])]  # no context observed
        O_one[0][positions[0]] = 1
        expected_O.append(O_one)

        # Iterate over the positions list and create subsequent states
        for i in range(1, len(positions)):
            O_new = copy.deepcopy(expected_O[i-1])
            O_new[0][positions[i-1]] = 0  # Set previous position to 0
            O_new[0][positions[i]] = 1    # Set current position to 1
            expected_O.append(O_new)
        
        # Special cases when the agent observes a resource / context
        expected_O[15][2] = np.array([0, 0, 1, 0, 0])  # context 2 observed on 16th time step

        # Call the get_observations() function
        for t in num_t:
            calculated_O = get_observations(
                A, true_states, t, num_modalities, num_states, num_resource_observations, num_context_observations
            )
            np.testing.assert_equal(calculated_O, expected_O[t], "Arrays are not equal.")

    def test_observe_resource(self):
        
        # Initialise experimental/environmental hyperparams
        num_states = 100
        contextual_food_locations = [70,42,56,77]
        contextual_water_locations = [72,32,47,66]
        contextual_sleep_locations = [63,43,48,58] 
        hill_1 = 54
        start_position = 50
        
        # Agentic hyperparams
        num_modalities = 3
        num_resource_observations = 4 # [none, food, water, sleep]
        num_context_observations = 5 # [summer, autumn, winter, spring, none]
        
        # Initialise distributions
        A, a, B, b, D = initialise_distributions(
            num_states,
            contextual_food_locations,
            contextual_water_locations,
            contextual_sleep_locations,
            hill_1,
            start_position)

        # From seventh trial in MATLAB with seed "1", specify equal-size vectors of actions, positions, and contexts at each time step
        actions = np.array([2, 3, 0, 3, 3, 1, 3, 2, 2, 2, 4, 4, 4, 4, 2, 2, 4, 2, 2, 4, 2])
        positions = [50, 51, 61, 61, 71, 81, 80, 90, 91, 92, 93, 83, 73, 63, 53, 54, 55, 45, 46, 47, 37]
        contexts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Combine positions and contexts into true states
        true_states = [np.array([num1, num2]) for num1, num2 in zip(positions, contexts)]

        # Number of time steps
        num_t = range(len(actions))
        
        # Expected observations calculated from MATLAB code, agent observes sleep on time step 14
        expected_O = []

        # Base state
        O_one = [np.zeros(100),
                np.array([1, 0, 0, 0]),  # no resource observed
                np.array([0, 0, 0, 0, 1])]  # no context observed
        O_one[0][positions[0]] = 1
        expected_O.append(O_one)

        # Iterate over the positions list and create subsequent states
        for i in range(1, len(positions)):
            O_new = copy.deepcopy(expected_O[i-1])
            O_new[0][positions[i-1]] = 0  # Set previous position to 0
            O_new[0][positions[i]] = 1    # Set current position to 1
            expected_O.append(O_new)

        # Special cases when the agent observes a resource / context
        expected_O[13][1] = np.array([0, 0, 0, 1])  # resource sleep observed on 14th time step
        expected_O[15][2] = np.array([1, 0, 0, 0, 0])  # context 0 observed on 16th time step

        # Call the get_observations() function
        for t in num_t:
            calculated_O = get_observations(
                A, true_states, t, num_modalities, num_states, num_resource_observations, num_context_observations
            )
            np.testing.assert_equal(calculated_O, expected_O[t], "Arrays are not equal.")
            
# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()