# test_update_needs.py

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

from agent_utils import initialise_distributions, update_needs
from sophisticated_agent import initialise_experiment

class TestUpdateNeeds(unittest.TestCase):
    
    def test_no_resource_found(self):
        
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
        
        # From first trial in MATLAB with seed "1", specify equal-size vectors of actions, positions, and contexts at each time step
        actions = np.array([3, 3, 0, 3, 3, 2, 4, 2, 3, 2, 2, 4, 4, 4, 4, 2, 4, 1, 4, 1, 1])
        positions = [50, 60, 70, 70, 80, 90, 91, 81, 82, 92, 93, 94, 84, 74, 64, 54, 55, 45, 44, 34, 33]
        contexts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
        # Combine positions and contexts into true states
        true_states = [np.array([num1, num2]) for num1, num2 in zip(positions, contexts)]

        # Number of time steps
        num_t = range(len(actions))
        
        # Values calculated from MATLAB code
        expected_time_since_resource = [{"Food": i, "Water": i, "Sleep": i} for i in range(21)]

        # Call the update_needs function
        for t in num_t:
            calculated_time_since_resource = update_needs(
                true_states, t, 
                contextual_food_locations, 
                contextual_water_locations, 
                contextual_sleep_locations, 
                time_since_resource
            )
            np.testing.assert_equal(calculated_time_since_resource, expected_time_since_resource[t], "Arrays are not equal.")


    def test_find_resource(self):
        
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
        
        # From seventh trial in MATLAB with seed "1", specify equal-size vectors of actions, positions, and contexts at each time step
        actions = np.array([2, 3, 0, 3, 3, 1, 3, 2, 2, 2, 4, 4, 4, 4, 2, 2, 4, 2, 2, 4, 2])
        positions = [50, 51, 61, 61, 71, 81, 80, 90, 91, 92, 93, 83, 73, 63, 53, 54, 55, 45, 46, 47, 37]
        contexts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Combine positions and contexts into true states
        true_states = [np.array([num1, num2]) for num1, num2 in zip(positions, contexts)]

        # Number of time steps
        num_t = range(len(actions))
        
        # Values calculated from MATLAB code, agent finds sleep on time step 14
        expected_time_since_resource = [
                                        {"Food": 0, "Water": 0, "Sleep": 0},
                                        {"Food": 1, "Water": 1, "Sleep": 1},
                                        {"Food": 2, "Water": 2, "Sleep": 2},
                                        {"Food": 3, "Water": 3, "Sleep": 3},
                                        {"Food": 4, "Water": 4, "Sleep": 4},
                                        {"Food": 5, "Water": 5, "Sleep": 5},
                                        {"Food": 6, "Water": 6, "Sleep": 6},
                                        {"Food": 7, "Water": 7, "Sleep": 7},
                                        {"Food": 8, "Water": 8, "Sleep": 8},
                                        {"Food": 9, "Water": 9, "Sleep": 9},
                                        {"Food": 10, "Water": 10, "Sleep": 10},
                                        {"Food": 11, "Water": 11, "Sleep": 11},
                                        {"Food": 12, "Water": 12, "Sleep": 12},
                                        {"Food": 13, "Water": 13, "Sleep": 0},
                                        {"Food": 14, "Water": 14, "Sleep": 1},
                                        {"Food": 15, "Water": 15, "Sleep": 2},
                                        {"Food": 16, "Water": 16, "Sleep": 3},
                                        {"Food": 17, "Water": 17, "Sleep": 4},
                                        {"Food": 18, "Water": 18, "Sleep": 5},
                                        {"Food": 19, "Water": 19, "Sleep": 6},
                                        {"Food": 20, "Water": 20, "Sleep": 7}]

        # Call the update_needs function
        for t in num_t:
            calculated_time_since_resource = update_needs(
                true_states, t, 
                contextual_food_locations, 
                contextual_water_locations, 
                contextual_sleep_locations, 
                time_since_resource
            )
            np.testing.assert_equal(calculated_time_since_resource, expected_time_since_resource[t], "Arrays are not equal.")
            
# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()