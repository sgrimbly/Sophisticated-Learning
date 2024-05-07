# test_determine_observation_preference.py

import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from active_inference_utils import determine_observation_preference

class TestDetermineObservationPreference(unittest.TestCase):
    
    def test_start(self):

        time_since_resource = {"Food": 1, "Water": 1, "Sleep": 1}
        resource_constraints = {"Food":22,"Water":20,"Sleep":25}
        weights = {"Novelty":10, "Learning":40, "Epistemic":1, "Preference":10}
  
        # Value calculated from MATLAB code
        expected_C = np.array([-.1, .1, .1, .1]) # uniform preference

        # Call the determine_observation_preference function
        calculated_C = determine_observation_preference(time_since_resource, resource_constraints, weights)

        # Assert that the expected and calculated values are close enough
        self.assertTrue(np.array_equal(expected_C, calculated_C),
                        "Arrays are not equal")
        
    def test_death(self):

        time_since_resource = {"Food": 21, "Water": 21, "Sleep": 21}
        resource_constraints = {"Food":22,"Water":20,"Sleep":25}
        weights = {"Novelty":10, "Learning":40, "Epistemic":1, "Preference":10}
  
        # Value calculated from MATLAB code
        expected_C = np.array([-50, -50, 2.1, -50]) # uniform preference

        # Call the determine_observation_preference function
        calculated_C = determine_observation_preference(time_since_resource, resource_constraints, weights)
        
        # Assert that the expected and calculated values are close enough
        self.assertTrue(np.array_equal(expected_C, calculated_C),
                        "Arrays are not equal")

# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()