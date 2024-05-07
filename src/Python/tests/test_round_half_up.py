import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from math_utils import round_half_up

class TestRoundHalfUp(unittest.TestCase):
    def test_number(self):
        
        # Test single number

        # Define number
        x = .5
        
        # Expected value
        expected_value = 1

        # Calculated rounded number
        calculated_value = round_half_up(x)
        
        np.testing.assert_equal(calculated_value, expected_value)

        # Define number
        x = .05
        
        # Expected value
        expected_value = .1

        # Calculated rounded number
        calculated_value = round_half_up(x,1)
        
        np.testing.assert_equal(calculated_value, expected_value)
        
        # Define number
        x = .005
        
        # Expected value
        expected_value = .01

        # Calculated rounded number
        calculated_value = round_half_up(x,2)
        
        np.testing.assert_equal(calculated_value, expected_value)

    def test_list_numbers(self):
        
        # Test single number

        # Define numbers
        x = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        y = [0.306276485014254, 0.307727356359374, 0.118639346645595, 0.267356811980777]
        
        # Expected rounded values
        expected_x = [0, 0, 0, 0, 1, 1, 1, 1, 1]
        expected_y = [0.3, 0.3, 0.1, 0.3]

        # Calculated rounded numbers
        calculated_x = round_half_up(x)
        calculated_y = round_half_up(y,1)
        
        np.testing.assert_equal(calculated_x, expected_x)
        np.testing.assert_equal(calculated_y, expected_y)

    def test_np_array(self):
        
        # Test array

        # Define numbers
        x = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        
        # Expected values from MATLAB
        expected_value = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Calculated rounded numbers
        calculated_value = round_half_up(x)
        
        np.testing.assert_equal(calculated_value, expected_value)
        
    def test_np_array(self):
        
        # Test single number

        # Define numbes
        x = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        
        # Expected values from MATLAB
        expected_value = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Calculated rounded numbers
        calculated_value = round_half_up(x)
        
        np.testing.assert_equal(calculated_value, expected_value)

    def test_list_np_arrays(self):
        
        # Test single number

        # Define numbers
        x = [np.array([2.05, 2.55]), np.array([3.005, 4.245]), np.array([5.1234, 6.7891])]
        
        # Expected values from MATLAB
        expected_value = [np.array([2.05, 2.55]), np.array([3.01, 4.25]), np.array([5.12, 6.79])]

        # Calculated rounded numbers
        calculated_value = round_half_up(x,2)
        
        np.testing.assert_equal(calculated_value, expected_value)
        
if __name__ == '__main__':
    unittest.main()