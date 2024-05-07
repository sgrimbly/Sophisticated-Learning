import unittest
import numpy as np
import copy
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agent_utils import initialise_distributions, smooth_beliefs

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

class TestSmoothBeliefs(unittest.TestCase):
    def test_start(self):
        
        # When the MATLAB agent reaches the update to "a" for the first time at t = 2
        
        # Define posterior Q for two subsequent time steps
        Q=[]
        Q_one = [np.zeros(100), 
             np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q_one[0][start_position] = 1
        Q.append(Q_one)
        Q_two = copy.deepcopy(Q_one)
        Q_two[0][start_position] = 0
        Q_two[0][60] = 1 # second time step, agent moved up once
        Q.append(Q_two)
        
        # Define the observations O for two subsequent time steps
        O = []
        O_one = [np.zeros(100), 
             np.array([1, 0, 0, 0]), # no resource observed
             np.array([0, 0, 0, 0, 1])] # no context observed
        O_one[0][start_position] = 1
        O.append(O_one)
        O_two = copy.deepcopy(O_one)
        O_two[0][start_position] = 0
        O_two[0][60] = 1 # second time step, agent moved up once
        O.append(O_two)
        
        # Current time step (2 in MATLAB)
        t = 1
        
        # Backwards smoothing starts from the current time step minus 6, or 0 if t < 6
        smoothing_start = 0
        
        # a is updated from the first smoothing time step
        smoothing_t = smoothing_start + 1
        
        # Updated a from MATLAB, given the above parameters
        expected_a = copy.deepcopy(a)
        expected_a[1][0,60,0] = .2750
        expected_a[1][0,60,1] = .2750
        expected_a[1][0,60,2] = .2750
        expected_a[1][0,60,3] = .2750
        
        # Calculated 
        calculated_a = smooth_beliefs(O, Q, A, a, b, smoothing_start, smoothing_t, t)
        
        np.testing.assert_allclose(calculated_a[1], expected_a[1], atol=.0001)
        
    def test_hill(self):
        
        # When the MATLAB agent reaches the update to "a" when it is at the hill for the first time
        
        # Initialise distributions
        A, a, B, b, D = initialise_distributions(
            num_states,
            contextual_food_locations,
            contextual_water_locations,
            contextual_sleep_locations,
            hill_1,
            start_position)
        
        # Define posterior Q for seven subsequent time steps        
        Q = []
        positions = [92, 93, 94, 84, 74, 64, 54]  # List of positions to be set to 1

        # Base state
        Q_one = [D[0].T, D[1].T]  # probability density of causes
        Q_one[0][0,start_position] = 0
        Q_one[0][0,positions[0]] = 1
        Q.append(Q_one)

        # Iterate over the positions list and create subsequent states
        for i in range(1, len(positions)):
            Q_new = copy.deepcopy(Q[i-1])
            Q_new[0][0,positions[i-1]] = 0  # Reset previous index to 0
            Q_new[0][0,positions[i]] = 1    # Set current index to 1
            Q.append(Q_new)

        # Define the observations O for seven subsequent time steps
        O = []
        positions = [92, 93, 94, 84, 74, 64, 54]  # List of positions to be set to 1

        # Base state
        O_one = [np.zeros(100),
                np.array([1, 0, 0, 0]),  # no resource observed
                np.array([0, 0, 0, 0, 1])]  # no context observed
        O_one[0][positions[0]] = 1
        O.append(O_one)

        # Iterate over the positions list and create subsequent states
        for i in range(1, len(positions)):
            O_new = copy.deepcopy(O[i-1])
            O_new[0][positions[i-1]] = 0  # Reset previous index to 0
            O_new[0][positions[i]] = 1    # Set current index to 1
            O.append(O_new)

        # Special case for the last element where the context is observed
        O[-1][0][positions[-1]] = 0  # Reset the last index to 0
        O[-1][2] = np.array([0, 0, 1, 0, 0])  # context 3 observed
        
        # Update "a" in line with the values from MATLAB
        a[1][0,54,0] = .2750
        a[1][0,60,0] = .2750
        a[1][0,64,0] = .2750
        a[1][0,70,0] = .4500
        a[1][0,74,0] = .2767
        a[1][0,80,0] = .2750
        a[1][0,81,0] = .2750
        a[1][0,82,0] = .2750
        a[1][0,84,0] = .2800
        a[1][0,90,0] = .2750
        a[1][0,91,0] = .2750
        a[1][0,92,0] = .2750
        a[1][0,93,0] = .2900
        a[1][0,94,0] = .2845
        
        a[1][1,54,1] = .2750
        a[1][1,60,1] = .2750
        a[1][1,64,1] = .3100
        a[1][1,70,1] = .4500
        a[1][1,74,1] = .3415
        a[1][1,80,1] = .2750
        a[1][1,81,1] = .2750
        a[1][1,82,1] = .2750
        a[1][1,84,1] = .3698
        a[1][1,90,1] = .2750
        a[1][1,91,1] = .2750
        a[1][1,92,1] = .2750
        a[1][1,93,1] = .4175
        a[1][1,94,1] = .3950
        
        a[1][2,54,2] = .2750
        a[1][2,60,2] = .2750
        a[1][2,64,2] = .9400
        a[1][2,70,2] = .4500
        a[1][2,74,2] = .9067
        a[1][2,80,2] = .2750
        a[1][2,81,2] = .2750
        a[1][2,82,2] = .2750
        a[1][2,84,2] = .8752
        a[1][2,90,2] = .2750
        a[1][2,91,2] = .2750
        a[1][2,92,2] = .2750
        a[1][2,93,2] = .8167
        a[1][2,94,2] = .8452
        
        a[1][3,54,3] = .2750
        a[1][3,60,3] = .2750
        a[1][3,64,3] = .2750
        a[1][3,70,3] = .4500
        a[1][3,74,3] = .2750
        a[1][3,80,3] = .2750
        a[1][3,81,3] = .2750
        a[1][3,82,3] = .2750
        a[1][3,84,3] = .2751
        a[1][3,90,3] = .2750
        a[1][3,91,3] = .2750
        a[1][3,92,3] = .2750
        a[1][3,93,3] = .2758
        a[1][3,94,3] = .2753
        
        # Current time step (16 in MATLAB)
        t = 6
        
        # Backwards smoothing starts from the current time step minus 6
        smoothing_start = 0
        
        # a is updated from the first smoothing time step
        smoothing_t = smoothing_start + 1
        
        # 3D indices where "a" changed in MATLAB, given the above parameters
        data_ordered = [
            [0, 54, 0], [0, 54, 1], [0, 54, 2], [0, 54, 3],
            [0, 64, 1], [0, 64, 2],
            [0, 74, 0], [0, 74, 1], [0, 74, 2],
            [0, 84, 0], [0, 84, 1], [0, 84, 2], [0, 84, 3],
            [0, 93, 0], [0, 93, 1], [0, 93, 2], [0, 93, 3],
            [0, 94, 0], [0, 94, 1], [0, 94, 2], [0, 94, 3]
        ]

        # Convert each vector into a NumPy array
        expected_a_difference_indices = [np.array(vector) for vector in data_ordered]
        # Expected summed absolute difference (SAD)
        expected_a_sad = 4.2
        # Expected mean squared error (MSE)
        expected_a_mse = 0.0012
        
        # Calculated a
        a_old = copy.deepcopy(a)
        for smoothing_t in range(smoothing_start, t+1): # In this case range() needs to be inclusive of t
            a = smooth_beliefs(O, Q, A, a, b, smoothing_start, smoothing_t, t)
        calculated_a = copy.deepcopy(a)
        
        # Extract indices where "a" changed
        calculated_a_difference_indices = np.column_stack(np.where(a_old[1] != calculated_a[1]))
        
        # Calculate SAD
        calculated_a_sad = np.sum(np.abs(calculated_a[1]-a_old[1]))
        
        # Calculate MSE
        calculated_a_mse = np.mean((a[1] - a_old[1])**2)
        
        # Compare the indices where we expect "a" to have changed based on what happens in MATLAB
        np.testing.assert_array_equal(calculated_a_difference_indices, expected_a_difference_indices)
        
        # Compare SAD for "a"
        self.assertAlmostEqual(calculated_a_sad, expected_a_sad, delta=.0001)
        
        # Compare MSE for "a"
        self.assertAlmostEqual(calculated_a_mse, expected_a_mse, delta=.0001)
                
if __name__ == '__main__':
    unittest.main()