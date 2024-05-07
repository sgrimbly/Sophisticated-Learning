import unittest
import numpy as np
import copy
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from spm_utils import spm_backwards
from agent_utils import initialise_distributions

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

class TestSPMBackwards(unittest.TestCase):
    def test_start(self):
        
        # When the MATLAB agent reaches the spm_backwards function for the first time at t = 2
        
        # Define posterior Q for two subsequent time steps
        Q=[]
        Q_one = [np.zeros(100), 
             np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q_one[0][start_position] = 1
        Q.append(Q_one)
        Q_two = copy.deepcopy(Q_one)
        Q_two[0][60] = 1 # second time step, agent moved once
        Q.append(Q_two)
        
        # Define the observations O for two subsequent time steps
        O = []
        O_one = [np.zeros(100), 
             np.array([1, 0, 0, 0]), # no resource observed
             np.array([0, 0, 0, 0, 1])] # no context observed
        O_one[0][start_position] = 1
        O.append(O_one)
        O_two = copy.deepcopy(O_one)
        O_two[0][60] = 1 # second time step, agent moved once
        O.append(O_two)
        
        # Current time step (2 in MATLAB)
        t = 1
        
        # Backwards smoothing starts from the current time step minus 6
        smoothing_start = 0
        
        # a is updated from the first smoothing time step
        smoothing_t = smoothing_start + 1
        
        # The posterior from spm_backwards in MATLAB, given the above parameters
        expected_P = [np.array([[.25,.25,.25,.25]]),np.array([[.25,.25,.25,.25]])]
        
        # Calculated P
        calculated_P = []
        for smoothing_t in range(smoothing_start, t+1): # In this case range() needs to be inclusive of t
            calc_P = spm_backwards(O, Q, A, B, smoothing_t, t)
            calculated_P.append(calc_P)
        
        np.testing.assert_allclose(calculated_P, expected_P, atol=.0001)

    def test_start_nans(self):
        
        # When the MATLAB agent reaches the spm_backwards function for the first time at t = 2
        
        # Define posterior Q for two subsequent time steps
        Q=[]
        Q_one = [np.zeros(100), 
             np.array([[float("nan"), float("nan"), float("nan"), float("nan")]])]  # probability density of causes
        Q_one[0][start_position] = 1
        Q.append(Q_one)
        Q_two = copy.deepcopy(Q_one)
        Q_two[0][60] = 1 # second time step, agent moved once
        Q.append(Q_two)
        
        # Define the observations O for two subsequent time steps
        O = []
        O_one = [np.zeros(100), 
             np.array([1, 0, 0, 0]), # no resource observed
             np.array([0, 0, 0, 0, 1])] # no context observed
        O_one[0][start_position] = 1
        O.append(O_one)
        O_two = copy.deepcopy(O_one)
        O_two[0][60] = 1 # second time step, agent moved once
        O.append(O_two)
        
        # Current time step (2 in MATLAB)
        t = 1
        
        # Backwards smoothing starts from the current time step minus 6
        smoothing_start = 0
        
        # a is updated from the first smoothing time step
        smoothing_t = smoothing_start + 1
        
        # The posterior from spm_backwards in MATLAB, given the above parameters
        expected_P = [np.array([[.25,.25,.25,.25]]),np.array([[.25,.25,.25,.25]])]
        
        # Calculated P
        calculated_P = []
        for smoothing_t in range(smoothing_start, t+1): # In this case range() needs to be inclusive of t
            calc_P = spm_backwards(O, Q, A, B, smoothing_t, t)
            calculated_P.append(calc_P)
        
        np.testing.assert_allclose(calculated_P, expected_P, atol=.0001)

    def test_hill(self):
        
        # When the MATLAB agent visits the hill for the first time
        
        # Define posterior Q for seven subsequent time steps        
        Q = []
        positions = [92, 93, 94, 84, 74, 64, 54]  # List of positions to be set to 1

        # Base state
        Q_one = [np.zeros(100), np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q_one[0][positions[0]] = 1
        Q.append(Q_one)

        # Iterate over the positions list and create subsequent states
        for i in range(1, len(positions)):
            Q_new = copy.deepcopy(Q[i-1])
            Q_new[0][positions[i-1]] = 0  # Reset previous index to 0
            Q_new[0][positions[i]] = 1    # Set current index to 1
            Q.append(Q_new)

        # Define the observations O for seven subsequent time steps
        O = []
        positions = [92, 93, 94, 84, 74, 64, 54]  # List of positions to be set to 1

        # Base posterior
        O_one = [np.zeros(100),
                np.array([1, 0, 0, 0]),  # no resource observed
                np.array([0, 0, 0, 0, 1])]  # no context observed
        O_one[0][positions[0]] = 1
        O.append(O_one)

        # Iterate over the positions list and create subsequent posteriors
        for i in range(1, len(positions)):
            O_new = copy.deepcopy(O[i-1])
            O_new[0][positions[i-1]] = 0  # Reset previous index to 0
            O_new[0][positions[i]] = 1    # Set current index to 1
            O.append(O_new)

        # Special case for the last element where the context is observed
        O[-1][0][positions[-1]] = 0  # Reset the last index to 0
        O[-1][2] = np.array([0, 0, 1, 0, 0])  # context 3 observed
        
        # Current time step (16 in MATLAB)
        t = 6
        
        # Backwards smoothing starts from the current time step minus 6
        smoothing_start = 0
        
        # a is updated from the first smoothing time step
        smoothing_t = smoothing_start + 1
        
        # The posterior from spm_backwards in MATLAB, given the above parameters
        expected_P = [np.array([[0.0305, 0.2321, 0.7352, 0.0021]]), np.array([[0.0214, 0.2036, 0.7738, 0.0011]]),
                      np.array([[0.0135, 0.1715, 0.8145, 0.0005]]), np.array([[0.0071, 0.1354, 0.8574, 0.0001]]),
                      np.array([[0.0025, 0.0950, 0.9025, 0]]), np.array([[0, 0.0500, 0.9500, 0]]),
                      np.array([[0.25, 0.25, 0.25, 0.25]])]
        
        # Calculated P
        calculated_P = []
        for smoothing_t in range(smoothing_start, t+1): # In this case range() needs to be inclusive of t
            temp_P = spm_backwards(O, Q, A, B, smoothing_t, t)
            calculated_P.append(temp_P)
        
        np.testing.assert_allclose(calculated_P, expected_P, atol=.0001)
        
    def test_post_hill(self):
        
        # When the MATLAB agent leaves the hill after having visited it for the first time
        
        # Define posterior Q for seven subsequent time steps
        Q = []
        positions = [93, 94, 84, 74, 64, 54, 55]  # List of positions to be set to 1

        # Base posterior
        Q_one = [np.zeros(100), 
                np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q_one[0][positions[0]] = 1
        Q.append(Q_one)

        # Iterate over the positions list and create subsequent posteriors
        for i in range(1, len(positions)):
            Q_new = copy.deepcopy(Q[i-1])
            Q_new[0][positions[i-1]] = 0  # Reset previous position to 0
            Q_new[0][positions[i]] = 1    # Set current position to 1

            # Handle special cases
            if positions[i] == 54:  # Agent arrives at hill
                Q_new[1] = np.array([[0, 0, 1, 0]])  # belief at hill time step corresponds to observed context
            elif positions[i] == 55:  # Agent moves off hill
                Q_new[1] = np.array([[0, 0, .95, .05]])  # agent is still quite sure it is in context 3

            Q.append(Q_new)

        # Define the observations O for seven subsequent time steps
        O = []
        positions = [93, 94, 84, 74, 64, 54, 55]  # List of positions to be set to 1

        # Base posterior
        O_one = [np.zeros(100), 
                np.array([1, 0, 0, 0]),  # no resource observed
                np.array([0, 0, 0, 0, 1])]  # no context observed
        O_one[0][positions[0]] = 1
        O.append(O_one)

        # Iterate over the positions list and create subsequent posteriors
        for i in range(1, len(positions)):
            O_new = copy.deepcopy(O[i-1])
            O_new[0][positions[i-1]] = 0  # Reset previous position to 0
            O_new[0][positions[i]] = 1    # Set current position to 1

            # Handle special cases
            if positions[i] == 54:  # Agent arrives at hill
                O_new[2] = np.array([0, 0, 1, 0, 0])  # context 3 observed
            elif positions[i] == 55:  # Agent moves off hill
                O_new[2] = np.array([0, 0, 0, 0, 1])  # no context observed

            O.append(O_new)

        # Current time step (17 in MATLAB)
        t = 6
        
        # Backwards smoothing starts from the current time step minus 6
        smoothing_start = 0
        
        # a is updated from the first smoothing time step
        smoothing_t = smoothing_start + 1
        
        # The posterior from spm_backwards in MATLAB, given the above parameters
        expected_P = [np.array([[0.0214, 0.2036, 0.7738, 0.0011]]), np.array([[0.0135, 0.1715, 0.8145, 0.0005]]),
                      np.array([[0.0071, 0.1354, 0.8574, 0.0001]]), np.array([[0.0025, 0.0950, 0.9025, 0]]),
                      np.array([[0, 0.0500, 0.9500, 0]]), np.array([[0, 0, 1, 0]]), np.array([[0, 0, 0.9500, 0.0500]])]
        
        # Calculated 
        calculated_P = []
        for smoothing_t in range(smoothing_start, t+1): # In this case range() needs to be inclusive of t
            temp_P = spm_backwards(O, Q, A, B, smoothing_t, t)
            calculated_P.append(temp_P)
        
        np.testing.assert_allclose(calculated_P, expected_P, atol=.0001)
        
if __name__ == '__main__':
    unittest.main()
