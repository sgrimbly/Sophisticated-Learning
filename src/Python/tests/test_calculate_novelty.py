import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from active_inference_utils import calculate_novelty

class TestCalculateNovelty(unittest.TestCase):
    def test_calculate_novelty_start(self):
        # Define P
        P = [np.zeros(100), 
             np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        P[0][50] = 1 # starting position
        
        # Define a
        a = [np.zeros((100, 100, 4)),  # 100x100x4
             np.zeros((4, 100, 4)),    # 4x100x4
             np.zeros((5, 100, 4))]    # 5x100x4

        # Set identity matrix as likelihood for position
        for i in range(100):
            a[0][i,i,:] = 1

        a[1] += .1
        
        a[2][4, :, :] = 1
        for i in range(4):
            a[2][i, 54, i] = 1
            a[2][4, 54, i] = 0
        
        # Define the observations O
        O = [np.zeros(100), 
             np.array([.25, .25, .25, .25]),
             np.array([0, 0, 0, 0, 1])] # no context observed
        O[0][50] = 1 # starting position
        
        num_factors = 2
        
        weights = {"Novelty":10, "Learning":40, "Epistemic":1, "Preference":10}
        
        # Novelty from MATLAB
        expected_novelty = .3628
        
        calculated_novelty = calculate_novelty(P, a, O, num_factors, weights)
        
        np.testing.assert_allclose(expected_novelty, calculated_novelty, atol=.01)
        
    def test_calculate_novelty_hill(self):
        # Define P
        P = [np.zeros(100), 
             np.array([[0, 0, 1, 0]])]  # probability density of causes
        P[0][50] = 1 # starting position
        
        # Define a
        a = [np.zeros((100, 100, 4)),  # 100x100x4
             np.zeros((4, 100, 4)),    # 4x100x4
             np.zeros((5, 100, 4))]    # 5x100x4

        # Set identity matrix as likelihood for position
        for i in range(100):
            a[0][i,i,:] = 1
        
        a[1] += .1
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
        
        a[2][4, :, :] = 1
        for i in range(4):
            a[2][i, 54, i] = 1
            a[2][4, 54, i] = 0
        
        # Define the observations O
        O = [np.zeros(100), 
             np.array([0.4783, 0.1739, 0.1739, 0.1739]), # no resource observed
             np.array([0, 0, 1, 0, 0])] # third context observed
        O[0][54] = 1 # agent on hill
        
        num_factors = 2
        
        weights = {"Novelty":10, "Learning":40, "Epistemic":1, "Preference":10}
        
        # Novelty from MATLAB
        expected_novelty = .3494
        
        # Calculate novelty from calculate_novelty
        calculated_novelty = calculate_novelty(P, a, O, num_factors, weights)
        
        # Compare whether they are (almost) equal
        np.testing.assert_allclose(expected_novelty, calculated_novelty, atol=.01)

if __name__ == '__main__':
    unittest.main()
