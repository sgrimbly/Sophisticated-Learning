import unittest
import numpy as np
import copy
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from math_utils import normalise_matrix_columns

class Test2DIndex3DArray(unittest.TestCase):
    def test_position_array(self):
        # Define y
        y = [np.zeros((100, 100, 4)),  # 100x100x4
             np.zeros((4, 100, 4)),    # 4x100x4
             np.zeros((5, 100, 4))]    # 5x100x4

        # Set identity matrix as likelihood for position
        for i in range(100):
            y[0][i,i,:] = 1
        
        # Array from MATLAB: vector of zeros with 1 at position 5, [4] in Python given zero-indexing
        expected_array = np.zeros(100)
        expected_array[4] = 1
        
        calculated_array = y[0].reshape(y[0].shape[0],-1, order='F')[:,4]
        
        np.testing.assert_array_equal(expected_array, calculated_array)
        
    def test_resource_array(self):
        
        y = [np.zeros((100, 100, 4)),  # 100x100x4
            np.zeros((4, 100, 4)),    # 4x100x4
            np.zeros((5, 100, 4))]    # 5x100x4
                
        y[1] += .1
        y[1] = normalise_matrix_columns(y[1])
        
        # Set updated concentration parameters y[2], values from MATLAB when agent is at hill for first time
        y[1][:,54,0] = [.4763, .1739, .1739, .1739]
        y[1][:,60,0] = [.4763, .1739, .1739, .1739]
        y[1][:,64,0] = [.4763, .1739, .1739, .1739]
        y[1][:,70,0] = [.6000, .1333, .1333, .1333]
        y[1][:,74,0] = [.4763, .1739, .1739, .1739]
        y[1][:,80,0] = [.4763, .1739, .1739, .1739]
        y[1][:,81,0] = [.4763, .1739, .1739, .1739]
        y[1][:,82,0] = [.4763, .1739, .1739, .1739]
        y[1][:,84,0] = [.4763, .1739, .1739, .1739]
        y[1][:,90,0] = [.4763, .1739, .1739, .1739]
        y[1][:,91,0] = [.4763, .1739, .1739, .1739]
        y[1][:,92,0] = [.4763, .1739, .1739, .1739]
        y[1][:,93,0] = [.4915, .1695, .1695, .1695]
        y[1][:,94,0] = [.4867, .1711, .1711, .1711]
        
        y[1][:,54,1] = [.4763, .1739, .1739, .1739]
        y[1][:,60,1] = [.4763, .1739, .1739, .1739]
        y[1][:,64,1] = [.5082, .1639, .1639, .1639]
        y[1][:,70,1] = [.6000, .1333, .1333, .1333]
        y[1][:,74,1] = [.5323, .1559, .1559, .1559]
        y[1][:,80,1] = [.4763, .1739, .1739, .1739]
        y[1][:,81,1] = [.4763, .1739, .1739, .1739]
        y[1][:,82,1] = [.4763, .1739, .1739, .1739]
        y[1][:,84,1] = [.5521, .1493, .1493, .1493]
        y[1][:,90,1] = [.4763, .1739, .1739, .1739]
        y[1][:,91,1] = [.4763, .1739, .1739, .1739]
        y[1][:,92,1] = [.4763, .1739, .1739, .1739]
        y[1][:,93,1] = [.5819, .1394, .1394, .1394]
        y[1][:,94,1] = [.5684, .1439, .1439, .1439]
        
        y[1][:,54,2] = [.4763, .1739, .1739, .1739]
        y[1][:,60,2] = [.4763, .1739, .1739, .1739]
        y[1][:,64,2] = [.7581, .0806, .0806, .0806]
        y[1][:,70,2] = [.6000, .1333, .1333, .1333]
        y[1][:,74,2] = [.7514, .0829, .0829, .0829]
        y[1][:,80,2] = [.4763, .1739, .1739, .1739]
        y[1][:,81,2] = [.4763, .1739, .1739, .1739]
        y[1][:,82,2] = [.4763, .1739, .1739, .1739]
        y[1][:,84,2] = [.7447, .0851, .0851, .0851]
        y[1][:,90,2] = [.4763, .1739, .1739, .1739]
        y[1][:,91,2] = [.4763, .1739, .1739, .1739]
        y[1][:,92,2] = [.4763, .1739, .1739, .1739]
        y[1][:,93,2] = [.7313, .0896, .0896, .0896]
        y[1][:,94,2] = [.7380, .0873, .0873, .0873]
        
        y[1][:,54,3] = [.4763, .1739, .1739, .1739]
        y[1][:,60,3] = [.4763, .1739, .1739, .1739]
        y[1][:,64,3] = [.4763, .1739, .1739, .1739]
        y[1][:,70,3] = [.6000, .1333, .1333, .1333]
        y[1][:,74,3] = [.4763, .1739, .1739, .1739]
        y[1][:,80,3] = [.4763, .1739, .1739, .1739]
        y[1][:,81,3] = [.4763, .1739, .1739, .1739]
        y[1][:,82,3] = [.4763, .1739, .1739, .1739]
        y[1][:,84,3] = [.4763, .1739, .1739, .1739]
        y[1][:,90,3] = [.4763, .1739, .1739, .1739]
        y[1][:,91,3] = [.4763, .1739, .1739, .1739]
        y[1][:,92,3] = [.4763, .1739, .1739, .1739]
        y[1][:,93,3] = [.4790, .1737, .1737, .1737]
        y[1][:,94,3] = [.4786, .1738, .1738, .1738]    
        
        # Array from MATLAB: vector of .25 from y{2}(:,300)
        expected_array = np.array((.25,.25,.25,.25))
        
        calculated_array = y[1].reshape(y[1].shape[0],-1, order='F')[:,299]
        
        np.testing.assert_array_equal(expected_array, calculated_array)
        
    def test_context_array(self):
        # Define y
        y = np.zeros((5, 100, 4))  # 5x100x4

        # Set likelihood for contexts
        y[4, :, :] = 1
        for i in range(4):
            y[i, 54, i] = 1
            y[4, 54, i] = 0
        
        # Array from MATLAB: vector of zeros and one gained by indexing y(:,400)
        expected_array = np.array((0, 0, 0, 0, 1))
        
        # Index y by [:,399] given zero-indexing
        calculated_array = y.reshape(y.shape[0],-1, order='F')[:,399]

        np.testing.assert_array_equal(expected_array, calculated_array)
    
    def test_multiplication(self):
        
        y = [np.zeros((100, 100, 4)),  # 100x100x4
            np.zeros((4, 100, 4)),    # 4x100x4
            np.zeros((5, 100, 4))]    # 5x100x4
                
        y[1] += .1
        y[1] = normalise_matrix_columns(y[1])
        
        # Set updated concentration parameters y[2], values from MATLAB when agent is at hill for first time
        y[1][:,54,0] = [.4763, .1739, .1739, .1739]
        y[1][:,60,0] = [.4763, .1739, .1739, .1739]
        y[1][:,64,0] = [.4763, .1739, .1739, .1739]
        y[1][:,70,0] = [.6000, .1333, .1333, .1333]
        y[1][:,74,0] = [.4763, .1739, .1739, .1739]
        y[1][:,80,0] = [.4763, .1739, .1739, .1739]
        y[1][:,81,0] = [.4763, .1739, .1739, .1739]
        y[1][:,82,0] = [.4763, .1739, .1739, .1739]
        y[1][:,84,0] = [.4763, .1739, .1739, .1739]
        y[1][:,90,0] = [.4763, .1739, .1739, .1739]
        y[1][:,91,0] = [.4763, .1739, .1739, .1739]
        y[1][:,92,0] = [.4763, .1739, .1739, .1739]
        y[1][:,93,0] = [.4915, .1695, .1695, .1695]
        y[1][:,94,0] = [.4867, .1711, .1711, .1711]
        
        y[1][:,54,1] = [.4763, .1739, .1739, .1739]
        y[1][:,60,1] = [.4763, .1739, .1739, .1739]
        y[1][:,64,1] = [.5082, .1639, .1639, .1639]
        y[1][:,70,1] = [.6000, .1333, .1333, .1333]
        y[1][:,74,1] = [.5323, .1559, .1559, .1559]
        y[1][:,80,1] = [.4763, .1739, .1739, .1739]
        y[1][:,81,1] = [.4763, .1739, .1739, .1739]
        y[1][:,82,1] = [.4763, .1739, .1739, .1739]
        y[1][:,84,1] = [.5521, .1493, .1493, .1493]
        y[1][:,90,1] = [.4763, .1739, .1739, .1739]
        y[1][:,91,1] = [.4763, .1739, .1739, .1739]
        y[1][:,92,1] = [.4763, .1739, .1739, .1739]
        y[1][:,93,1] = [.5819, .1394, .1394, .1394]
        y[1][:,94,1] = [.5684, .1439, .1439, .1439]
        
        y[1][:,54,2] = [.4763, .1739, .1739, .1739]
        y[1][:,60,2] = [.4763, .1739, .1739, .1739]
        y[1][:,64,2] = [.7581, .0806, .0806, .0806]
        y[1][:,70,2] = [.6000, .1333, .1333, .1333]
        y[1][:,74,2] = [.7514, .0829, .0829, .0829]
        y[1][:,80,2] = [.4763, .1739, .1739, .1739]
        y[1][:,81,2] = [.4763, .1739, .1739, .1739]
        y[1][:,82,2] = [.4763, .1739, .1739, .1739]
        y[1][:,84,2] = [.7447, .0851, .0851, .0851]
        y[1][:,90,2] = [.4763, .1739, .1739, .1739]
        y[1][:,91,2] = [.4763, .1739, .1739, .1739]
        y[1][:,92,2] = [.4763, .1739, .1739, .1739]
        y[1][:,93,2] = [.7313, .0896, .0896, .0896]
        y[1][:,94,2] = [.7380, .0873, .0873, .0873]
        
        y[1][:,54,3] = [.4763, .1739, .1739, .1739]
        y[1][:,60,3] = [.4763, .1739, .1739, .1739]
        y[1][:,64,3] = [.4763, .1739, .1739, .1739]
        y[1][:,70,3] = [.6000, .1333, .1333, .1333]
        y[1][:,74,3] = [.4763, .1739, .1739, .1739]
        y[1][:,80,3] = [.4763, .1739, .1739, .1739]
        y[1][:,81,3] = [.4763, .1739, .1739, .1739]
        y[1][:,82,3] = [.4763, .1739, .1739, .1739]
        y[1][:,84,3] = [.4763, .1739, .1739, .1739]
        y[1][:,90,3] = [.4763, .1739, .1739, .1739]
        y[1][:,91,3] = [.4763, .1739, .1739, .1739]
        y[1][:,92,3] = [.4763, .1739, .1739, .1739]
        y[1][:,93,3] = [.4790, .1737, .1737, .1737]
        y[1][:,94,3] = [.4786, .1738, .1738, .1738]
        
        # Array from MATLAB: vector y(2:end, 55) multiplied by 10
        expected_array = np.array((1.739, 1.739, 1.739))
        
        # Index y by [1:,54] given zero-indexing does not work, gives a (3,4) matrix
        # calculated_array = y[1][1:,54] * 10
        
        # Requires reshaping with column-major order
        calculated_array = y[1].reshape(y[1].shape[0],-1, order='F')[1:,54]*10
        
        np.testing.assert_allclose(expected_array, calculated_array, atol=.0001)
        
    def test_multiplication_toy(self): 
        # Now as in the case of the a_learning update within tree search
               
        # MATLAB 3D target array: y = cat(3, [1, 2, 3; 4, 5, 6; 7, 8, 9], [10, 11, 12; 13, 14, 15; 16, 17, 18]);
        
        # Define y
        y = np.array([[[1, 10], [2, 11], [3, 12]],
                      [[4, 13], [5, 14], [6, 15]],
                      [[7, 16], [8, 17], [9, 18]]])
        
        # Shape of y in MATLAB
        expected_y_shape = np.array((3,3,2))
        # Actual shape of y
        calculated_y_shape = np.array(np.shape(y))
        # Check whether shapes are equal
        np.testing.assert_array_equal(expected_y_shape, calculated_y_shape)
        
        # Expected value from y(2,3,1) in MATLAB
        expected_y_value = 6
        # Actual value of y 
        calculated_y_value = y[1, 2, 0]
        # Check whether values are equal
        np.testing.assert_equal(expected_y_value, calculated_y_value)
        
        # Expected array from y(2,:,1) in MATLAB
        expected_y_array = np.array((4, 5, 6))
        # Actual value of y 
        calculated_y_array = y[1, :, 0]
        # Check whether values are equal
        np.testing.assert_equal(expected_y_array, calculated_y_array)
        
        # Array in MATLAB when indexed by y(2:end, :) multiplied by 40 (learning weight)
        expected_array = np.array([[[1, 10], [2, 11], [3, 12]],
                                   [[160, 520], [200, 560], [240, 600]],
                                   [[280, 640], [320, 680], [360, 720]]])
        expected_array_shape = np.array((3,3,2))
        # Expected value from y(3,2,1) in MATLAB
        expected_value = 320
        # Expected sub array from y(:,3,1) in MATLAB
        expected_sub_array = np.array((3, 240, 360))
        
        # Indexing y by [1:, :] works, without correcting for row-major vs column-major
        calculated_array = copy.deepcopy(y)
        calculated_array[1:,:] = calculated_array[1:,:] * 40
        calculated_array_shape = np.array(np.shape(calculated_array))
        calculated_value = calculated_array[2,1,0]
        calculated_sub_array = calculated_array[:,2,0]

        # Check whether shapes are equal
        np.testing.assert_array_equal(calculated_array_shape, expected_array_shape)
        # Check whether specific value is equal
        self.assertEqual(calculated_value, expected_value)
        # Check whether indexed sub array is equal
        np.testing.assert_array_equal(calculated_sub_array, expected_sub_array)
        # Check whether arrays are equal
        np.testing.assert_array_equal(calculated_array, expected_array)

if __name__ == '__main__':
    unittest.main()
