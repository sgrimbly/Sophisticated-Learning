import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from spm_utils import spm_cross  

class TestSPMCross(unittest.TestCase):
    # NOTE: Some results may _look_ different to MATLAB results but still be the same. This is due to Python's row-major ordering vs MATLAB column-major ordering. See https://www.youtube.com/watch?v=b5lYGvcBjy4. The results are the same, but the order of the elements in the array is different.
    def test_single_array(self):
        array = np.array([1, 2, 3])
        result = spm_cross(array)
        np.testing.assert_array_equal(result, array)

    def test_two_arrays(self):
        array1 = np.array([1, 2])
        array2 = np.array([3, 4])
        expected = np.outer(array1, array2)
        result = spm_cross(array1, array2)
        np.testing.assert_array_equal(result, expected)

    def test_more_than_two_arrays(self):
        array1 = np.array([1, 2])
        array2 = np.array([3, 4])
        array3 = np.array([5, 6])
        expected = np.outer(array1, np.outer(array2, array3)).reshape(2, 2, 2)
        result = spm_cross(array1, array2, array3)
        np.testing.assert_array_equal(result, expected)

    def test_list_input(self):
        array1 = np.array([1, 2])
        array2 = np.array([3, 4])
        list_input = [array1, array2]
        expected = np.outer(array1, array2)
        result = spm_cross(list_input)
        np.testing.assert_array_equal(result, expected)

    def test_empty_list(self):
        empty_list = []
        expected = np.array([])
        result = spm_cross(empty_list)
        np.testing.assert_array_equal(result, expected)

    def test_single_element_list(self):
        single_element_list = [np.array([1, 2, 3])]
        expected = np.array([1, 2, 3])
        result = spm_cross(single_element_list)
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
