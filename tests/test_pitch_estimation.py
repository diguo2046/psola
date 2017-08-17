"""
tests.test_pitch_estimation

unit test for psola.pitch.estimation

Author: jreinhold
Created on: Aug 17, 2017
"""

import unittest

import numpy as np

import psola.pitch.estimation as pest


class TestPitchEstimation(unittest.TestCase):

    def setUp(self):
        pass

    def test_erbs2hz(self):
        """
        test my erbs2hz function against the MATLAB version
        created in the original SWIPE algorithm
        """
        # below value created via MATLAB call: erbs2hz(10)
        matlab_erbs2hz = 4.432225043305505e+02
        my_erbs2hz = pest.erbs2hz(10)
        self.assertTrue(np.isclose(matlab_erbs2hz, my_erbs2hz))

    def test_hz2erbs(self):
        """
        test my hz2erbs function against the MATLAB version
        created in the original SWIPE algorithm
        """
        # below value created via MATLAB call: hz2erbs(10)
        matlab_hz2erbs = 0.391982243396023
        my_hz2erbs = pest.erbs2hz(10)
        self.assertTrue(np.isclose(matlab_hz2erbs, my_hz2erbs))

    def test_primes_2_to_n(self):
        primes_2_to_200 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                                    37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                                    79, 83, 89, 97, 101, 103, 107, 109, 113,
                                    127, 131, 137, 139, 149, 151, 157, 163,
                                    167, 173, 179, 181, 191, 193, 197, 199])
        self.assertTrue(np.array_equal(primes_2_to_200, pest.primes_2_to_n(200)))

    def test_pitch_strength_all_candidates(self):
        pass

    def test_pitch_strength_one_candidate(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
