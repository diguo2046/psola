"""
tests.test_pitch

unit test for psola.pitch module

Author: jreinhold
Created on: Aug 17, 2017
"""

import unittest

import numpy as np

import psola.pitch.estimation as pest
from psola.experiment_config import ExperimentConfig


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
        py_erbs2hz = pest.erbs2hz(10)
        msg = "MATLAB version value: {}, Python version value: {}".format(matlab_erbs2hz, py_erbs2hz)
        self.assertTrue(np.isclose(matlab_erbs2hz, py_erbs2hz), msg=msg)

    def test_hz2erbs(self):
        """
        test my hz2erbs function against the MATLAB version
        created in the original SWIPE algorithm
        """
        # below value created via MATLAB call: hz2erbs(10)
        matlab_hz2erbs = 0.391982243396023
        py_hz2erbs = pest.hz2erbs(10)
        msg = "MATLAB version value: {}, Python version value: {}".format(matlab_hz2erbs, py_hz2erbs)
        self.assertTrue(np.isclose(matlab_hz2erbs, py_hz2erbs), msg=msg)

    def test_primes_2_to_n(self):
        """
        check that algorithm correctly produces primes
        """
        primes_2_to_200 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                                    37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                                    79, 83, 89, 97, 101, 103, 107, 109, 113,
                                    127, 131, 137, 139, 149, 151, 157, 163,
                                    167, 173, 179, 181, 191, 193, 197, 199])
        self.assertTrue(np.array_equal(primes_2_to_200, pest.primes_2_to_n(200)))

    def test_find(self):
        """ test the `find' implementation """
        x = np.arange(1, 5)
        self.assertEqual(pest.find(x > 2).shape, (2,))

    def test_pitch_estimation(self):
        """
        test pitch estimation algo with contrived small example
        if pitch is within 5 Hz, then say its good (for this small example,
        since the algorithm wasn't made for this type of synthesized signal)
        """
        cfg = ExperimentConfig(pitch_strength_thresh=-np.inf)
        # the next 3 variables are in Hz
        tolerance = 5
        fs = 48000
        f = 150
        # create a sine wave of f Hz freq sampled at fs Hz
        x = np.sin(2*np.pi * f/fs * np.arange(2**10))
        # estimate the pitch, it should be close to f
        p, t, s = pest.pitch_estimation(x, fs, cfg)
        self.assertTrue(np.all(np.abs(p - f) < tolerance))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
