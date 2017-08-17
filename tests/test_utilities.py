"""
tests.test_utilities

unit test for psola.utilities module

Author: jreinhold
Created on: Aug 10, 2017
"""

import unittest

import numpy as np

from psola.errors import PsolaError
from psola.utilities.low_pass_filter import lpf
from psola.utilities.center_clipping import center_clipping


class TestLowPassFilter(unittest.TestCase):

    def setUp(self):
        self.fs = 2000
        t = np.linspace(0, 1.0, 2**11)  # make 2^n for computational efficiency
        xlow = np.sin(2 * np.pi * 5 * t)
        xhigh = np.sin(2 * np.pi * 250 * t)
        self.x = xlow + xhigh

    def test_low_pass_filter(self):
        """
        check that low pass filter has expected behavior

        filter sinusoid with 5 Hz and 250 Hz component (see setUp method)
        with 125 Hz cutoff lpf, the magnitude of the 5 Hz signal
        should be highest
        """
        filtered = lpf(self.x, 125, self.fs)
        freqs = np.fft.rfftfreq(len(filtered), d=1/self.fs)
        mag = np.abs(np.fft.rfft(filtered))
        max_idx = np.argmax(mag)
        tol = freqs[1] - freqs[0]
        self.assertTrue(np.isclose(5, freqs[max_idx], rtol=tol))

    def test_low_pass_filter_fail(self):
        """ check that filter raises exception properly """
        self.assertRaises(PsolaError, lpf, self.x, self.fs/2, self.fs)

    def tearDown(self):
        del self.fs, self.x


class TestCenterClipping(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(11)  # create array from 0 to 10

    def test_center_clipping(self):
        cc, clip_level = center_clipping(self.x, percent=30)
        truth = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(clip_level, 3)
        self.assertTrue(np.array_equal(cc, truth))

    def tearDown(self):
        del self.x


if __name__ == '__main__':
    unittest.main()
