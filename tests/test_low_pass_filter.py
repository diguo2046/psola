"""
tests.test_low_pass_filter

unit test for psola.utilities.low_pass_filter

Author: jreinhold
Created on: Aug 10, 2017
"""

import unittest

import numpy as np

from psola.utilities.low_pass_filter import lpf


class TestLowPassFilter(unittest.TestCase):

    def setUp(self):
        self.fs = 2000
        t = np.linspace(0, 1.0, 2**11)  # make 2^n for computational efficiency
        xlow = np.sin(2 * np.pi * 5 * t)
        xhigh = np.sin(2 * np.pi * 250 * t)
        self.x = xlow + xhigh

    def test_low_pass_filter(self):
        """
        filter sinusoid with 5 Hz and 250 Hz component with
        with 125 Hz cutoff lpf, the magnitude of the 5 Hz signal
        should be highest
        """
        filtered = lpf(self.x, 125, self.fs)
        freqs = np.fft.rfftfreq(len(filtered), d=1/self.fs)
        mag = np.abs(np.fft.rfft(filtered))
        max_idx = np.argmax(mag)
        tol = freqs[1] - freqs[0]
        self.assertTrue(np.isclose(5, freqs[max_idx], rtol=tol))

    def tearDown(self):
        del self.fs, self.x


if __name__ == '__main__':
    unittest.main()
