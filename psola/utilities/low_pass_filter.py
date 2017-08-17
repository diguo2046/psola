"""
psola.utilities.low_pass_filter

Implements a low pass butterworth filter without
the headaches

Author: jreinhold
Created on: Aug 09, 2017
"""

import numpy as np
from scipy.signal import butter, filtfilt

from psola.errors import PsolaError


def lpf(x, cutoff, fs, order=5):
    """
    low pass filters signal with Butterworth digital
    filter according to cutoff frequency

    filter uses Gustafsson’s method to make sure
    forward-backward filt == backward-forward filt

    Note that edge effects are expected

    Args:
        x      (array): signal data (numpy array)
        cutoff (float): cutoff frequency (Hz)
        fs       (int): sample rate (Hz)
        order    (int): order of filter (default 5)

    Returns:
        filtered (array): low pass filtered data
    """
    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist)
    if not np.all(np.abs(np.roots(a)) < 1):
        raise PsolaError('Filter with cutoff at {} Hz is unstable given '
                         'sample frequency {} Hz'.format(cutoff, fs))
    filtered = filtfilt(b, a, x, method='gust')
    return filtered
