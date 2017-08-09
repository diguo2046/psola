"""
psola.pitch.estimation


Author: jreinhold
Created on: Aug 09, 2017
"""

import numpy as np
from scipy import signal

from psola.psola.constants import MAX_PITCH_FREQ, MIN_PITCH_FREQ


def center_clipping(x, percent=30):
    """
    Performs center clipping, a spectral whitening process

    need some type of spectrum flattening so that the
    speech signal more closely approximates a periodic impulse train

    Args:
        x       (array): signal data
        percent (float): percent threshold to clip

    Returns:
        cc         (array): center clipped signal
        clip_level (float): value of clipping
    """
    max_amp = np.max(np.abs(x))
    clip_level = max_amp * (percent / 100)
    positive_mask = x > clip_level
    negative_mask = x < -clip_level
    cc = np.zeros(x.shape)
    cc[positive_mask] = x[positive_mask] - clip_level
    cc[negative_mask] = x[negative_mask] + clip_level
    return cc, clip_level


def pitch_detection(x, fs):
    """


    Args:
        x  (array): signal data
        fs (float): sample frequency

    Returns:
        pitch (float):
    """
    # Pitch range

    min_lag = np.round(fs / MAX_PITCH_FREQ)  # max freq corresponds to smallest lag
    max_lag = np.round(fs / MIN_PITCH_FREQ)  # vice versa

    cc = center_clipping(x)
    auto_corr = signal.correlate(cc, cc)
    pitch = 0

    return pitch


def pitch_estimation():
    pass


if __name__ == "__main__":
    pass