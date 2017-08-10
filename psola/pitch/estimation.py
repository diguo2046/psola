"""
psola.pitch.estimation

Estimates pitch in a single speech frame

Author: jreinhold
Created on: Aug 09, 2017
"""

import numpy as np
from scipy import signal

from psola.constants import PITCH_FRAME_SIZE, HNR_FRAME_SIZE


def pitch_estimation(x, fs):
    """
    Estimate pitch (fundamental frequency, F0) of a single frame
    using autocorrelation

    Args:
        x  (array): signal data
        fs (float): sample frequency

    Returns:
        pitch (float): fundamental frequency in Hz
        hnr   (float): harmonic-to-noise ratio (in dB), measure of quality

    References:
        [1] Boersma, P. (1993). Accurate Short-term Analysis of the Fundamental Frequency
            and the Harmonics-to-Noise Ratio of a Samples Sound. IFA Proceedings Institute
            of Phonetic Sciences Proceedings, 17(17), 97–110.
    """

    # Hann window has best HNR values and least likely to change sounds [1]
    window = signal.hann(len(x))
    xw = x * window

    # calculate autocorrelation of windowed signal
    ra = signal.correlate(xw, xw, method='full') / np.sum(xw * xw)  # normalize by 0th lag

    # calculate autocorrelation of window for accurate estimation
    rw = signal.correlate(window, window)

    # estimate original autocorrelation
    rx = ra / rw

    # find max cor value's lag <=> F0 (pitch)
    i_h = len(rx) // 2      # index at half the data
    rx_h = rx[i_h:]         # split data in half (remove redundancy)
    i_max = rx_h.argmax()
    lags = np.linspace(0, PITCH_FRAME_SIZE, len(rx_h))
    pitch = lags[i_max]

    # calculate HNR (harmonic-to-noise ratio)
    peak = rx_h[i_max]
    hnr = 10 * np.log10(peak / (1 - peak))

    return pitch, hnr


if __name__ == "__main__":
    pass