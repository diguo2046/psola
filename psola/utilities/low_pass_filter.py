"""
psola.utilities.low_pass_filter


Author: jreinhold
Created on: Aug 09, 2017
"""

from scipy.signal import butter, filtfilt


def lpf(x, cutoff, fs, order=5):
    """
    low pass filters signal with Butterworth digital
    filter according to cutoff frequency

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
    filtered = filtfilt(b, a, x)
    return filtered


if __name__ == "__main__":
    pass
