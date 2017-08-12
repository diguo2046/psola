#!/usr/bin/env python
"""
psola.pitch.estimation

Methods to estimates pitch for a single frame or entire signal

Author: jreinhold
Created on: Aug 09, 2017
"""

import numpy as np
from scipy import signal


def preprocessing(x):
    """
    Interpolates (2x) and reduces power of frequencies in signal
    near Nyquist. Works on the entire signal, follows Step 1 of
    the algorithm listed in [1]

    Args:
        x  (array): signal data (array of real numbers)

    Returns:
       preprocessed (array): preprocessed signal data (interpolated and filtered)

    References:
        [1] Boersma, P. (1993). Accurate Short-term Analysis of the Fundamental Frequency
            and the Harmonics-to-Noise Ratio of a Samples Sound. IFA Proceedings Institute
            of Phonetic Sciences Proceedings, 17(17), 97–110.
    """
    # front-load info definition
    l = len(x)

    # preprocessing will be done in the freq domain
    # do rfft instead of fft because signal is real, computational efficiency
    X = np.fft.rfft(x)

    # create window to multiply frequency by to reduce power linearly
    # of frequencies near Nyquist (in this case at 95% of Nyquist)
    window = np.ones(l)
    i_w = np.floor(0.95 * l)
    taper = np.linspace(1, 0, l - i_w)
    window[i_w:] = taper

    # implement the phrase: "do ifft of order one higher than the first fft"
    next_power_2 = int(np.floor(np.log2(l)) + 1)  # TODO: determine if ceil or floor correct
    preprocessed = np.fft.irfft(X, 2**next_power_2)

    return preprocessed


def pitch_estimation(x, fs, global_max=None, experiment_config=None):
    """
    Estimate pitch (fundamental frequency, F0) of a single frame
    using autocorrelation

    Args:
        x  (array): signal data (array of real numbers)
        fs (float): sampling frequency of data
        global_max (float): global maximum of signal
        experiment_config (psola.experiment_config.ExperimentConfig instance)

    Returns:
        pitch (float): fundamental frequency in Hz
        hnr   (float): harmonic-to-noise ratio (in dB), measure of quality

    References:
        [1] Boersma, P. (1993). Accurate Short-term Analysis of the Fundamental Frequency
            and the Harmonics-to-Noise Ratio of a Samples Sound. IFA Proceedings Institute
            of Phonetic Sciences Proceedings, 17(17), 97–110.
    """
    # handle these cases to make function more standalone
    if global_max is None:
        global_max = np.max(x)
    if experiment_config is None:
        from psola.experiment_config import ExperimentConfig
        experiment_config = ExperimentConfig()

    # Zero-mean signal, Step 3.2 in [1]
    frame_ = x - x.mean()
    frame_size = len(frame_)

    # Hann window has best HNR values and least likely to change sounds [1]
    window = signal.hann(frame_size)
    xw = frame_ * window  # Step 3.4 in [1]

    # Append half a window length of zeroes and go to next power of 2,
    # Step 3.5 & 3.6 in [1]
    l = len(xw) + len(window) // 2
    next_power_2 = int(np.floor(np.log2(l)) + 1)
    xw_ = np.zeros(2**next_power_2)

    # calculate autocorrelation of windowed signal, Step 3.7-3.9
    ra = signal.correlate(xw_, xw_, method='full') / np.sum(xw_ * xw_)  # normalize by 0th lag

    # calculate autocorrelation of window for accurate estimation
    window = signal.hann(len(ra))  # need to recalculate window for Step 3.10  # TODO: verify correct
    rw = signal.correlate(window, window)

    # estimate original autocorrelation, Step 3.10
    rx = ra / rw

    # define range of indices over which to search for pitch candidates
    search_range = (1/experiment_config.max_pitch, experiment_config.min_pitch // fs)

    # find max cor value's lag <=> F0 (pitch)
    i_h = len(rx) // 2      # index at half the data
    rx_h = rx[i_h:]         # split data in half (remove redundancy)
    i_max = rx_h.argmax()
    frame_time = frame_size / fs  # length of time associated with a frame (s)
    lags = np.linspace(0, frame_time, len(rx_h))
    pitch = lags[i_max]

    # calculate HNR (harmonic-to-noise ratio)
    peak = rx_h[i_max]
    hnr = 10 * np.log10(peak / (1 - peak))

    return pitch, hnr


def pitch_track(pitch_obj):
    """
    tracks pitch through entire signal, main driver in module

    Args:
        pitch_obj (psola.pitch.pitch_config.Pitch instance)

    Returns:

    References:
        [1] Boersma, P. (1993). Accurate Short-term Analysis of the Fundamental Frequency
            and the Harmonics-to-Noise Ratio of a Samples Sound. IFA Proceedings Institute
            of Phonetic Sciences Proceedings, 17(17), 97–110.
    """

    # upsample and reduce power of freqs near Nyquist, Step 1 in Algorithm of [1]
    preprocessed = preprocessing(pitch_obj.x)

    # compute global absolute peak value of signal, Step 2 in Algorithm of [1]
    global_max = np.max(np.abs(preprocessed))

    # Get pitch for all frames, Step 3 in Algorithm of [1]
    # frames is a frame that uses parameters defined in ExperimentConfig and
    # creates frames (of which each are a Pitch obj) from the array passed in as an argument
    for frame in pitch_obj.frames(preprocessed):  # getting the frames is Step 3.1
        pitch, hnr = pitch_estimation(frame, pitch_obj.fs, global_max, pitch_obj.config)

if __name__ == "__main__":
    pass