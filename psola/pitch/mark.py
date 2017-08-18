#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
psola.pitch.mark

mark glottal closure instants as approximation to pitch
period for TD-PSOLA implementation, based off [1]

References:
    [1] Serwy, R. D. (2017, April 7). Hilbert phase methods
        for glottal activity detection. University of Illinois
        at Urbana-Champaign. Retrieved from
        https://www.ideals.illinois.edu/handle/2142/97304

Author: jreinhold
Created on: Aug 18, 2017
"""

import numpy as np
from scipy import signal

from psola.utilities.find import find


def pitch_mark(data, fs, cfg=None,
               theta=-np.pi/2, filter_reps=2,
               gci_thresh=-1.5*np.pi):
    """
    Mark glottal closure instances in a speech signal
    using the QuickGCI algorithm as described in [1]

    Will only work for clean speech signals
    (i.e., little noise or interference)

    Note that inline steps of the form: n)
    are taken directly from [1]

    Args:
        data (array): numpy array of speech audio data
        fs   (float): sample frequency of `data'
        cfg  (psola.experiment_config.ExperimentConfig instance)
        theta (float): rotation parameter, default value for speech [1]
        filter_reps (int): number of times to run filter on data
        gci_thresh (float): threshold for magnitude of phase
                              discontinuities to accept as GCI

    Returns:
        gci (array): indices of GCIs in `data'

    References:
        [1] Serwy, R. D. (2017, April 7). Hilbert phase methods
            for glottal activity detection. University of Illinois
            at Urbana-Champaign. Retrieved from
            https://www.ideals.illinois.edu/handle/2142/97304
    """
    if cfg is None:
        from psola.experiment_config import ExperimentConfig
        cfg = ExperimentConfig()

    # 1) Apply a first-order high-pass and first-order
    #    low-pass filter to the input signal, forward
    #    and backward twice in time to preserve GCI locations.

    # define filter parameters
    nyquist = fs / 2
    n_min_f = cfg.min_pitch / nyquist
    n_max_f = cfg.max_pitch / nyquist

    # create filter coefficients
    b_h, a_h = signal.butter(1, n_max_f, 'high')
    b_l, a_l = signal.butter(1, n_min_f, 'low')

    # band-pass filter data `filter_reps' times
    for _ in range(filter_reps):
        data = signal.filtfilt(b_h, a_h, data)
        data = signal.filtfilt(b_l, a_l, data)

    # 2) Compute the analytic signal for x(t) by taking its
    #    Hilbert transform and allow for rotation by θ
    x = __hilbert(data) * np.exp(1j * theta)

    # 3) Multiply the envelope by the negative imaginary
    #    component of the analytic signal.
    q = np.abs(x) * np.imag(-x)

    # 4) Low-pass filter the signal q(t) to smooth
    #    high-frequency self-modulations.
    for _ in range(filter_reps):
        q = signal.filtfilt(b_l, a_l, q)

    # 5) Compute the analytic signal of r(t) and find its
    #    positive-to-negative 2π phase discontinuities
    r = __hilbert(q)
    dphi = __diff(np.angle(r))
    gci = find(dphi < gci_thresh)

    return gci


def __hilbert(x):
    """
    Hilbert transform to the power of 2

    Args:
        x (array): numpy array of data

    Returns:
        y (array): numpy array of Hilbert transformed x
                     (same size as x)
    """
    l = x.size
    pad = int(2**(np.floor(np.log2(l)) + 1))
    y = signal.hilbert(x, N=pad)[:l]
    return y


def __diff(x):
    """
    First derivative/diff (while keeping same size as input)

    Args:
        x (array): numpy array of data

    Returns:
        dx (array): numpy array of first derivative of data
                      (same size as x)
    """
    dx = np.diff(x)
    dx = np.concatenate((dx[0], dx))  # output len == input len
    return dx
