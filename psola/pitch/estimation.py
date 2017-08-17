#!/usr/bin/env python
"""
psola.pitch.estimation

Implements a ``sawtooth waveform inspired pitch estimator'' (SWIPE) [1]

A previous swipe implementation in python [2] was also used as
a reference

References:
    [1] Camacho, A., & Harris, J. G. (2008). A sawtooth waveform
        inspired pitch estimator for speech and music. The Journal
        of the Acoustical Society of America, 124(3), 1638–1652.
        https://doi.org/10.1121/1.2951592

    [2] Bolaños, C. pyswipe, (2016), Github Repository.
        https://github.com/Tinuxx/pyswipe

Author: jreinhold
Created on: Aug 16, 2017
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def pitch_estimation(x, fs, cfg):
    """
    Estimates pitch for a time-series, implements SWIPE algorithm
    see [1] for more details

    Args:
        x   (array): real-valued numpy array (e.g., speech signal)
        fs  (float): sampling frequency of x
        cfg (psola.experiment_config.ExperimentConfig instance)

    Returns:
        pitch    (array): pitch corresponding to times
        t        (array): times
        strength (array): strength of pitch corresponding to times

    References:
        [1] Camacho, A., & Harris, J. G. (2008). A sawtooth waveform
            inspired pitch estimator for speech and music. The Journal
            of the Acoustical Society of America, 124(3), 1638–1652.
            https://doi.org/10.1121/1.2951592
    """
    dt = cfg.frame_step  # define time resolution of pitch estimation
    times = np.arange(0, x.size / np.float_(fs), dt)

    # define pitch candidates, lp := log2(pitch)
    lp_min, lp_max = np.log2(cfg.min_pitch), np.log2(cfg.max_pitch)
    lp_step = cfg.dlog2p
    lp_candidates = np.arange(lp_min, lp_max, lp_step)
    pitch_candidates = 2 ** lp_candidates.T

    # pitch strength matrix
    S = np.zeros((pitch_candidates.size, times.size))

    # determine power-of-2 window-sizes (P2-WSs)
    lws = np.round(np.log2(8 * fs / np.array([cfg.min_pitch, cfg.max_pitch])))
    wss = 2 ** np.arange(lws[0], lws[1]-1, -1)  # note the -1 for inclusion
    p0s = 8 * fs / wss  # optimal pitches for P2-WSs

    # determine window sizes used by each pitch candidate
    window_size = 1 + lp_candidates - np.log2(8 * fs / wss[0])

    # create ERB-scale uniformly-spaced freqs (Hz)
    min_hz = hz2erbs(pitch_candidates.min() / 4)  # TODO: why divide by 4?
    max_hz = hz2erbs(fs / 2)  # Nyquist freq
    hz_scale = np.arange(min_hz, max_hz, cfg.dERBs)
    f_erbs = erbs2hz(hz_scale)

    for i, (ws, p0) in enumerate(zip(wss, p0s)):

        hop_size = max(1, np.round(8 * (1 - cfg.window_overlap) * fs / p0))

        # zero pad signal
        before = np.zeros(int(ws/2))
        after = np.zeros(int(ws/2 + hop_size))
        xzp = np.concatenate((before, x, after))

        w = signal.hann(int(ws))
        noverlap = int(max(0, np.round(ws - hop_size)))  # window overlap for spectrogram

        # Note that there MATLAB and Scipy implement specgram diff,
        # so values will NOT be equivalent (there are no options in Python's
        # implementation to make the two funcs equal).
        spec_kwargs = dict(x=xzp, fs=fs, window=w, nperseg=w.size,
                           noverlap=noverlap, nfft=int(ws),
                           scaling='spectrum', mode='complex')
        freqs, ti, X = signal.spectrogram(**spec_kwargs)

        # Note: this is very opaque, learn what this is doing and provide explanation
        # select candidates that use this window size
        # np.squeeze used to remove single-dimension entries from array
        ii = i + 1
        if wss.size == 1:
            j = pitch_candidates.T
            k = np.array([])
        elif wss.size == ii:
            j = find(window_size-ii > -1)
            k = find(window_size[j] - ii < 0)
        elif ii == 1:
            j = find(window_size-ii < 1)
            k = find(window_size[j]-ii > 0)
        else:
            j = find(np.abs(window_size-ii) < 1)
            k = np.arange(0, j.size)

        # compute loudness at ERBs uniformly-spaced freqs
        idx = find(f_erbs > pitch_candidates[j[0]]/4)[0]
        f_erbs = f_erbs[idx:]
        f = interp1d(freqs, np.abs(X), axis=0, kind='cubic', fill_value=0)  # interp on columns
        interpd_X = f(f_erbs)
        interpd_X[interpd_X < 0] = 0
        L = np.sqrt(interpd_X)

        # compute pitch strength
        Si_ = pitch_strength_all_candidates(f_erbs, L, pitch_candidates[j])

        # replicate matlab behavior with ti, default extends one elem too far and doesn't include 0
        # default ti makes the interp1d stage not work, since 0 is not included
        ti = np.concatenate((np.zeros(1), ti[:-1]))
        # interpolate pitch strength at desired times
        if Si_.shape[1] > 1:
            f = interp1d(ti, Si_.T, axis=0, kind='linear', fill_value=np.nan)
            Si = f(times).T
        else:
            Si = np.empty((Si_.shape[0], times.size)) * np.nan  # TODO: test this line

        # add pitch strength to combination
        lambda_ = window_size[j[k]] - (i + 1)
        mu = np.ones(j.shape)
        mu[k] = 1 - np.abs(lambda_)
        S[j, :] = S[j, :] + np.tile(mu, (Si.shape[1], 1)).T * Si

    # fine-tune pitch using parabolic interp
    pitch = np.empty(S.shape[1]) * np.nan
    strength = np.empty(S.shape[1]) * np.nan
    for j in range(S.shape[1]):
        strength[j] = np.nanmax(S[:, j])
        i = np.nanargmax(S[:, j])
        if strength[j] < cfg.pitch_strength_thresh:
            continue
        if i == 0 or i == pitch_candidates.size - 1:
            pitch[j] = pitch_candidates[i]
        else:
            I = np.arange(i - 1, i + 2)  # funky additions to mimic MATLAB
            tc = 1 / pitch_candidates[I]
            ntc = (tc / tc[1] - 1) * 2 * np.pi
            c = np.polyfit(ntc, S[I, j], 2)
            # TODO: why are these params hardcoded and what is meaning?
            ftc_low = np.log2(pitch_candidates[I[0]])
            ftc_high = np.log2(pitch_candidates[I[2]])
            ftc_step = 1/12/100
            ftc = 1 / 2 ** np.arange(ftc_low, ftc_high, ftc_step)
            nftc = (ftc/tc[1] - 1) * 2 * np.pi
            polyfit_nftc = np.polyval(c, nftc)
            strength[j] = np.nanmax(polyfit_nftc)
            k = np.argmax(polyfit_nftc)
            pitch[j] = 2 ** (ftc_low + (k - 1) / 12 / 100)

    return pitch, times, strength


def pitch_strength_all_candidates(f_erbs, L, pc):
    """
    Calculates the pitch ``strength'' of all candidate
    pitches

    Args:
        f_erbs (array): frequencies in ERBs
        L     (matrix): loudness matrix
        pc     (array): pitch candidates array

    Returns:
        S   (array): strength of pitches corresponding to pc's
    """
    # create pitch strength matrix
    S = np.zeros((pc.size, L.shape[1]))

    # define integration regions
    k = np.zeros(pc.size+1)

    for j in range(k.size-1):
        idx = int(k[j])
        f = f_erbs[idx:]
        val = find(f > pc[j] / 4)[0]
        k[j+1] = k[j] + val

    k = k[1:]  # TODO: fix this sloppiness

    # create loudness normalization matrix
    N = np.sqrt(np.flipud(np.cumsum(np.flipud(L * L), 0)))
    for j in range(pc.size):
        # normalize loudness
        n = N[int(k[j]), :]
        n[n == 0] = -np.inf  # to make zero-loudness equal zero after normalization
        nL = L[int(k[j]):] / np.tile(n, (int(L.shape[0] - k[j]), 1))

        # compute pitch strength
        S[j] = pitch_strength_one_candidate(f_erbs[int(k[j]):], nL, pc[j])

    return S


def pitch_strength_one_candidate(f_erbs, nL, pc):
    """
    Calculates the pitch ``strength'' for a single
    candidate

    Args:
        f_erbs (array):
        nL           : normalized loudness
        pc           : pitch candidate

    Returns:
        s     (float): value of strength for a pitch
    """

    # fix rounds a number *towards* zero
    n = int(np.fix(f_erbs[-1] / pc - 0.75))  # number of harmonics
    if n == 0:
        return np.nan
    k = np.zeros(f_erbs.shape)  # kernel

    # normalize freq w.r.t. candidate
    q = f_erbs / pc

    # create kernel
    primes = np.concatenate((np.ones(1), primes_2_to_n(n)))
    for i in primes:
        a = np.abs(q - i)

        # peak's weight
        p = a < 0.25
        k[p] = np.cos(2 * np.pi * q[p])

        # valley's weight
        v = np.logical_and(0.25 < a, a < 0.75)
        k[v] = k[v] + np.cos(2 * np.pi * q[v]) / 2

    # apply envelope
    k = k * np.sqrt(1 / f_erbs)

    # K+-normalized kernel
    k = k / np.linalg.norm(k[k>0])

    # strength value of pitch
    s = np.dot(k, nL)

    return s


def hz2erbs(hz):
    """
    Convert values in Hertz to values in Equivalent rectangle bandwidth (ERBs)

    Args:
        hz (float): real number in Hz

    Returns:
        erbs (float): real number in ERBs

    References:
    [1] Camacho, A., & Harris, J. G. (2008). A sawtooth waveform
        inspired pitch estimator for speech and music. The Journal
        of the Acoustical Society of America, 124(3), 1638–1652.
        https://doi.org/10.1121/1.2951592
    """
    erbs = 6.44 * (np.log2(229 + hz) - 7.84)
    return erbs


def erbs2hz(erbs):
    """
    Convert values in Equivalent rectangle bandwidth (ERBs) to
    values in Hertz (Hz)

    Parameters from [1]

    Args:
        erbs (float): real number in ERBs

    Returns:
        hz (float): real number in Hertz

    References:
        [1] Camacho, A., & Harris, J. G. (2008). A sawtooth waveform
            inspired pitch estimator for speech and music. The Journal
            of the Acoustical Society of America, 124(3), 1638–1652.
            https://doi.org/10.1121/1.2951592
    """
    hz = (2 ** ((erbs / 6.44) + 7.84)) - 229
    return hz


def primes_2_to_n(n):
    """
    Efficient algorithm to find and list primes from
    2 to `n'.

    Args:
        n (int): highest number from which to search for primes

    Returns:
        np array of all primes from 2 to n

    References:
        Robert William Hanks,
        https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/
    """
    sieve = np.ones(int(n / 3 + (n % 6 == 2)), dtype=np.bool)
    for i in range(1, int((n ** 0.5) / 3 + 1)):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[int(k * k / 3)::2 * k] = False
            sieve[int(k * (k - 2 * (i & 1) + 4) / 3)::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def find(x):
    """
    kind-of mimics the find command in matlab,
    really created to avoid repetition in code

    Args:
        x (numpy mask): condition, e.g., x < 5

    Returns:
        indices where x is true
    """
    return np.squeeze(np.where(x))


def test():
    """
    little test, NOT FOR REAL USE, put target filename as first and only argument
    will plot the pitch and strength v. time to screen
    safety is not guaranteed.
    """
    # imports specific to this test
    import sys
    import warnings
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    from psola.experiment_config import ExperimentConfig

    # get the data and do the estimation
    filename = sys.argv[1]    # filename is first command line arg
    cfg = ExperimentConfig()  # use default settings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore annoying WavFileWarning
        fs, data = wavfile.read(filename)
    pitch, t, strength = pitch_estimation(data, fs, cfg)

    # Plot estimated pitch and strength of pitch values
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,9))
    ax1.plot(t, pitch); ax1.set_title('Pitch v. Time'); ax1.set_ylabel('Freq (Hz)')
    ax2.plot(t, strength); ax2.set_title('Strength v. Time'); ax1.set_ylabel('Strength')
    plt.show()


if __name__ == "__main__":
    test()
