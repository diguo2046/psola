#!/usr/bin/env python
"""
psola.utilites.center_clipping

provides a function to do center clipping, i.e.,
 moving values *close* to zero to zero

Author: jreinhold
Created on: Aug 10, 2017
"""

import numpy as np


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


if __name__ == "__main__":
    pass
