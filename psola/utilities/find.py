#!/usr/bin/env python
"""
psola.utilities.find

Implements a function that sort-of works like MATLAB's find
This is preferable to importing `find' from pylab, IMO

Author: jreinhold
Created on: Aug 18, 2017
"""

import numpy as np


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
