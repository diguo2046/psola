"""
psola.utilities.constants

Holds constants relevant to PSOLA package

Author: jreinhold
Created on: Aug 09, 2017
"""

# Define the range of speech pitch frequencies (Hz)
MIN_PITCH_FREQ = 60
MAX_PITCH_FREQ = 500

# Define frame size and step size (seconds)
PITCH_FRAME_SIZE = 3 * (1 / MIN_PITCH_FREQ)
HNR_FRAME_SIZE = 6 * (1 / MIN_PITCH_FREQ)
FRAME_STEP = 0.010

# Maximum pitch candidates per frame
MAX_CANDIDATE_FRAME = 4
