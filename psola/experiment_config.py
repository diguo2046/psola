#!/usr/bin/env python
"""
psola.experiment_config

Module holds the ExperimentConfig object which defines parameters
specific to a specific experiment as defined by the user

Author: jreinhold
Created on: Aug 12, 2017
"""

import json


class ExperimentConfig(object):
    """
    Holds experiment specific parameters
    """

    def __init__(self, config_file=None,
                 min_pitch=75, max_pitch=500,
                 frame_step=0.010, dlog2p=1/48,
                 dERBs=1/20, window_overlap=0.5,
                 pitch_strength_thresh=0.2):
        """
        initialize instance of experiment configuration,
        default values come from code associated with [1]

        if json file is used, the user needs to verify that
        key values match attribute names

        Args:
            config_file          (string): file path for json file with
                                             parameters as defined below
            min_pitch               (int): minimum pitch frequency (Hz)
            max_pitch               (int): maximum pitch frequency (Hz)
            frame_step            (float): frame step size (s)
            dlog2p                (float): freq resolution, default 48 steps
                                             per octave
            dERBs                 (float): sample spectrum every `dERBs'th
                                             of an ERB
            window_overlap        (float): window overlap factor
            pitch_strength_thresh (float): discard samples with pitch strength
                                             lower than this value
        References:
            [1] Camacho, A., & Harris, J. G. (2008). A sawtooth waveform
                inspired pitch estimator for speech and music. The Journal
                of the Acoustical Society of America, 124(3), 1638–1652.
                https://doi.org/10.1121/1.2951592
        """

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.frame_step = frame_step
        self.dlog2p = dlog2p
        self.dERBs = dERBs
        self.window_overlap = window_overlap
        self.pitch_strength_thresh = pitch_strength_thresh

        # fill in user defined parameters from json file
        # (will clobber values above)
        if config_file is not None:
            with open(config_file, 'r') as f:
                config = json.load(f)
            for key, val in config.items():
                setattr(self, key, val)
