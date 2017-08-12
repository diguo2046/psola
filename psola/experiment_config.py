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

    def __init__(self, config_file=None, min_pitch=75, max_pitch=500,
                 frame_step=0.010, max_candidates=4,
                 voice_thresh=0.40, silence_thresh=0.05,
                 octave_cost=0.2**2):
        """
        initialize instance of experiment configuration,
        default values come from [1]

        if json file is used, the user needs to verify that
        key values match attribute names

        Args:
            config_file   (string): file path for json file with
                                      parameters as defined below
            min_pitch        (int): minimum pitch frequency (Hz)
            max_pitch        (int): maximum pitch frequency (Hz)
            frame_step     (float): frame step size (s)
            max_candidates   (int): maximum number of candidates for
                                      periodicity of frame
            voice_thresh   (float): this and silence_thresh used
                                      to estimate voiceless areas
            silence_thresh (float): this and voice_thresh used
                                      to estimate voiceless areas
            octave_cost    (float): see [1] page 105

        References:
            [1] Boersma, P. (1993). Accurate Short-term Analysis of the Fundamental Frequency
                and the Harmonics-to-Noise Ratio of a Samples Sound. IFA Proceedings Institute
                of Phonetic Sciences Proceedings, 17(17), 97–110.
        """

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.frame_step = frame_step
        self.max_candidates = max_candidates
        self.voice_thresh = voice_thresh
        self.silence_thresh = silence_thresh
        self.octave_cost = octave_cost

        # fill in user defined parameters from json file
        # (will clobber values above)
        if config_file is not None:
            with open(config_file, 'r') as f:
                config = json.load(f)
            for key, val in config.items():
                setattr(self, key, val)


if __name__ == "__main__":
    pass
