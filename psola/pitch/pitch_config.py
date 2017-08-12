#!/usr/bin/env python
"""
psola.pitch.pitch_config

object that holds data and parameters specific to pitch estimation,
marking, as well as experiment parameters

Author: jreinhold
Created on: Aug 12, 2017
"""

class Pitch(object):
    """
    Object to hold data, parameters, and methods relevant
    to estimating pitch and marking pitch periods, but
    should not explicitly relevant to those processes
    """

    def __init__(self, x, fs, experiment_config=None):
        """
        initializes instance of class with signal data and
        sample frequency

        Args:
            x         (array): signal data (array of real numbers)
            fs        (float): sample frequency (Hz)
            experiment_config (instance of ExperimentConfig)
        """
        # handling to make more standalone
        if experiment_config is None:
            from psola.experiment_config import ExperimentConfig
            self.config = ExperimentConfig()
        else:
            self.config = experiment_config

        self.x = x
        self.fs = fs

        # define frame parameters
        self.frame_size = 3 * (1 / self.config.min_pitch)  # from seconds to samples, 3 periods
        self.frame_step = self.config.frame_step * fs  # from seconds to samples

    def frames(self, x):
        """
        iterator to get frames sequentially
        (used in for loop)

        This could be put in __iter__, but this will more
        explicitly tell the reader what operation is
        performed in the for loop

        Args:
            x  (array): signal data (preprocessed or not, array of real numbers)

        Yields:
            frame of x; length frame_size
        """
        i = 0
        while i < x.size:
            yield x[i:i + self.frame_size]
            i += self.frame_step


if __name__ == "__main__":
    pass
