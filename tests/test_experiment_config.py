#!/usr/bin/env python
"""
test.test_experiment_config

test ExperimentConfig class

Author: jreinhold
Created on: Aug 17, 2017
"""

import json
import tempfile
import unittest

from psola.experiment_config import ExperimentConfig


class TestExperimentConfig(unittest.TestCase):

    def setUp(self):
        self.config_file = dict(
            min_pitch=0, max_pitch=0,
            frame_step=0, dlog2p=0,
            dERBs=0, window_overlap=0,
            pitch_strength_thresh=0)
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w+')
        self.filename = self.tmpfile.name
        with open(self.filename, 'w') as f:
            json.dump(self.config_file, f)

    def test_experiment_config(self):
        """ test that experiment_config accepts json as expected """
        cfg = ExperimentConfig(config_file=self.filename)
        self.assertEqual(cfg.min_pitch, self.config_file['min_pitch'])
        self.assertEqual(cfg.max_pitch, self.config_file['max_pitch'])
        self.assertEqual(cfg.frame_step, self.config_file['frame_step'])
        self.assertEqual(cfg.dlog2p, self.config_file['dlog2p'])
        self.assertEqual(cfg.dERBs, self.config_file['dERBs'])
        self.assertEqual(cfg.window_overlap, self.config_file['window_overlap'])
        self.assertEqual(cfg.pitch_strength_thresh, self.config_file['pitch_strength_thresh'])

    def tearDown(self):
        self.tmpfile.close()  # this should delete the tmp file


if __name__ == '__main__':
    unittest.main()
