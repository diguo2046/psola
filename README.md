psola
=====

[![Build Status](https://travis-ci.org/jcreinhold/psola.svg?branch=master)](https://travis-ci.org/jcreinhold/psola)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/psola/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/psola?branch=master)
[![Code Climate](https://codeclimate.com/github/jcreinhold/psola/badges/gpa.svg)](https://codeclimate.com/github/jcreinhold/psola)
[![Issue Count](https://codeclimate.com/github/jcreinhold/psola/badges/issue_count.svg)](https://codeclimate.com/github/jcreinhold/psola)

Implements the TD-PSOLA algorithm.


Requirements
------------
- Numpy
- Matplotlib
- Scipy


Basic Usage
-----------

Install from PyPI

    pip install psola

or from the source directory

    python setup.py install

and use the command line script to interface with the package

    psola_exec ...
    
Project Structure
-----------------
```
psola
│
└───psola (source code)
│   │   experiment_config.py (define parameters for a run)
│   │   errors.py (project specific exceptions)
│   │   
│   └───exec (holds executables)
│   │   │   psola_exec (main script to run PSOLA implementation)
│   │   
│   └───pitch (modules for estimating and marking pitch)
│   │   │   estimation.py (estimates pitch for a speech time series)
│   │   │   mark.py (marks pitch periods)
│   │   │   pitch_config.py (holds data and experiemental params)
│   │
│   └───plot (plotting routines)
│   │
│   └───utilities (functions not explicitly part of PSOLA)
│   │   │   low_pass_filter.py (implements a standard LPF)
│   │   │   center_clipping.py (center clipping for spectral whitening)
│   │
│   └───voice (detecting voiced and unvoiced regions)
│   │   │   detection.py (segments voiced and unvoiced regions)
│
└───tests (unit tests)
│   
└───docs (documentation)
```
