#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import pickle

from .utils import _get_model_filename


def load(model_name):
    """
    Load a pickled model. It first tries to load a TauPy internal model with
    the given name. Otherwise it is treated as a filename.
    """
    # Get the model filename in a unified manner.
    filename = _get_model_filename(model_name)
    # If that file does not exist, just treat it as a filename.
    if not os.path.exists(filename):
        filename = model_name

    with open(filename, 'rb') as f:
        return pickle.load(f)
