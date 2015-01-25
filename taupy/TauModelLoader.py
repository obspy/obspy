#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *

import os
import pickle

from .utils import _get_model_filename


def load(model_name):
    """
    Load a pickled model. It first tries to load a TauPy internal model with
    the given name. Otherwise it is treated as a filename.
    """
    #
    filename = _get_model_filename(model_name)
    if not os.path.exists(filename):
        filename = model_name

    with open(filename, 'rb') as f:
        return pickle.load(f)
