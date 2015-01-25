#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import sys


ROOT = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))


def _get_model_filename(model_name):
    """
    Get the pickled filename of a model. Depends on the Python version.

    :param model_name: The model name.
    """
    model_dir = os.path.join(ROOT, "data", "models")
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    filename = os.path.join(
        model_dir, model_name +
        ("__py%i%i__tvel" % sys.version_info[:2]) + os.path.extsep + "pickle")
    return filename
