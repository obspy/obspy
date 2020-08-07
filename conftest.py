#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make sure the doctests run across numpy versions when using pytest.
"""
import numpy as np

try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass
