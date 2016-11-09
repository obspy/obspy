#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The util test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest


class UtilTestCase(unittest.TestCase):
    """
    Test cases for L{obspy.signal.util}.
    """
    pass


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
