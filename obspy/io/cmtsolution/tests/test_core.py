#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import os
import unittest

from obspy.io.cmtsolution.core import is_cmtsolution, read_cmtsolution



class CmtsolutionTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.cmtsolution
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_read_cmtsolution(self):
        """
        Test reading cmtsolution files.
        """
        pass


def suite():
    return unittest.makeSuite(CmtsolutionTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
