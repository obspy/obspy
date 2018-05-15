# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import unittest

from obspy.io.focmec.core import _is_focmec


class FOCMECTestCase(unittest.TestCase):
    """
    Test everything related to reading FOCMEC files
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.lst_file = os.path.join(self.path, 'data', 'focmec_8sta.lst')
        self.out_file = os.path.join(self.path, 'data', 'focmec_8sta.out')

    def test_is_focmec(self):
        for file_ in (self.lst_file, self.out_file):
            self.assertTrue(_is_focmec(file_))


def suite():
    return unittest.makeSuite(FOCMECTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
