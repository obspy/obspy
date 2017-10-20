# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy.io.iaspei.core import _read_ims10_bulletin


class IASPEITestCase(unittest.TestCase):
    """
    Test suite for obspy.io.iaspei.core
    """
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        # XXX the converted QuakeML file is not complete.. many picks and
        # station magnitudes were removed from it to save space.
        self.path_to_quakeml = os.path.join(data_dir, '19670130012028.xml')
        self.path_to_ims = os.path.join(data_dir, '19670130012028.isf')

    def test_serialize(self):
        """
        Test reading IMS10 bulletin format
        """
        cat = _read_ims10_bulletin(self.path_to_ims)
        self.assertEqual(len(cat), 1)


def suite():
    return unittest.makeSuite(IASPEITestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
